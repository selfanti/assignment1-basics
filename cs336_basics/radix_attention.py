from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .model import KVCache, softmax


LayerKVCache = list[KVCache]


@dataclass
class RadixAttentionNode:
    """
    单个 radix 节点。

    这里按“一个 token 一条边”来组织前缀树。真实线上系统通常会做 path compression
    来减少节点数，但当前需求是“单用户多轮交互的测试实现”，并且用户已经明确给了
    “上下文永远不会超过训练时设定窗口”的假设，因此使用更直接、更容易审计的结构
    更合适。
    """

    depth: int
    token_id: int | None = None
    parent: "RadixAttentionNode | None" = None
    children: dict[int, "RadixAttentionNode"] = field(default_factory=dict)
    kv_cache: LayerKVCache | None = None


@dataclass(frozen=True)
class RadixPrefixMatch:
    """最长前缀匹配结果。"""

    matched_length: int
    kv_cache: LayerKVCache | None


@dataclass(frozen=True)
class RadixDecodeResult:
    """
    交互式生成结果。

    `reused_prefix_length` 直接告诉调用方本轮从 radix tree 里复用了多少 prompt token 的缓存，
    这样脚本打印调试信息时不需要再自己推断命中长度。
    """

    tokens: Tensor
    generated_tokens: Tensor
    reused_prefix_length: int


def clone_kv_cache(kv_cache: LayerKVCache | None) -> LayerKVCache | None:
    """
    对每层 K/V 做深拷贝，保证 radix tree 里不同前缀节点不会共享同一块可变 tensor 视图。

    这里选择 clone 而不是只保存引用，原因是生成过程中 cache 会持续追加新 token；
    如果多个前缀节点共享同一个底层张量，后续写入会把“旧前缀快照”一起污染掉，
    最后 longest-prefix 命中到的就不再是真正属于那个前缀的缓存。
    """
    if kv_cache is None:
        return None

    return [
        {
            "k": layer_cache["k"].detach().clone(),
            "v": layer_cache["v"].detach().clone(),
        }
        for layer_cache in kv_cache
    ]


class RadixAttentionCache:
    """
    使用 radix tree 管理多轮对话中的 KV cache。

    设计目标很明确:
    1. 每轮构造完整对话 prompt 时，先在树里找最长公共前缀。
    2. 只对“未命中的新增 token”重新前向，把新得到的 cache 挂回对应前缀节点。
    3. 因为当前只做单用户测试实现，所以不额外引入 eviction、引用计数、分页显存等复杂机制。
    """

    def __init__(self, context_length: int | None) -> None:
        self.context_length = context_length
        self.root = RadixAttentionNode(depth=0)

    @staticmethod
    def _normalize_token_ids(token_ids: Tensor | list[int]) -> list[int]:
        if isinstance(token_ids, Tensor):
            if token_ids.dim() != 1:
                raise ValueError(
                    f"RadixAttentionCache expects a 1D token tensor, got shape {tuple(token_ids.shape)}"
                )
            return token_ids.tolist()
        return list(token_ids)

    def clear(self) -> None:
        """重置整棵 radix tree，用于开始一段全新的对话。"""
        self.root = RadixAttentionNode(depth=0)

    def insert(self, token_ids: Tensor | list[int], kv_cache: LayerKVCache) -> None:
        """
        把“某个完整前缀”对应的缓存挂到树上。

        调用约定是: `token_ids` 必须精确对应 `kv_cache` 当前表示的前缀长度。
        例如 `token_ids` 长度为 12，那么每层 cache 的时间维也应该正好覆盖这 12 个 token。
        """
        normalized_ids = self._normalize_token_ids(token_ids)
        if self.context_length is not None and len(normalized_ids) > self.context_length:
            raise ValueError(
                "The radix-attention demo assumes the conversation never exceeds the trained context length; "
                f"got prefix length {len(normalized_ids)} > context_length {self.context_length}."
            )

        node = self.root
        for depth, token_id in enumerate(normalized_ids, start=1):
            child = node.children.get(token_id)
            if child is None:
                child = RadixAttentionNode(depth=depth, token_id=token_id, parent=node)
                node.children[token_id] = child
            node = child

        node.kv_cache = clone_kv_cache(kv_cache)

    def match(self, token_ids: Tensor | list[int]) -> RadixPrefixMatch:
        """
        返回给定 token 序列的最长缓存前缀。

        这个方法用于“看树里目前最多能复用到哪里”。对于生成任务本身，通常还会把结果再退一位，
        因为继续采样下一个 token 时至少需要重新前向 prompt 的最后一个 token 来拿到 logits。
        """
        normalized_ids = self._normalize_token_ids(token_ids)

        node = self.root
        matched_length = 0
        for token_id in normalized_ids:
            next_node = node.children.get(token_id)
            if next_node is None:
                break
            node = next_node
            matched_length += 1

        kv_cache = None if matched_length == 0 else clone_kv_cache(node.kv_cache)
        return RadixPrefixMatch(matched_length=matched_length, kv_cache=kv_cache)

    def get_generation_match(self, token_ids: Tensor | list[int]) -> RadixPrefixMatch:
        """
        返回适合“继续生成”的最长可复用前缀。

        如果整段 prompt 已经完全命中树，我们会把复用长度回退 1 个 token。
        原因是语言模型需要“最后一个 prompt token 的 logits”才能采样下一个 token；
        仅有完整 prompt 的 cache 而不重新跑最后一个 token，本轮就拿不到首个采样 logits。
        """
        normalized_ids = self._normalize_token_ids(token_ids)
        if not normalized_ids:
            return RadixPrefixMatch(matched_length=0, kv_cache=None)

        node = self.root
        matched_nodes: list[RadixAttentionNode] = []
        for token_id in normalized_ids:
            next_node = node.children.get(token_id)
            if next_node is None:
                break
            node = next_node
            matched_nodes.append(node)

        matched_length = len(matched_nodes)
        if matched_length == len(normalized_ids):
            matched_length -= 1

        if matched_length <= 0:
            return RadixPrefixMatch(matched_length=0, kv_cache=None)

        matched_node = matched_nodes[matched_length - 1]
        return RadixPrefixMatch(
            matched_length=matched_length,
            kv_cache=clone_kv_cache(matched_node.kv_cache),
        )


def sample_next_token(logits: Tensor, temperature: float | None = None, top_p: float | None = None) -> Tensor:
    """
    从单步 logits 里采样下一个 token。

    这段逻辑与 `decode(...)` 保持一致，目的是让“普通生成路径”和“radix 复用路径”在采样策略上
    完全对齐；否则 benchmark 或交互行为差异会混入采样实现差异，难以判断到底是谁造成了输出变化。
    """
    if temperature is not None and temperature > 0:
        logits = logits / temperature

    probs = softmax(logits, dim=-1)

    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        mask = cumulative_probs > top_p
        mask[1:] = mask[:-1].clone()
        mask[0] = False

        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-8)
        return sorted_indices[torch.multinomial(sorted_probs, 1)]

    return torch.multinomial(probs, 1)


def generate_with_radix_attention(
    model: torch.nn.Module,
    prompt: Tensor,
    radix_cache: RadixAttentionCache,
    max_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_token_id: int = 0,
) -> RadixDecodeResult:
    """
    基于 radix tree 复用前缀 KV cache，并完成一轮自回归生成。

    当前实现特意选择“逐 token prefill + 逐 token decode”的直白写法，原因有两点:
    1. 单用户多轮测试里，可读性和可验证性比极致 prefill 吞吐更重要。
    2. 逐 token 更新能自然把每个中间前缀的 cache 都挂到 radix tree 上，后续 turn 就能做更细粒度复用。

    同时这里严格执行用户给出的假设: 当前项目的对话上下文不会超过训练窗口；
    一旦超限，直接抛出异常而不是偷偷滑窗截断，因为截断会改变“多轮对话历史完整保留”的语义。
    """
    device = next(model.parameters()).device
    prompt_tokens = prompt.clone().to(device)
    if prompt_tokens.dim() != 1:
        raise ValueError(
            f"generate_with_radix_attention expects a 1D prompt tensor of token ids, got shape {tuple(prompt_tokens.shape)}"
        )
    if prompt_tokens.numel() == 0:
        raise ValueError("The interactive radix-attention demo requires a non-empty prompt.")

    context_length = getattr(model, "context_length", None)
    if context_length is not None and prompt_tokens.shape[0] > context_length:
        raise ValueError(
            "The radix-attention demo assumes the conversation never exceeds the trained context length; "
            f"got prompt length {prompt_tokens.shape[0]} > context_length {context_length}."
        )

    match = radix_cache.get_generation_match(prompt_tokens)
    kv_cache = match.kv_cache
    last_logits: Tensor | None = None
    tokens = prompt_tokens.clone()

    with torch.no_grad():
        for token_index in range(match.matched_length, prompt_tokens.shape[0]):
            token_slice = prompt_tokens[token_index: token_index + 1].unsqueeze(0)
            token_positions = torch.tensor([[token_index]], device=device)
            try:
                last_logits, kv_cache = model(
                    token_slice,
                    token_positions=token_positions,
                    kv_cache=kv_cache,
                    use_kv_cache=True,
                )
            except TypeError as exc:
                raise TypeError(
                    "generate_with_radix_attention requires a model whose forward(...) accepts "
                    "token_positions, kv_cache, and use_kv_cache."
                ) from exc

            radix_cache.insert(prompt_tokens[: token_index + 1], kv_cache)

        if last_logits is None:
            raise RuntimeError(
                "The radix-attention prefill phase did not produce logits. "
                "This usually means the reusable-prefix calculation is inconsistent with the prompt length."
            )

        generated_token_ids: list[int] = []
        for _ in range(max_tokens):
            next_token = sample_next_token(
                last_logits[:, -1, :].squeeze(0),
                temperature=temperature,
                top_p=top_p,
            ).squeeze(0)

            if int(next_token.item()) == stop_token_id:
                break

            generated_token_ids.append(int(next_token.item()))
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])

            if context_length is not None and tokens.shape[0] > context_length:
                raise ValueError(
                    "The radix-attention demo assumes the conversation never exceeds the trained context length; "
                    f"got generated sequence length {tokens.shape[0]} > context_length {context_length}."
                )

            token_positions = torch.tensor([[tokens.shape[0] - 1]], device=device)
            last_logits, kv_cache = model(
                next_token.view(1, 1),
                token_positions=token_positions,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
            radix_cache.insert(tokens, kv_cache)

    generated_tokens = torch.tensor(generated_token_ids, dtype=torch.long, device=device)
    return RadixDecodeResult(
        tokens=tokens,
        generated_tokens=generated_tokens,
        reused_prefix_length=match.matched_length,
    )
