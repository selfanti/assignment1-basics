import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import decode


def test_transformer_lm_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=12,
        d_ff=64,
        num_layers=2,
        num_heads=3,
        rope_theta=10000.0,
    )
    model.eval()

    tokens = torch.tensor([[1, 5, 7, 2, 9]])

    # 这条测试验证最核心的不变量:
    # “逐 token + kv cache” 的输出，必须与“整段一次性前向”逐位置完全一致。
    # 如果这里不一致，说明缓存后的 attention 读到了错误的历史 K/V，
    # 或者 RoPE/causal mask 在增量路径上的位置对齐出了问题。
    full_logits = model(tokens)

    cache = None
    cached_logits = []
    for position in range(tokens.shape[1]):
        step_logits, cache = model(
            tokens[:, position: position + 1],
            token_positions=torch.tensor([[position]]),
            kv_cache=cache,
            use_kv_cache=True,
        )
        cached_logits.append(step_logits)

    cached_logits = torch.cat(cached_logits, dim=1)
    torch.testing.assert_close(cached_logits, full_logits, atol=1e-5, rtol=1e-5)


def test_transformer_lm_kv_cache_keeps_bounded_cache_after_context_window():
    torch.manual_seed(1)
    context_length = 4
    model = TransformerLM(
        vocab_size=32,
        context_length=context_length,
        d_model=12,
        d_ff=64,
        num_layers=2,
        num_heads=3,
        rope_theta=10000.0,
    )
    model.eval()

    tokens = torch.tensor([[3, 1, 4, 1, 5, 9]])
    cache = None

    # 这条测试覆盖“历史长度超过 context_length”后的真实缓存契约:
    # 1. RoPE 位置表会按需扩展，不会因为 position id 超过初始窗口而越界。
    # 2. 每层 cache 都会被裁剪到最近 context_length 个 token，空间占用保持有界。
    #
    # 这里不要求与“整段重算滑动窗口”逐 logit 完全一致。
    # 原因是窗口左移后，若想重新得到所有保留 token 的精确隐藏状态，
    # 必须把整个窗口再前向一遍，这会抵消 kv cache 的主要收益。
    for position in range(tokens.shape[1]):
        step_logits, cache = model(
            tokens[:, position: position + 1],
            token_positions=torch.tensor([[position]]),
            kv_cache=cache,
            use_kv_cache=True,
        )
        assert step_logits.shape == (1, 1, 32)

        for layer_cache in cache:
            assert layer_cache["k"].shape[-2] <= context_length
            assert layer_cache["v"].shape[-2] <= context_length


class _ToyCacheAwareModel(torch.nn.Module):
    def __init__(self, context_length: int = 4, vocab_size: int = 8):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.call_lengths: list[int] = []
        self.call_count = 0

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        kv_cache=None,
        use_kv_cache: bool = False,
    ):
        self.call_lengths.append(int(x.shape[1]))
        self.call_count += 1

        logits = torch.full(
            (x.shape[0], x.shape[1], self.vocab_size),
            -1e9,
            device=x.device,
        )

        # 通过固定输出序列 [5, 6, 0] 让 decode 走完两轮增量生成后停止。
        # 这里不关心语言模型质量，只关心 decode 是否真的在复用 cache。
        next_token = 0 if self.call_count >= 3 else self.call_count + 4
        logits[:, -1, next_token] = 0.0

        if not use_kv_cache:
            return logits

        previous_length = 0 if kv_cache is None else kv_cache[0]["k"].shape[-2]
        total_length = min(self.context_length, previous_length + x.shape[1])
        next_cache = [{
            "k": torch.zeros(1, 1, total_length, 1, device=x.device),
            "v": torch.zeros(1, 1, total_length, 1, device=x.device),
        }]
        return logits, next_cache


def test_decode_uses_kv_cache_incrementally():
    model = _ToyCacheAwareModel(context_length=4)
    prompt = torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)

    # prompt 比 context_length 更长，用它可以同时验证:
    # 1. 首轮 prefill 只会送最近窗口。
    # 2. 之后每一轮都只送 1 个新 token，而不是重复送整段历史。
    output = decode(model, prompt, max_tokens=5, temperature=1.0, top_p=None)

    assert output.tolist() == [11, 12, 13, 14, 15, 5, 6]
    assert model.call_lengths == [4, 1, 1]
