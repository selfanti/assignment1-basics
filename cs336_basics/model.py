import math
from typing import Literal

import torch
import torch.nn
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        自定义线性层

        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            device: 设备 (CPU/GPU)
            dtype: 数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        self.init_parameters()

    def init_parameters(self):
        """
        初始化参数
        均值为0,方差为 2 / (d_in + d_out)
        """
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 参数名必须叫 weight，才能与测试和 PyTorch 约定对齐。
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

    def init_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 参数名必须叫 weight，才能与测试和 PyTorch 约定对齐。
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class PositionWise_FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        FFN(x) = W2(SiLU(W1x) ⊙ W3x)
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        swiglu = w1_out * torch.sigmoid(w1_out) * w3_out
        return self.w2(swiglu)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # inv_freq 是 RoPE 的核心频率表。后续即便需要扩展位置缓存，
        # 也只需要基于这组频率重新展开 cos/sin，而不需要重建整套模块状态。
        self.register_buffer(
            "inv_freq",
            torch.pow(theta, -2 * torch.arange(0, d_k // 2, device=device) / d_k),
            persistent=False,
        )
        self._build_rotation_cache(max_seq_len, device)

    def _build_rotation_cache(self, cache_len: int, device=None) -> None:
        positions = torch.arange(cache_len, device=device or self.inv_freq.device)
        freqs_expo = einsum(self.inv_freq, positions, "d2, seq -> seq d2")

        cos_vals = torch.cos(freqs_expo)
        sin_vals = torch.sin(freqs_expo)
        if hasattr(self, "cos_vals"):
            self.cos_vals = cos_vals
            self.sin_vals = sin_vals
        else:
            self.register_buffer("cos_vals", cos_vals, persistent=False)
            self.register_buffer("sin_vals", sin_vals, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions.dim() == 1:
            expanded_shape = [1] * (x.dim() - 2) + list(token_positions.shape)
            pos_expanded = token_positions.view(expanded_shape)
        else:
            pos_expanded = token_positions

        max_position = int(pos_expanded.max().item()) + 1
        if max_position > self.cos_vals.shape[0]:
            self._build_rotation_cache(max_position, x.device)

        cos_subset = self.cos_vals[pos_expanded]
        sin_subset = self.sin_vals[pos_expanded]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated_even = x_even * cos_subset - x_odd * sin_subset
        x_rotated_odd = x_even * sin_subset + x_odd * cos_subset

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / (d_k ** 0.5)
    if mask is not None:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, "... q k, ... k d -> ... q d")


AttentionBackend = Literal["standard", "flash_attention_v2"]


def validate_attention_backend(attention_backend: str) -> AttentionBackend:
    valid_backends = {"standard", "flash_attention_v2"}
    if attention_backend not in valid_backends:
        raise ValueError(
            f"attention_backend must be one of {sorted(valid_backends)}, got {attention_backend!r}"
        )
    return attention_backend  # type: ignore[return-value]


def flash_attention_v2(Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
    """
    手写 Triton FlashAttention v2 路径的 Python 入口。

    这里不再借 PyTorch SDPA 走现成 fused 实现，而是显式调用我们自己写的 Triton kernel。
    目前这条路径只覆盖本项目真正使用的 causal self-attention 语义，
    也就是训练整段前向和 kv cache 增量解码这两种场景。
    """
    from cs336_basics.triton_flash_attention import triton_flash_attention_v2

    if Q.device.type != "cuda":
        raise ValueError("Manual Triton FlashAttention v2 is only available on CUDA devices.")
    if mask is not None:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
    return triton_flash_attention_v2(Q, K, V)


KVCache = dict[str, Tensor]


class multihead_self_attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        if_rope=False,
        theta=None,
        max_seq_len=None,
        attention_backend: AttentionBackend = "standard",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.if_rope = if_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        # attention_backend 是模型结构配置的一部分，而不是临时运行时 flag。
        # 这样 benchmark、训练和交互脚本都能通过同一初始化参数明确选择后端。
        self.attention_backend = validate_attention_backend(attention_backend)
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if self.if_rope and self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_model // self.num_heads, self.max_seq_len)

    def forward(
        self,
        x,
        token_positions=None,
        kv_cache: KVCache | None = None,
        use_kv_cache: bool = False,
    ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        d_k = self.d_model // self.num_heads

        q = rearrange(q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        k = rearrange(k, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)
        v = rearrange(v, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads, d_k=d_k)

        if self.if_rope:
            assert self.theta is not None and self.max_seq_len is not None
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device).unsqueeze(0)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2)
            v = torch.cat([kv_cache["v"], v], dim=-2)

        if use_kv_cache and self.max_seq_len is not None and k.shape[-2] > self.max_seq_len:
            k = k[..., -self.max_seq_len:, :]
            v = v[..., -self.max_seq_len:, :]

        num_queries = q.shape[-2]
        num_keys = k.shape[-2]
        past_kv_len = num_keys - num_queries
        query_positions = torch.arange(num_queries, device=q.device).unsqueeze(-1)
        key_positions = torch.arange(num_keys, device=q.device).unsqueeze(0)
        causal_mask = key_positions <= (past_kv_len + query_positions)

        if self.attention_backend == "flash_attention_v2":
            attn_output = flash_attention_v2(q, k, v, mask=causal_mask)
        else:
            attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        attn_output = rearrange(attn_output, "... head seq d_k -> ... seq (head d_k)", head=self.num_heads, d_k=d_k)
        out = self.output_proj(attn_output)

        if not use_kv_cache:
            return out
        return out, {"k": k.detach(), "v": v.detach()}


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float, max_seq_len: int,
                 device=None, dtype=None,
                 attention_backend: AttentionBackend = "standard"):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            if_rope=True,
            theta=theta,
            max_seq_len=max_seq_len,
            attention_backend=attention_backend,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionWise_FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        kv_cache: KVCache | None = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        normed = self.ln1(x)
        if use_kv_cache:
            attn_out, next_kv_cache = self.attn(
                normed,
                token_positions=token_positions,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
        else:
            attn_out = self.attn(normed, token_positions=token_positions)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        if use_kv_cache:
            return x, next_kv_cache
        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, d_ff: int,
                 num_layers: int, num_heads: int,
                 rope_theta: float, device=None, dtype=None,
                 attention_backend: AttentionBackend = "standard"):
        super().__init__()
        self.context_length = context_length
        self.attention_backend = validate_attention_backend(attention_backend)

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope_theta,
                context_length,
                device=device,
                dtype=dtype,
                attention_backend=self.attention_backend,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        kv_cache: list[KVCache | None] | None = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        seq_len = x.shape[-1]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError(
                f"kv_cache should have one entry per transformer layer, expected {len(self.layers)} got {len(kv_cache)}"
            )

        h = self.token_embeddings(x)
        next_kv_cache: list[KVCache] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            if use_kv_cache:
                h, updated_layer_cache = layer(
                    h,
                    token_positions=token_positions,
                    kv_cache=layer_cache,
                    use_kv_cache=True,
                )
                next_kv_cache.append(updated_layer_cache)
            else:
                h = layer(h, token_positions=token_positions)

        h = self.ln_final(h)
        logits = self.lm_head(h)
        if use_kv_cache:
            return logits, next_kv_cache
        return logits


if __name__ == '__main__':
    linear_layer = Linear(3, 3)
    input = torch.randn(3, 1)
    output = linear_layer(input)
    print(output)
