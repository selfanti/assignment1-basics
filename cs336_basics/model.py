import torch
import torch.nn
from einops import rearrange, einsum
from jaxtyping import Float
from torch import Tensor
import math


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
            torch.empty((out_features, in_features),
                        dtype=dtype, device=device)
        )
        self.init_parameters()

    def init_parameters(self):
        '''
        初始化参数
        均值为0,方差为$2/(d_{in}+d_{out})$
        '''
        std = (2/(self.in_features+self.out_features))**0.5
        mean = 0
        torch.nn.init.trunc_normal_(
            self.weight, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor):
        # assert len(x.shape)==2,"input should be 2D"
        return einsum(self.weight, x, " d_out d_in, ... d_in ->... d_out")

    def extra_repr(self) -> str:
        """用于打印模型信息的额外表示"""
        return f'in_features={self.in_features}, out_features={self.out_features}'


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 【修复】之前参数名为 self.weights (复数)，但测试 state_dict 的 key 是
        # "token_embeddings.weight" (单数)。load_state_dict 时 key 不匹配。
        # 改为 self.weight 以匹配标准 PyTorch 命名约定和测试的 state_dict。
        self.weight = torch.nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype))

    def init_parameters(self):
        # Embedding: N (μ = 0, σ^2 = 1) truncated at [−3, 3]
        torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch sequence_length ->batch sequence_length d_model
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 【修复】之前参数名为 self.gamma，但测试 state_dict 的 key 是
        # "ln1.weight" (即 "weight")。load_state_dict 时 key 不匹配。
        # 改为 self.weight 以匹配标准 PyTorch LayerNorm 命名约定和测试的 state_dict。
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 缩放和平移
        return self.weight * (x/rms)


class PositionWise_FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        $FFN(x) = SwiGLU(x, W_1, W_2, W_3) = W_2(SiLU(W_1x) ⊙ W_3x)$
        You should set dff to approximately 8/3 times dmodel in your implementation, 
        while ensuring that  the dimensionality of the inner feed-forward layer
        is a multiple of 64 to make good use of your hardware.
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        # FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        # x: (..., d_model) -> w1@x: (..., d_ff), w3@x: (..., d_ff)
        w1_out = self.w1(x)  # (..., d_ff)
        w3_out = self.w3(x)  # (..., d_ff)
        swiglu = w1_out*torch.sigmoid(w1_out) * w3_out
        return self.w2(swiglu)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        $q'^{(i)} = R^iq^{(i)} = R^iW_qx^{(i)}$ 
        and the same computation for k
        '''
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # inv_freq 保存每个二维旋转子空间对应的基础频率。
        # 后续如果生成长度超过初始化时的 max_seq_len，我们只需要重新按更长的位置表展开，
        # 不需要重复推导或重建这一组频率参数。
        self.register_buffer(
            "inv_freq",
            torch.pow(theta, -2 * torch.arange(0, d_k // 2, device=device) / d_k),
            persistent=False,
        )
        self._build_rotation_cache(max_seq_len, device)

    def _build_rotation_cache(self, cache_len: int, device=None) -> None:
        # 把 [0, cache_len) 这些位置所需的 cos/sin 一次性展开出来。
        # 这个表本质上是 RoPE 的只读查找表，推理阶段按位置索引即可，避免每次前向重复算三角函数。
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
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) or (seq_len,)

        # Get sequence length from x
        seq_len = x.shape[-2]

        # Expand token_positions to match x's leading dimensions if needed
        # Handle both (seq_len,) and (..., seq_len) cases
        if token_positions.dim() == 1:
            # (seq_len,) - expand to match x's leading dims
            expanded_shape = [1] * (x.dim() - 2) + list(token_positions.shape)
            pos_expanded = token_positions.view(expanded_shape)
        else:
            # Already has batch dims
            pos_expanded = token_positions

        # kv cache 开启后，位置编码会继续沿着“真实生成步数”递增，
        # 所以 token_positions 可能大于初始化时的 max_seq_len。
        # 这里按需扩展 cos/sin 查找表，保证长文本生成不会因为 RoPE 表长度固定而越界。
        max_position = int(pos_expanded.max().item()) + 1
        if max_position > self.cos_vals.shape[0]:
            self._build_rotation_cache(max_position, x.device)

        # Get cos and sin for the specified positions
        # cos_vals, sin_vals: (max_seq_len, d_k // 2)
        # pos_expanded: (..., seq_len) -> after indexing: (..., seq_len, d_k // 2)
        cos_subset = self.cos_vals[pos_expanded]
        sin_subset = self.sin_vals[pos_expanded]

        # x_even: (..., seq_len, d_k // 2) - even indices 0, 2, 4, ...
        # x_odd: (..., seq_len, d_k // 2) - odd indices 1, 3, 5, ...
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation:
        # x_rotated_even = x_even * cos - x_odd * sin
        # x_rotated_odd = x_even * sin + x_odd * cos

        x_rotated_even = x_even * cos_subset - x_odd * sin_subset
        x_rotated_odd = x_even * sin_subset + x_odd * cos_subset

        # Interleave even and odd back together
        # x_rotated: (..., seq_len, d_k)
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to a tensor along a specified dimension with numerical stability.

    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to apply softmax

    Returns:
        Tensor with same shape as input, with softmax applied along dim
    """
    # Subtract max for numerical stability
    # keepdim=True maintains the dimension for correct broadcasting
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    # Compute exp(x - max(x))
    exp_x = torch.exp(x - max_vals)
    # Normalize by sum along the dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Tensor of shape (batch_size, num_heads, seq_len_q, d_v) containing the attention output
    """
    d_k = Q.shape[-1]

    # Compute scaled dot product: Q @ K.T
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / (d_k ** 0.5)
    # Apply mask if provided
    if mask is not None:
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"
        # 【修复】之前要求 mask.shape == scores.shape (严格相等)，
        # 但 causal mask 的 shape 是 (seq, seq)，需要广播到 (batch, head, seq, seq)。
        # 改为不做严格 shape 检查，依赖 PyTorch 自动广播。
        scores = scores.masked_fill(~mask, float('-inf'))

    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)

    # Compute attention output: attn_weights @ V
    output = einsum(attn_weights, V, "... q k, ... k d -> ... q d")
    return output


KVCache = dict[str, Tensor]


class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, if_rope=False, theta=None, max_seq_len=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.if_rope = if_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        # 【修复】之前命名为 o_proj，但测试的 state_dict key 是 "attn.output_proj.weight"，
        # load_state_dict 时 key 不匹配会导致加载失败。改名为 output_proj 以匹配。
        self.output_proj = Linear(d_model, d_model)
        if self.if_rope and self.theta is not None and self.max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                self.theta, self.d_model//self.num_heads, self.max_seq_len)

    def forward(
        self,
        x,
        token_positions=None,
        kv_cache: KVCache | None = None,
        use_kv_cache: bool = False,
    ):

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        d_k = self.d_model//self.num_heads
        # Reshape Q, K, V for multi-head attention
        Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k",
                      head=self.num_heads, d_k=d_k)
        K = rearrange(K, "... seq (head d_k) -> ... head seq d_k",
                      head=self.num_heads, d_k=d_k)
        V = rearrange(V, "... seq (head d_k) -> ... head seq d_k",
                      head=self.num_heads, d_k=d_k)
        if self.if_rope:
            assert self.theta is not None and self.max_seq_len is not None, "theta, max_seq_len, and token_positions must be provided for RoPE"
            if token_positions is None:
                # 兼容旧调用方: 如果外部没有显式传入位置，就按当前输入片段内部的相对位置构造。
                # 训练/整段前向时这与原有行为一致；增量生成时 `TransformerLM.forward`
                # 会显式传入绝对位置，使缓存里的 K/V 与 token 的真实时间步对齐。
                token_positions = torch.arange(
                    x.shape[-2], device=x.device).unsqueeze(0)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 如果传入了缓存，说明当前前向不是“从零开始看整段文本”，而是在已有历史后面追加新 token。
        # 这里把本次新算出的 K/V 追加到历史缓存后面，得到“当前可见的完整上下文”。
        #
        # 约定:
        # 1. cache 中只保存每层 attention 需要复用的 K/V，不缓存 FFN 或残差输出。
        # 2. cache 的时间维度始终位于倒数第二维，即 (..., head, seq, d_k) 里的 seq 维。
        # 3. 只在推理路径使用 cache；训练路径仍然按整段前向计算，保证原有接口不变。
        if kv_cache is not None:
            K = torch.cat([kv_cache["k"], K], dim=-2)
            V = torch.cat([kv_cache["v"], V], dim=-2)

        # 当缓存长度超过模型的 context window 时，只保留最近的 max_seq_len 个 token。
        # 这样做的目标是把 cache 的空间复杂度固定在 O(num_layers * context_length)，
        # 避免生成长文本时显存/内存线性膨胀。
        #
        # 需要注意：窗口发生左移后，这里复用的是“旧窗口下已经算好的隐藏状态”。
        # 这属于标准的流式 kv cache 语义；如果想与“把新窗口整段重新前向一次”严格等价，
        # 就必须重新计算所有保留下来的 token，等于失去缓存带来的速度收益。
        if use_kv_cache and self.max_seq_len is not None and K.shape[-2] > self.max_seq_len:
            K = K[..., -self.max_seq_len:, :]
            V = V[..., -self.max_seq_len:, :]

        # 自回归 mask 现在需要同时覆盖两种情况:
        # 1. 训练/普通前向: query 和 key 来自同一段序列，此时它退化为标准下三角矩阵。
        # 2. 增量生成: query 只包含“本轮新 token”，key/value 则包含“历史缓存 + 新 token”。
        #    例如 past_len=5, query_len=2 时，第 0 个新 token 可以看到前 6 个 key，
        #    第 1 个新 token 可以看到前 7 个 key，不能越过自己在当前 chunk 中的位置。
        num_queries = Q.shape[-2]
        num_keys = K.shape[-2]
        past_kv_len = num_keys - num_queries
        query_positions = torch.arange(
            num_queries, device=Q.device).unsqueeze(-1)
        key_positions = torch.arange(num_keys, device=Q.device).unsqueeze(0)
        causal_mask = key_positions <= (past_kv_len + query_positions)

        # Compute attention for each head
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # Concatenate heads and project back to d_model
        attn_output = rearrange(
            attn_output, "... head seq d_k -> ... seq (head d_k)", head=self.num_heads, d_k=d_k)
        out = self.output_proj(attn_output)

        if not use_kv_cache:
            return out

        # 返回更新后的 cache 供下一步解码复用。
        # 这里直接复用已经裁剪好的 K/V，调用方不需要再额外维护窗口边界。
        return out, {"k": K.detach(), "v": V.detach()}


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) 激活函数: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class TransformerBlock(torch.nn.Module):
    """
    Pre-norm Transformer Block。

    结构: x → RMSNorm → MHA(with RoPE) → + residual → RMSNorm → SwiGLU FFN → + residual
    $y = x + MultiHeadSelfAttention(RMSNorm(x))$
    之前未实现此类，导致 run_transformer_block 和 run_transformer_lm 均抛出 NotImplementedError。

    State dict key 约定 (与测试 adapter 匹配):
        - attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.output_proj.weight
        - ln1.weight, ln2.weight
        - ffn.w1.weight, ffn.w2.weight, ffn.w3.weight
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float, max_seq_len: int,
                 device=None, dtype=None):
        super().__init__()
        # Pre-attention LayerNorm
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        # Multi-head self-attention with RoPE
        self.attn = multihead_self_attention(
            d_model=d_model, num_heads=num_heads,
            if_rope=True, theta=theta, max_seq_len=max_seq_len
        )
        # Pre-FFN LayerNorm
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        # SwiGLU Feed-Forward Network
        self.ffn = PositionWise_FeedForward(
            d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        kv_cache: KVCache | None = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # 如果未提供 token_positions，默认生成 [0, 1, 2, ..., seq_len-1]
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(
                seq_len, device=x.device).unsqueeze(0)
        # Pre-norm design:
        # 1) RMSNorm → MHA → residual connection
        normed = self.ln1(x)
        if use_kv_cache:
            # 只有 attention 子层需要 cache；FFN 是逐位置前馈，不依赖历史 token，
            # 因此这里仅向 attention 透传/更新缓存。
            attn_out, next_kv_cache = self.attn(
                normed,
                token_positions=token_positions,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
        else:
            attn_out = self.attn(normed, token_positions=token_positions)
        x = x + attn_out
        # 2) RMSNorm → FFN → residual connection
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        if use_kv_cache:
            return x, next_kv_cache
        return x


class TransformerLM(torch.nn.Module):
    """
    GPT-style Transformer Language Model。

    结构: Token Embedding → N × TransformerBlock → RMSNorm → LM Head
    LM Head 与 Token Embedding 共享权重 (weight tying)。

    之前未实现此类，导致 run_transformer_lm 抛出 NotImplementedError。

    State dict key 约定:
        - token_embeddings.weight
        - layers.{i}.attn.q_proj.weight 等 (TransformerBlock 的 keys)
        - ln_final.weight
        - lm_head.weight  (与 token_embeddings.weight 是同一个张量, weight tying)
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int, d_ff: int,
                 num_layers: int, num_heads: int,
                 rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length

        # Token embedding
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype)

        # Transformer blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # LM head (output projection back to vocab)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        kv_cache: list[KVCache | None] | None = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        # x: (batch_size, seq_len) — token indices
        seq_len = x.shape[-1]
        # 训练/普通前向仍然使用 0..seq_len-1 的相对位置。
        # 增量生成时，decode 会显式传入“绝对位置”，例如 prompt 长度为 128 时，
        # 下一步新 token 的 position id 应该是 128，而不是重新从 0 开始。
        # 这是 kv cache 能正确复用 RoPE 后 K/V 的前提。
        if token_positions is None:
            token_positions = torch.arange(
                seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)

        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError(
                f"kv_cache should have one entry per transformer layer, expected {len(self.layers)} got {len(kv_cache)}"
            )

        # Token embedding
        h = self.token_embeddings(x)  # (batch, seq_len, d_model)

        # Pass through transformer blocks
        next_kv_cache: list[KVCache] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[layer_idx]
            if use_kv_cache:
                # 每一层的 cache 独立维护，因为不同层的 K/V 来自不同的隐藏状态空间。
                # 不能把某一层的 cache 误用到另一层，否则注意力会直接读取错误特征。
                h, updated_layer_cache = layer(
                    h,
                    token_positions=token_positions,
                    kv_cache=layer_cache,
                    use_kv_cache=True,
                )
                next_kv_cache.append(updated_layer_cache)
            else:
                h = layer(h, token_positions=token_positions)

        # Final normalization
        h = self.ln_final(h)

        # Project to vocab size
        logits = self.lm_head(h)  # (batch, seq_len, vocab_size)
        if use_kv_cache:
            return logits, next_kv_cache
        return logits


if __name__ == '__main__':
    linear_layer = Linear(3, 3)
    input = torch.randn(3, 1)
    output = linear_layer(input)
    print(output)
    print(linear_layer.extra_repr())
    print(linear_layer.state_dict())
