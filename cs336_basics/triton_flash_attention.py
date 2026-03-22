from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _flash_attention_v2_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_len,
    k_len,
    head_dim,
    sm_scale,
    stride_qb,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    手写 Triton FlashAttention v2 风格前向 kernel。

    这个 kernel 覆盖当前项目真正需要的场景:
    1. Q/K/V 都是连续的 `[batch_heads, seq, head_dim]` 布局。
    2. attention mask 固定为 causal，并且允许 `k_len > q_len`，从而兼容 kv cache 追加场景。
    3. 采用 online softmax，把整个 K/V 轴按 block 流式扫过，避免显式物化完整注意力矩阵。
    """
    block_m = tl.program_id(0)
    batch_head = tl.program_id(1)

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + batch_head * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    past_len = k_len - q_len
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for start_n in range(0, k_len, BLOCK_N):
        current_n = start_n + offs_n

        k_ptrs = k_ptr + batch_head * stride_kb + offs_d[:, None] * stride_kd + current_n[None, :] * stride_kn
        k_mask = (offs_d[:, None] < head_dim) & (current_n[None, :] < k_len)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, k) * sm_scale
        causal_mask = current_n[None, :] <= (past_len + offs_m[:, None])
        valid_mask = (offs_m[:, None] < q_len) & (current_n[None, :] < k_len) & causal_mask
        scores = tl.where(valid_mask, scores, -float("inf"))

        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_i_new)

        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + batch_head * stride_vb + current_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (current_n[:, None] < k_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = acc + tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = m_i_new

    out = acc / l_i[:, None]
    out_ptrs = out_ptr + batch_head * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, out, mask=out_mask)


def reference_causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    参考实现同时承担两个职责:
    1. Triton kernel 目前只负责 forward，这个函数用于 backward 里的重计算求梯度。
    2. 当输入不满足 Triton kernel 约束时，可以作为最后的正确性保底路径。
    """
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    past_len = k_len - q_len
    query_positions = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)
    key_positions = torch.arange(k_len, device=q.device).view(1, 1, 1, k_len)
    causal_mask = key_positions <= (past_len + query_positions)

    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class TritonFlashAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.device.type != "cuda":
            raise ValueError("Manual Triton FlashAttention v2 is only available on CUDA devices.")
        if q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported dtype for Triton FlashAttention v2: {q.dtype}")

        # Triton kernel 直接处理 [batch_heads, seq, head_dim]，
        # 所以这里先把 batch/head 两个维度拍平，减少 kernel 的索引复杂度。
        q_contiguous = q.contiguous()
        k_contiguous = k.contiguous()
        v_contiguous = v.contiguous()
        batch_size, num_heads, q_len, head_dim = q_contiguous.shape
        k_len = k_contiguous.shape[-2]
        batch_heads = batch_size * num_heads

        q_flat = q_contiguous.reshape(batch_heads, q_len, head_dim)
        k_flat = k_contiguous.reshape(batch_heads, k_len, head_dim)
        v_flat = v_contiguous.reshape(batch_heads, k_len, head_dim)
        out_flat = torch.empty_like(q_flat)

        block_d = triton.next_power_of_2(head_dim)
        block_m = 64
        block_n = 64
        grid = (triton.cdiv(q_len, block_m), batch_heads)
        sm_scale = 1.0 / math.sqrt(head_dim)

        _flash_attention_v2_forward_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            q_len,
            k_len,
            head_dim,
            sm_scale,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            out_flat.stride(0),
            out_flat.stride(1),
            out_flat.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q_contiguous, k_contiguous, v_contiguous)
        return out_flat.reshape(batch_size, num_heads, q_len, head_dim)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k, v = ctx.saved_tensors

        # 这里没有偷用现成 fused backward kernel。
        # 相反，我们显式用 reference attention 重新前向一次，再借 PyTorch autograd 求梯度，
        # 从而保证“前向路径”确实走的是手写 Triton kernel，同时训练/benchmark 仍然可运行。
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            ref_out = reference_causal_attention(q_ref, k_ref, v_ref)
            dq, dk, dv = torch.autograd.grad(ref_out, (q_ref, k_ref, v_ref), grad_out)
        return dq, dk, dv


def triton_flash_attention_v2(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return TritonFlashAttentionV2.apply(q, k, v)
