import torch
from .model import softmax
import torch.nn.functional as F


def cross_entropy(pres: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss.

    li = -log(softmax(oi)[xi+1])

    For numerical stability, we use the log-sum-exp trick:
    - Subtract max for numerical stability
    - Cancel out log and exp: 
      -log(exp(xi)/sum(exp(xj))) = -xi + log(sum(exp(xj)))

    Args:
        pres: Predicted logits with shape (..., vocab_size)
        targets: Target indices with shape (...)

    Returns:
        Average cross entropy loss across the batch (scalar tensor)
    """
    # Get the max logit for numerical stability (subtract largest element)
    # keepdim=True for broadcasting
    max_logits = torch.max(pres, dim=-1, keepdim=True)[0]

    # Subtract max from logits for stability
    shifted_logits = pres - max_logits

    # Compute log(sum(exp(shifted_logits)))
    # This is the log-sum-exp term
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # Gather the shifted logit at the target index for each example
    # shifted_logits: (..., vocab_size), targets: (...)
    # We want to get shifted_logits[..., targets[...]]
    target_shifted_logits = shifted_logits.gather(
        dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Cross entropy = -target_shifted_logits + log_sum_exp
    # = -log(exp(target_shifted_logits) / sum(exp(shifted_logits)))
    # = -log(softmax(shifted_logits)[target])
    losses = -target_shifted_logits + log_sum_exp

    # Return the average across the batch
    return losses.mean()

# def decode(model:torch.nn.Module,prompt:torch.Tensor,vocab_size=10000,max_tokens=2048,temperature=None,threshold=None):
#     length=prompt.shape[0]
#     vocab_size=
#     for i in range(max_tokens):
#         output=model(prompt)
#         if temperature:
#             scale=1/temperature
#             output.mul_(scale)
#         p_next_token=softmax(output,dim=1)[length-1]
#         if threshold:
#             dict_next_token={index:p for index,p in enumerate(p_next_token)}
#             sorted_next_token=sorted(dict_next_token,key=lambda x:x.value(),reverse=True)
#             for i in range(vocab_size):


def decode(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_tokens: int = 2048,
    temperature: float | None = None,
    top_p: float | None = None
):

    device = next(model.parameters()).device
    tokens = prompt.clone().to(device)
    if tokens.dim() != 1:
        raise ValueError(
            f"decode expects a 1D prompt tensor of token ids, got shape {tuple(tokens.shape)}"
        )

    context_length = getattr(model, "context_length", None)
    kv_cache = None
    can_use_kv_cache = True

    # 生成阶段不需要保留计算图。这里显式关闭 autograd，
    # 可以避免 cache 把整条历史前向图串起来，减少显存占用并加快采样。
    with torch.no_grad():
        for _ in range(max_tokens):

            if can_use_kv_cache:
                if kv_cache is None:
                    # 第一步没有历史 cache，需要把“当前可见窗口”整段送进模型做一次预填充(prefill)。
                    # 如果 prompt 已经长于 context_length，这里只保留最后一个窗口，
                    # 与旧版 decode 的截断行为保持一致。
                    window_start = 0 if context_length is None else max(
                        tokens.shape[0] - context_length, 0
                    )
                    input_ids = tokens[window_start:].unsqueeze(0)
                    token_positions = torch.arange(
                        window_start, tokens.shape[0], device=device
                    ).unsqueeze(0)
                else:
                    # 从第二步开始，只需把“最新生成的一个 token”送进模型。
                    # 之前所有历史信息都已经压进各层的 K/V cache 里，不再需要整段重算。
                    input_ids = tokens[-1:].unsqueeze(0)
                    token_positions = torch.tensor(
                        [[tokens.shape[0] - 1]], device=device
                    )

                try:
                    model_output = model(
                        input_ids,
                        token_positions=token_positions,
                        kv_cache=kv_cache,
                        use_kv_cache=True,
                    )
                    logits, kv_cache = model_output
                except TypeError:
                    # 保持 decode 的通用性: 如果外部传入的是不支持 kv cache 的模型，
                    # 自动退回到旧逻辑，而不是直接报错。
                    can_use_kv_cache = False
                    kv_cache = None

            if not can_use_kv_cache:
                # 回退路径沿用原实现: 每步重新计算最近的上下文窗口。
                if context_length is not None:
                    input_ids = tokens[-context_length:].unsqueeze(0)
                else:
                    input_ids = tokens.unsqueeze(0)

                logits = model(input_ids)           # (1, T, vocab)

            logits = logits[:, -1, :]           # (1, vocab)
            logits = logits.squeeze(0)          # (vocab)

            # temperature
            if temperature is not None and temperature > 0:
                logits = logits / temperature

            probs = softmax(logits, dim=-1)

            # top-p (nucleus sampling)
            if top_p is not None:

                sorted_probs, sorted_indices = torch.sort(
                    probs,
                    descending=True
                )

                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                mask = cumulative_probs > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False

                sorted_probs[mask] = 0

                sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-8)

                next_token = sorted_indices[
                    torch.multinomial(sorted_probs, 1)
                ]

            else:
                next_token = torch.multinomial(probs, 1)

            next_token = next_token.squeeze()
            if next_token.item() == 0:
                break

            # append token
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])

    return tokens
