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

    for _ in range(max_tokens):

        # Keep the full generated sequence in `tokens`, but only feed the most
        # recent context window to the model.
        if context_length is not None:
            input_ids = tokens[-context_length:].unsqueeze(0)
        else:
            input_ids = tokens.unsqueeze(0)

        # forward
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
