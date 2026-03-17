import torch
from typing import Optional, Callable
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        beta_1, beta_2 = betas
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr, "beta_1": beta_1, "beta_2": beta_2,
                    "weight_decay": weight_decay, "eps": eps}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            lam = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if state.get("m") is None:
                    state["m"] = torch.zeros_like(p.data)
                if state.get("v") is None:
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                t = state.get("t", 1)
                m.mul_(beta_1).add_(g, alpha=1-beta_1)
                v.mul_(beta_2).add_(g.pow(2), alpha=1-beta_2)
                lr_t = lr*math.sqrt(1-beta_2**t)/(1-beta_1**t)
                p.data -= lr_t*m/(v.sqrt() + eps)
                p.data.mul_(1-lr*lam)
                state["t"] = t+1
        return loss


def cos_lr(
    iteration: int,
    max_lr: float,
    min_lr: float,
    warmup_iterations: int,
    annealing_iterations: int
) -> float:
    """
    带 warmup 的余弦退火学习率调度器。

    Args:
        iteration: 当前迭代步数（从 0 开始）
        max_lr: 最大学习率（warmup 结束时的值）
        min_lr: 最小学习率（退火结束时的值）
        warmup_iterations: warmup 阶段的总步数
        annealing_iterations: 退火阶段结束的总步数（即总步数 = warmup_iterations + 退火长度）

    Returns:
        当前步数的学习率
    """
    if iteration < warmup_iterations:
        # 线性 warmup
        return iteration * max_lr / warmup_iterations
    elif iteration <= annealing_iterations:
        # 余弦退火（从 max_lr 下降到 min_lr）
        progress = (iteration - warmup_iterations) / \
            (annealing_iterations - warmup_iterations)
        cos_term = math.cos(progress * math.pi)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos_term)
    else:
        # 退火结束后保持 min_lr
        return min_lr


def gradient_clipping(parameters, max_l2_norm):
    """
    所有梯度的L2范数如果小于max_l2_norm，进行缩放
    """
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    total_norm = torch.norm(torch.stack([grad.norm() for grad in grads]))
    if total_norm <= max_l2_norm:
        return
    scale = max_l2_norm/(total_norm+1e-6)
    for grad in grads:
        grad.mul_(scale)


if __name__ == "__main__":
    weights = torch.nn.Parameter(5*torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
    print(20*"-")
    weights = torch.nn.Parameter(5*torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
    print(20*"-")
    weights = torch.nn.Parameter(5*torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
