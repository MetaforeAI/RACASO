"""Muon — MomentUm Orthogonalized by Newton-schulz.

Vendored from https://github.com/KellerJordan/Muon (MIT License, ©
2024 Keller Jordan), pinned to the upstream `master` revision read on
2026-05-22. Lightly adapted for the bench harness:

  - The distributed `Muon` class (which uses torch.distributed for
    sharded all-gather) is replaced with the upstream's own
    `SingleDeviceMuon` class — equivalent math, no dist dependency.
  - `bfloat16` casts inside the Newton-Schulz iteration are gated on
    float-dtype compatibility; for float64 / CPU bench problems we keep
    the original dtype.
  - 1-D parameters (norms, biases) use a fallback AdamW path, mirroring
    Sophia/Muon paper recommendation. The fallback uses the same
    `adam_update` helper as upstream.

Provenance: https://github.com/KellerJordan/Muon (MIT). The Newton-
Schulz quintic was contributed by @YouJiacheng et al.; the
batched-Muon trick by @scottjmaddox; see upstream for the full
attribution chain.
"""
from __future__ import annotations

import torch


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz quintic iteration for the zeroth-power/orthogonalization
    of G. See upstream comment for the rationale on slope tuning at zero."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Upstream uses bf16 on GPU; for our CPU/float64 bench we use the
    # input's own dtype (or float32 if int) so the iteration stays in
    # a regime numerically compatible with the problem.
    if G.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        X = G.clone()
    else:
        X = G.to(torch.float32)
    if X.size(-2) > X.size(-1):
        X = X.mT
    # Spectral norm <= 1 normalization.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def _muon_update(
    grad: torch.Tensor, momentum: torch.Tensor,
    beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters — flatten the last 3 dims
        update = update.view(len(update), -1)
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(
    grad: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor,
    step: int, betas, eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Single-device Muon (non-distributed). 2-D params use the
    Newton-Schulz orthogonalized update; 1-D params fall back to a
    standard AdamW update.

    Args:
        params: list of nn.Parameter (or tensors).
        lr: learning rate (units of spectral norm per update on 2-D
            params; standard LR on 1-D params).
        weight_decay: AdamW-style weight decay (applied to 2-D and 1-D).
        momentum: Muon momentum (β) for the 2-D update.
        adam_betas: (β1, β2) for the 1-D AdamW fallback.
        adam_eps: epsilon for the 1-D AdamW fallback.
        ns_steps: Newton-Schulz iteration count.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        adam_betas=(0.9, 0.999),
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            adam_betas=adam_betas, adam_eps=adam_eps, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            adam_betas = group["adam_betas"]
            adam_eps = group["adam_eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if p.ndim >= 2:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _muon_update(
                        g, state["momentum_buffer"],
                        beta=momentum, ns_steps=ns_steps,
                    )
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    # 1-D fallback: AdamW.
                    if len(state) == 0:
                        state["step"] = 0
                        state["buf1"] = torch.zeros_like(p)
                        state["buf2"] = torch.zeros_like(p)
                    state["step"] += 1
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                    update = _adam_update(
                        g, state["buf1"], state["buf2"],
                        state["step"], adam_betas, adam_eps,
                    )
                    p.add_(update, alpha=-lr)
        return loss
