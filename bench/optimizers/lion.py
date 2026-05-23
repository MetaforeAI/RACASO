"""Lion optimizer — vendored reference implementation.

Sign-momentum optimizer from Chen et al. 2023, "Symbolic Discovery of
Optimization Algorithms" (arXiv:2302.06675). This is the canonical
``lion-pytorch`` form by Phil Wang, simplified to remove the optional
triton-fused kernel path (we want CPU + ROCm + CUDA to all work
identically for benchmarking).

Update rule per parameter ``p`` with gradient ``g`` at step ``t``:

    update_t = sign(β1 · m_{t-1} + (1 - β1) · g)
    m_t      = β2 · m_{t-1} + (1 - β2) · g          # separate β2 for buffer
    p ← p · (1 - lr · weight_decay)                  # AdamW-style decay
    p ← p - lr · update_t

The two-coefficient form (β1 for update direction, β2 for momentum
buffer) is the published Lion update. Liger uses the shared-coefficient
simplification (β1 = β2) which produces equivalent behavior at half the
buffer cost; for fair comparison the Lion baseline here keeps the
two-coefficient form.

Reference: https://github.com/lucidrains/lion-pytorch (MIT licensed).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion optimizer.

    Args:
        params: iterable of parameters.
        lr: learning rate. Lion typically wants 3-10x lower than Adam.
        betas: ``(β1, β2)``. β1 = direction momentum coefficient,
            β2 = buffer momentum coefficient. Defaults (0.9, 0.99) match
            the Lion paper.
        weight_decay: AdamW-style decoupled weight decay. Default 0.0.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {weight_decay}"
            )
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]

                # Decoupled weight decay (AdamW-style).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update direction uses *previous* momentum (β1 mix).
                # ``update = sign(β1 · m_{t-1} + (1 - β1) · g)``.
                update = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1).sign_()

                # Now advance the momentum buffer with its own coefficient β2.
                exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)

                # Apply the sign-update.
                p.add_(update, alpha=-lr)

        return loss
