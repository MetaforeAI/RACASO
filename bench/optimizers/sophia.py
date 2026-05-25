"""Sophia (SophiaG variant) — Scalable Second-order Optimizer for
Language Model Pre-training.

Vendored from https://github.com/Liuhong99/Sophia (MIT License, ©
2023 Hong Liu). Reference: Liu et al. 2023,
"Sophia: A Scalable Stochastic Second-order Optimizer for Language
Model Pre-training" (arXiv:2305.14342).

This is the SophiaG variant (Gauss-Newton-Bartlett Hessian estimator);
the upstream's `update_hessian()` method computes `g * g` as the
diagonal Hessian proxy and `step()` applies the per-element clip
`min(|m| / (rho * bs * h + eps), 1) * sign(m)`.

Adaptations for the bench harness:
  - Single-tensor step inlined (the upstream's foreach API is GPU-only
    on CUDA; we collapse to single-tensor on every step).
  - `update_hessian()` is called inline at the start of `step()` so the
    bench harness does not need a separate hook; this is a no-op on
    semantics (upstream pattern: call `update_hessian()` every step
    after the gradient is available).
  - `bs` (batch size used in GNB reweighting) defaults to 1; the
    bench harness's loss objectives are per-element scaled already.

Provenance: https://github.com/Liuhong99/Sophia (MIT).
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class Sophia(Optimizer):
    """SophiaG-style optimizer with GNB Hessian-diagonal proxy g·g.

    Args:
        lr: learning rate.
        betas: (β1, β2) — momentum + Hessian-EMA decay.
        rho: clipping parameter (max per-element step magnitude in
             units of |m| / h before the clip).
        weight_decay: AdamW-style weight decay.
        bs: batch-size scaling for the GNB Hessian proxy; bench uses 1
            (loss is already per-element normalized).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        bs: int = 1,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, bs=bs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Update the GNB Hessian-diagonal proxy h <- β2 * h + (1-β2) * g²."""
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update GNB Hessian proxy first (upstream calls it as a
        # separate user-driven hook; we fold it in for the bench).
        self.update_hessian()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            rho = group["rho"]
            wd = group["weight_decay"]
            bs = group["bs"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")
                grad = p.grad
                state = self.state[p]
                # update_hessian() may have initialized state["hessian"]
                # already on first call; ensure all keys exist.
                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                # Stepweight decay.
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                # First-moment EMA.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Sophia clip: ratio = min(|m| / (rho·bs·h + eps), 1)
                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(max=1.0)
                p.addcmul_(exp_avg.sign(), ratio, value=-lr)
        return loss
