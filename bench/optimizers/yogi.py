"""Yogi optimizer (vendored, self-contained).

Yogi (Zaheer et al., 2018 — "Adaptive Methods for Nonconvex Optimization",
NeurIPS 2018): AdamW with an additive v_t update that bounds the rate of
v_t change so a sudden gradient burst can't blow through into the
denominator.

Reference implementations:
  - jettify/pytorch-optimizer (Apache-2.0)
  - google-research/yogi
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class Yogi(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid Yogi learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid Yogi beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid Yogi beta2: {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            initial_accumulator=initial_accumulator,
            weight_decay=weight_decay,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            init_acc = group["initial_accumulator"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)

                grad_sq = g * g
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_sq), grad_sq, value=-(1.0 - beta2)
                )

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss
