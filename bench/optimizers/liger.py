"""Liger optimizer — Layered Iterative Gradient Estimator with Rectification.

by Richard I Christopher, 2026

A hybrid optimizer that routes parameters by gradient regime rather than by
parameter count. Matrix-shaped parameters (``ndim >= 2``) take a Lion
sign-momentum update; vector- and scalar-shaped parameters (``ndim <= 1``)
take a Yogi variance-rectified update. The routing decision is pinned per
parameter at its first encounter so there is no per-step branching cost
beyond an ``ndim`` check.

See ``Liger_Paper.md`` for the design rationale.

## The dispatch by dimensionality

Modern transformer-style modules are gradient-heterogeneous. Two pathologies
motivate the split:

  **Pathology A — adaptive warmup coupling.**
    Adam/AdamW/Muon/Shampoo accumulate ``v_t`` over training and rely on
    bias-corrected ``v_hat``. In the first 5-50 steps the accumulator is
    nearly empty, so practitioners stack an LR warmup on top. The LR
    warmup keeps gradients small, which keeps ``v_t`` small, which keeps
    the adaptive engine cold — the two warmups gate each other. Liger's
    matrix path is Lion's sign-momentum update, which has *no* ``v_t`` to
    wait on: ``m_0 = init_acc ≈ 0``, so at step 1 the update direction is
    ``sign((1-β1)·g) = sign(g)`` — exactly the right direction, no warmup.

  **Pathology B — rank-1 destruction on scalars.**
    Adam's ``m / sqrt(v + eps)`` update destroys rank-1 gradient structure.
    A scalar gate's gradient is a single bursty dot-product per step;
    Adam normalizes by ``sqrt(v_t)`` and converts that burst into a
    normalized update indistinguishable from steady state. Yogi's
    ``v_t -= (1-β2)·sign(v_{t-1} - g²)·g²`` bounds the accumulator so a
    single burst cannot inflate ``v_t`` and crush all future updates.
    Yogi is strictly the right tool for the 1-D/0-D case.

The architectural observation: matrix gradients in attention/MLP/norm
modules flow through softmax-normalized or RMSNormed paths and arrive
already well-conditioned — they don't need second-moment scaling.
Scalar/vector gradients sit *in* the affine corrections of those norms
and bypass the conditioning. The optimizer should match its tool to the
workpiece. Lion for matrices, Yogi for scalars.

## Mathematical pipeline

For each parameter ``p`` with gradient ``g`` at step ``t``:

  **2-D+ path (Lion).** Let ``m_t = β1·m_{t-1} + (1-β1)·g``. Then
      update = sign(m_t)
      p -= lr · update
      p -= lr · weight_decay · p          # decoupled weight decay

  *Mathematical equivalence note.* The published Lion paper writes the
  update as

      c_t    = β1·m_{t-1} + (1-β1)·g
      update = sign(c_t)
      m_t    = β2·m_{t-1} + (1-β2)·g       # (separate momentum coefficient)

  with *two* momentum coefficients (β1 for the direction, β2 for the
  buffer). Under the spec's shared-coefficient simplification
  (β1 = β2 in the Lion-paper sense), c_t and m_t become the same
  expression. So computing ``m_t`` first and taking ``sign(m_t)``
  is exactly equivalent to computing ``c_t = β1·m_{t-1} + (1-β1)·g``
  separately, taking ``sign(c_t)``, and *then* assigning m_t ← c_t.
  The in-place form saves one temporary buffer per Lion-route param —
  a non-trivial memory saving at scale, and the basis of Liger's
  "~50% of AdamW state" headline. This is the same form used by the
  reference ``lion-pytorch`` implementation.

  **1-D/0-D path (Yogi).** Let
      m_t = β1·m_{t-1} + (1-β1)·g
      v_t = v_{t-1} - (1-β2)·sign(v_{t-1} - g²)·g²
      m_hat = m_t / (1 - β1^t)
      v_hat = v_t / (1 - β2^t)
  Then
      p -= lr · m_hat / (sqrt(v_hat).clamp_(eps_yogi) + eps_adam)
      p -= lr · weight_decay · p          # decoupled weight decay

## Four guarantees

  1.  **No warmup interaction.** Lion's sign-momentum is meaningful from
      step 1. Yogi's accumulator is only used on the 1-D/0-D path where
      the parameter count is tiny and even short warmups are tolerable.
  2.  **No rank-1 destruction.** Scalar and vector parameters take the
      Yogi path; ``v_t`` is variance-bounded, not variance-accumulating.
  3.  **No preconditioning overhead.** Lion is ``O(P)`` per parameter:
      no eigendecomposition, no Newton-Schulz iteration, no Kronecker
      factor maintenance. Matrix parameters get the cheapest possible
      update consistent with bounded direction.
  4.  **~55% of AdamW memory.** Lion needs one buffer per matrix
      parameter (``m_t``); Yogi needs two per vector/scalar parameter
      (``m_t``, ``v_t``). Since matrix params dominate total parameter
      count by 50-500x in any transformer-derivative, total optimizer
      state is approximately half of AdamW's.

## State route pinning

The route is decided on the first ``step()`` call for each parameter
and stored in ``state["is_lion"]`` (bool). Reshape-after-step-1 is
undefined behavior — this matches Lion/Yogi reference impls and avoids
a per-step ``ndim`` re-check. A bool (rather than a string) is used so
``torch.optim.Optimizer.load_state_dict`` round-trips cleanly: its
``_cast`` helper iterates into strings (treating them as iterables) but
passes bools through unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class Liger(Optimizer):
    """Layered Iterative Gradient Estimator with Rectification.

    A torch.optim.Optimizer that dispatches by parameter dimensionality:
    matrix-shaped params (``ndim >= 2``) take a Lion sign-momentum update,
    vector- and scalar-shaped params (``ndim <= 1``) take a Yogi variance-
    rectified update.

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate. Default 1e-4.
        betas: ``(β1, β2)`` tuple. ``β1`` is the momentum coefficient
            shared between both routes. ``β2`` is the Yogi second-moment
            timescale; default 0.99 (lower than Adam's 0.999 because
            Yogi's update rule already bounds runaway variance).
        eps_yogi: floor on ``sqrt(v_hat)`` on the Yogi path. Default 1e-3
            (matches the Yogi paper; larger than Adam's 1e-8 because the
            larger floor makes the 1-D path robust to near-zero v_hat at
            cold-start).
        eps_adam: additive eps in the Yogi denominator. Default 1e-8.
        weight_decay: AdamW-style decoupled weight decay. Default 0.0.
        initial_accumulator: initial value for ``m_t`` and ``v_t``
            buffers. Default 1e-6 (matches Yogi/Muogi/RACASO family).
    """

    _optimizer_handles_own_clip: bool = False

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps_yogi: float = 1e-3,
        eps_adam: float = 1e-8,
        weight_decay: float = 0.0,
        initial_accumulator: float = 1e-6,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (isinstance(betas, tuple) and len(betas) == 2):
            raise ValueError(f"betas must be a (β1, β2) tuple, got {betas}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if eps_yogi <= 0.0:
            raise ValueError(f"eps_yogi must be positive, got {eps_yogi}")
        if eps_adam < 0.0:
            raise ValueError(f"eps_adam must be non-negative, got {eps_adam}")
        if weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {weight_decay}"
            )
        if initial_accumulator < 0.0:
            raise ValueError(
                "initial_accumulator must be non-negative, "
                f"got {initial_accumulator}"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps_yogi=eps_yogi,
            eps_adam=eps_adam,
            weight_decay=weight_decay,
            initial_accumulator=initial_accumulator,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Run a single optimization step.

        Args:
            closure: optional callable that re-evaluates the model and
                returns the loss. Required for optimizers that need to
                re-evaluate the closure multiple times (Liger does not).

        Returns:
            Loss value if ``closure`` is provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps_yogi = group["eps_yogi"]
            eps_adam = group["eps_adam"]
            wd = group["weight_decay"]
            init_acc = group["initial_accumulator"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError(
                        "Liger does not support sparse gradients"
                    )

                state = self.state[p]
                # Lazy init.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    state["is_lion"] = bool(p.ndim >= 2)
                    if not state["is_lion"]:
                        state["exp_avg_sq"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format
                        )
                    # Telemetry slots (overwritten on every step).
                    state["last_momentum_norm"] = 0.0
                    state["last_update_l1"] = 0.0
                    state["last_v_hat_max"] = 0.0
                    state["last_v_hat_min"] = 0.0

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]

                # Decoupled weight decay (AdamW-style), applied identically
                # on both routes so the route choice does not couple to the
                # decay schedule.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                if state["is_lion"]:
                    # ── Lion path: sign-momentum, no second moment. ──
                    # m_t = β1·m_{t-1} + (1-β1)·g
                    # update = sign(m_t)
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    update = exp_avg.sign()
                    p.add_(update, alpha=-lr)
                    state["last_momentum_norm"] = float(
                        exp_avg.detach().norm().item()
                    )
                    state["last_update_l1"] = float(
                        update.detach().abs().sum().item()
                    )
                else:
                    # ── Yogi path: bias-corrected sign-bounded v_t. ──
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    grad_sq = g * g
                    exp_avg_sq.addcmul_(
                        torch.sign(exp_avg_sq - grad_sq),
                        grad_sq,
                        value=-(1.0 - beta2),
                    )
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    m_hat = exp_avg / bc1
                    v_hat = exp_avg_sq / bc2
                    denom = v_hat.sqrt().clamp_(min=eps_yogi).add_(eps_adam)
                    p.addcdiv_(m_hat, denom, value=-lr)
                    # v_hat was modified by clamp_ in-place via sqrt result;
                    # re-derive max/min from exp_avg_sq directly so the
                    # telemetry reflects the underlying accumulator, not
                    # the (post-clamp) denominator.
                    v_hat_raw = exp_avg_sq / bc2
                    state["last_v_hat_max"] = float(
                        v_hat_raw.detach().max().item()
                    )
                    state["last_v_hat_min"] = float(
                        v_hat_raw.detach().min().item()
                    )

        return loss

    # ── Telemetry ────────────────────────────────────────────────────────
    def get_telemetry(self) -> dict:
        """Aggregate per-parameter diagnostics across the optimizer.

        Returns:
            dict with keys:
              - ``step_count``: max step across all params
              - ``num_2d_params``: count of params on the Lion route
              - ``num_1d_params``: count of params on the Yogi route
              - ``last_max_momentum_norm``: max ||m_t||₂ across Lion params
              - ``last_max_update_l1``: max Σ|sign(m_t)| across Lion params
                (equals element count when momentum is fully sign-saturated)
              - ``last_max_v_hat``: max v_hat across Yogi params
              - ``last_min_v_hat``: min v_hat across Yogi params (watch for
                near-zero values triggering the eps_yogi floor)
        """
        step_count = 0
        num_2d = 0
        num_1d = 0
        max_mom_norm = 0.0
        max_update_l1 = 0.0
        max_v_hat = 0.0
        min_v_hat = float("inf")
        any_yogi = False
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if not state:
                    continue
                step_count = max(step_count, state.get("step", 0))
                is_lion = state.get("is_lion")
                if is_lion is True:
                    num_2d += 1
                    max_mom_norm = max(
                        max_mom_norm, state.get("last_momentum_norm", 0.0)
                    )
                    max_update_l1 = max(
                        max_update_l1, state.get("last_update_l1", 0.0)
                    )
                elif is_lion is False:
                    num_1d += 1
                    any_yogi = True
                    max_v_hat = max(max_v_hat, state.get("last_v_hat_max", 0.0))
                    min_v_hat = min(min_v_hat, state.get("last_v_hat_min", 0.0))
        if not any_yogi:
            min_v_hat = 0.0
        return {
            "step_count": step_count,
            "num_2d_params": num_2d,
            "num_1d_params": num_1d,
            "last_max_momentum_norm": max_mom_norm,
            "last_max_update_l1": max_update_l1,
            "last_max_v_hat": max_v_hat,
            "last_min_v_hat": min_v_hat,
        }
