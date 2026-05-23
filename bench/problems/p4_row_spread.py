"""P4 — Row-spread pathology. Validates C4 + C5.

* C4: RACASO's spread cap (L1) bounds the magnitude of rotated updates
  on inputs with extreme per-row spread in gradient scale.
* C5: Eigh safe-skip (L2) survives near-rank-deficient covariance
  produced by the burst pattern.

Setup: a single matrix parameter ``W`` of shape ``(8, 8)``. The clean
objective is ``f(W) = 0.5 * ||W - M||_F^2`` for a fixed random target
``M``. The natural autograd gradient is ``W - M``.

We then **inject** a burst into one row of the gradient on alternating
steps. The burst multiplier cycles through ``[1e2, 1e4, 1e6, 1e8]``; the
chosen row also cycles ``0 -> 1 -> ... -> 7 -> 0``. On "calm" steps the
gradient passes through untouched. The injection lives in
``loss_and_grad`` because the burst is a property of the *input
distribution* (gradient stream) rather than the loss surface itself.

This pattern produces gradient covariance with one row's spectral mass
1e4 - 1e16 times larger than the others — the exact stress test for
spread-aware preconditioning.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


_BURST_MULTIPLIERS: Tuple[float, ...] = (1e2, 1e4, 1e6, 1e8)


class P4RowSpread(BenchProblem):
    """Row-spread gradient-injection pathology on an 8x8 matrix."""

    name = "p4_row_spread"
    max_steps = 3000
    converged_tol = 1e-3

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        gen = self._generator
        self._M = torch.randn(8, 8, generator=gen, dtype=torch.float64).to(self.device)
        # Step counter drives the alternating burst / calm cycle and
        # the row-and-multiplier rotation. Reset on every call to
        # ``init_params`` so multiple runs from the same instance are
        # reproducible (the harness only calls ``init_params`` once
        # per run, so this is belt-and-braces).
        self._step: int = 0

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        w = torch.randn(8, 8, generator=gen, dtype=torch.float64).to(self.device)
        w.requires_grad_(True)
        self._step = 0
        return [w]

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (w,) = params
        wd = w.detach()
        diff = wd - self._M
        loss = 0.5 * float((diff * diff).sum())
        grad = diff.clone()

        # Alternating calm / burst: burst on odd steps.
        if (self._step % 2) == 1:
            cycle_idx = (self._step // 2) % len(_BURST_MULTIPLIERS)
            mult = _BURST_MULTIPLIERS[cycle_idx]
            row_idx = (self._step // 2) % grad.shape[0]
            grad[row_idx] = grad[row_idx] * mult

        self._step += 1
        return float(loss), [grad]
