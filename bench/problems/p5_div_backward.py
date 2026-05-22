"""P5 — DivBackward0 hazard. Validates C6 (L5 safe-skip on unbounded
second derivative).

The forward graph is

    diff = (x . y / ||x||) - target
    loss = diff * diff             # squared so loss >= 0 and tol=1e-3 is meaningful

with ``||x|| = torch.norm(x)``. The 1/||x|| factor makes the *second*
derivative blow up as ``||x||`` shrinks — the second-order term has a
``1/||x||^3`` structure. First-order autograd is finite, so naive
optimizers proceed; methods that take a Hessian-vector product (RACASO
Hutchinson) hit unbounded curvature and must safe-skip (L5).

Sizes:
    x, y: vectors of length 4
    target: fixed scalar

We override ``forward`` so the default autograd path supplies the
first-order gradients. The L5 trigger happens inside the optimizer's
HVP refresh, not here.

``max_steps=1000``, ``converged_tol=1e-3``.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class P5DivBackward(BenchProblem):
    """Tiny ratio-form objective with unbounded second derivative near
    ``||x||=0``.
    """

    name = "p5_div_backward"
    max_steps = 1000
    converged_tol = 1e-3

    _DIM: int = 4

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        gen = self._generator
        # Fixed target chosen so the optimum is non-trivial but bounded.
        self._target = float(torch.randn(1, generator=gen).item())

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Initialize x at modest norm — not at zero (singular point) but
        # close enough that the second derivative is large. y is freely
        # initialized.
        x = 0.3 * torch.randn(self._DIM, generator=gen, dtype=torch.float64)
        y = torch.randn(self._DIM, generator=gen, dtype=torch.float64)
        x.requires_grad_(True)
        y.requires_grad_(True)
        return [x, y]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        x, y = params
        # Use a tiny eps inside the norm only if numerical breakdown
        # would kill the run before the optimizer even sees a chance.
        # We deliberately use plain torch.norm here so the DivBackward0
        # graph structure is what RACASO's HVP refresh sees.
        n = torch.norm(x)
        ratio = (x * y).sum() / n
        diff = ratio - self._target
        return diff * diff
