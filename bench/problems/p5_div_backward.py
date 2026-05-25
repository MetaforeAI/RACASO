"""P5 — DivBackward0 hazard on MATRIX parameters. Validates C6 (L5
safe-skip on unbounded second derivative).

The forward graph is the Frobenius-generalized version of the original
ratio:

    diff = (<x, y>_F / ||x||_F) - target
    loss = diff * diff             # squared so loss >= 0 and tol is meaningful

where ``x, y ∈ R^{2×2}``, ``<x, y>_F = sum_ij x_ij y_ij``, and
``||x||_F = sqrt(sum_ij x_ij^2)``. The 1/||x||_F factor makes the
*second* derivative blow up as ``||x||_F`` shrinks — the second-order
term has a ``1/||x||_F^3`` structure. First-order autograd is finite;
methods that take a Hessian-vector product (RACASO Hutchinson) hit
unbounded curvature and must safe-skip (L5).

The matrix shape ensures RACASO's 2-D rotation pipeline engages on both
parameters (the legacy 1-D version in p5_div_backward_v1.py routed to
L3 Yogi and never exercised that path).

Sizes: x, y ∈ R^{2×2}, target fixed scalar.

``max_steps=1000``, ``converged_tol=1e-3``.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class P5DivBackward(BenchProblem):
    """Matrix ratio-form objective with unbounded second derivative near
    ``||x||_F = 0``.
    """

    name = "p5_div_backward"
    max_steps = 1000
    converged_tol = 1e-3

    _SHAPE = (2, 2)

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        gen = self._generator
        # Fixed target chosen so the optimum is non-trivial but bounded.
        self._target = float(torch.randn(1, generator=gen).item())

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Initialize x at modest Frobenius norm — not at zero (singular
        # point) but close enough that the second derivative is large.
        x = (0.3 * torch.randn(*self._SHAPE, generator=gen, dtype=torch.float64)).to(self.device)
        y = torch.randn(*self._SHAPE, generator=gen, dtype=torch.float64).to(self.device)
        x.requires_grad_(True)
        y.requires_grad_(True)
        return [x, y]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        x, y = params
        # Plain torch.norm so the DivBackward0 graph structure is what
        # RACASO's HVP refresh sees.
        n = torch.norm(x)
        ratio = (x * y).sum() / n
        diff = ratio - self._target
        return diff * diff
