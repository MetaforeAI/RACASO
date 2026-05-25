"""P2 — Rosenbrock. Validates C1 at scale (curved off-axis valley).

Two registered subclasses:

* ``p2a_rosenbrock_2d`` — classic 2-D Rosenbrock,
  ``f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2``, init at ``(-1.2, 1.0)``,
  converged at ``loss < 1e-4``, ``max_steps=5000``. Intrinsically a 2-D
  problem; the parameter is a length-2 vector that routes to RACASO's
  L3 (see paper §8 note on tiny-2-D regimes).

* ``p2b_rosenbrock_n100`` — generalized N=100 Rosenbrock on a
  ``10 × 10`` MATRIX parameter, summed over the flattened indices. This
  is the matrix-shape rebuild — exercises RACASO's 2-D rotation pipeline
  (the legacy 1-D version routed to L3 Yogi and never tested the 2-D
  path; see p2_rosenbrock_v1.py).
  init drawn with the seeded generator (per-seed variation), converged
  at ``loss < 1e-3``, ``max_steps=10000``.

Both subclasses use the default ``loss_and_grad`` (autograd against
``forward``).
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class P2aRosenbrock2D(BenchProblem):
    """Classic 2-D Rosenbrock starting at (-1.2, 1.0)."""

    name = "p2a_rosenbrock_2d"
    max_steps = 5000
    converged_tol = 1e-4

    def init_params(self) -> List[torch.Tensor]:
        w = torch.tensor([-1.2, 1.0], dtype=torch.float64, device=self.device)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        x = w[0]
        y = w[1]
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


class P2bRosenbrockN100(BenchProblem):
    """Generalized N=100 Rosenbrock on a 10×10 MATRIX parameter.

    The objective is the standard generalized Rosenbrock summed over the
    flattened element index:
        f(W) = Σ_{i=0..N-2} [(1 - x_i)² + 100 (x_{i+1} - x_i²)²]
    where x_i = vec(W)_i. Matrix shape so RACASO's 2-D rotation engages.
    """

    name = "p2b_rosenbrock_n100"
    max_steps = 10000
    converged_tol = 1e-3

    _SHAPE = (10, 10)

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Seeded init — slight perturbation around -1.2 so different
        # seeds produce distinguishable runs (deterministic baselines
        # on a deterministic init = degenerate sweep). The mean is
        # still -1.2 so the convergence target is comparable across
        # seeds.
        base = torch.full(self._SHAPE, -1.2, dtype=torch.float64)
        noise = 0.05 * torch.randn(*self._SHAPE, generator=gen, dtype=torch.float64)
        w = (base + noise).to(self.device)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (W,) = params
        v = W.reshape(-1)
        xi = v[:-1]
        xip1 = v[1:]
        return ((1.0 - xi) ** 2 + 100.0 * (xip1 - xi * xi) ** 2).sum()
