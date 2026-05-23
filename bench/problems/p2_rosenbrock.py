"""P2 — Rosenbrock. Validates C1 at scale (curved off-axis valley).

Two registered subclasses:

* ``p2a_rosenbrock_2d`` — classic 2-D Rosenbrock,
  ``f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2``, init at ``(-1.2, 1.0)``,
  converged at ``loss < 1e-4``, ``max_steps=5000``.
* ``p2b_rosenbrock_n100`` — generalized N=100 Rosenbrock,
  ``f(x) = sum_{i=1}^{N-1} [(1 - x_i)^2 + 100 * (x_{i+1} - x_i^2)^2]``,
  init at all ``-1.2``, converged at ``loss < 1e-3``, ``max_steps=10000``.

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
    """Generalized N=100 Rosenbrock starting at all -1.2."""

    name = "p2b_rosenbrock_n100"
    max_steps = 10000
    converged_tol = 1e-3

    _N: int = 100

    def init_params(self) -> List[torch.Tensor]:
        w = torch.full((self._N,), -1.2, dtype=torch.float64, device=self.device)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        xi = w[:-1]
        xip1 = w[1:]
        return ((1.0 - xi) ** 2 + 100.0 * (xip1 - xi * xi) ** 2).sum()
