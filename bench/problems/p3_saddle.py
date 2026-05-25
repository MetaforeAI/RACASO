"""P3 — Saddle problems. Validate C2 + C3.

* C2: Hutchinson HVP captures negative curvature (escapes the saddle).
* C3: GNB fallback is positive-semidefinite-safe (cannot escape on its
  own — confirmation by absence of escape).

Two registered subclasses:

* ``p3a_saddle_2d`` — ``f(x, y) = x^2 - y^2``, init ``(0.1, 0.1)``,
  ``max_steps=2000``. Intrinsically a 2-D problem; routes through L3 in
  RACASO (paper §8 documents this as a tiny-2-D regime).
* ``p3b_saddle_n20`` — quadratic saddle on a ``5 × 4`` MATRIX parameter:
  ``f(W) = 0.5 * sum D_ij W_ij^2``, where ``D`` is a fixed (10/+1,
  10/-1) shuffled mask reshaped to 5×4. Matrix shape so RACASO's 2-D
  rotation pipeline engages (the legacy 1-D version routed to L3 Yogi
  and never tested the 2-D path; see p3_saddle_v1.py).

The objective is unbounded below in the negative-curvature directions,
so the converged criterion is **escape from origin**, not loss below a
tolerance. ``converged()`` returns False so the run always runs the full
``max_steps`` budget — downstream analysis reads the trajectory.

Subclasses set ``saddle = True`` so harness / plot code can identify
saddle problems and interpret their trajectories accordingly.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


_ESCAPE_THRESHOLD: float = 5.0  # ||W||^2 > 5.0 counts as "escaped"


class P3aSaddle2D(BenchProblem):
    """2-D quadratic saddle: f(x, y) = x^2 - y^2.

    Negative-curvature direction is the y-axis; positive curvature
    along x. A purely PSD-only second-order method (GNB) cannot escape.
    """

    name = "p3a_saddle_2d"
    max_steps = 2000
    converged_tol = 0.0  # unused; see overridden ``converged``
    saddle: bool = True

    def init_params(self) -> List[torch.Tensor]:
        w = torch.tensor([0.1, 0.1], dtype=torch.float64, device=self.device)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        x = w[0]
        y = w[1]
        return x * x - y * y

    def converged(self, current_loss: float, step: int) -> bool:
        del current_loss, step
        return False


class P3bSaddleN20(BenchProblem):
    """20-dim quadratic saddle on a 5×4 MATRIX parameter.

    f(W) = 0.5 * sum_{i,j} D_{ij} W_{ij}^2 where D has 10 positive and
    10 negative entries (signed-magnitude mask reshaped to 5×4). The
    matrix shape exercises RACASO's 2-D rotation pipeline.
    """

    name = "p3b_saddle_n20"
    max_steps = 2000
    converged_tol = 0.0
    saddle: bool = True

    _SHAPE = (5, 4)
    _N = 20

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        eigvals = torch.cat(
            [
                torch.ones(10, dtype=torch.float64),
                -torch.ones(10, dtype=torch.float64),
            ]
        )
        # Shuffle so positive and negative directions are interspersed
        # in the matrix layout — increases pressure on rotation refresh.
        perm_gen = torch.Generator()
        perm_gen.manual_seed(7)  # fixed across seeds for reproducibility
        perm = torch.randperm(self._N, generator=perm_gen)
        eigvals = eigvals[perm]
        self._D = eigvals.reshape(self._SHAPE).to(self.device)

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Small symmetric offset; seeded so different seeds produce
        # different runs.
        W = (0.1 * torch.randn(*self._SHAPE, generator=gen, dtype=torch.float64)).to(self.device)
        W.requires_grad_(True)
        return [W]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (W,) = params
        return 0.5 * (self._D * W * W).sum()

    def converged(self, current_loss: float, step: int) -> bool:
        del current_loss, step
        return False
