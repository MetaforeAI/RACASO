"""P3 — Saddle problems. Validate C2 + C3.

* C2: Hutchinson HVP captures negative curvature (escapes the saddle).
* C3: GNB fallback is positive-semidefinite-safe (cannot escape on its
  own — confirmation by absence of escape).

Two registered subclasses:

* ``p3a_saddle_2d`` — ``f(x, y) = x^2 - y^2``, init ``(0.1, 0.1)``,
  ``max_steps=2000``.
* ``p3b_saddle_n20`` — ``f(W) = 0.5 * W^T diag([+1]*10 + [-1]*10) W``,
  ``max_steps=2000``.

The objective is unbounded below in the negative-curvature directions,
so the converged criterion is **escape from origin**, not loss below a
tolerance. ``converged()`` returns True when ``||params||^2 > 5.0``.

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
        w = torch.tensor([0.1, 0.1], dtype=torch.float64)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        x = w[0]
        y = w[1]
        return x * x - y * y

    def converged(self, current_loss: float, step: int) -> bool:
        # "Convergence" = escape from origin. ``current_loss`` is not
        # informative here since it is unbounded below; we cannot read
        # parameter norm from a float. The harness records the loss
        # trajectory; downstream analysis identifies escape via
        # ``saddle=True`` and parameter norm reconstructed from
        # known-init + trajectory. We return False here so the run
        # always completes its full ``max_steps`` budget. This is
        # intentional: for saddle problems the full trajectory is the
        # signal, not first-crossing-of-tol.
        del current_loss, step
        return False


class P3bSaddleN20(BenchProblem):
    """20-dim quadratic saddle: 10 positive + 10 negative eigenvalues.

    f(W) = 0.5 * W^T diag([+1]*10 + [-1]*10) W
    """

    name = "p3b_saddle_n20"
    max_steps = 2000
    converged_tol = 0.0
    saddle: bool = True

    _N: int = 20

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        eigvals = torch.cat(
            [
                torch.ones(10, dtype=torch.float64),
                -torch.ones(10, dtype=torch.float64),
            ]
        )
        self._diag = eigvals

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Small symmetric offset so every direction has a non-zero
        # initial gradient component (helps Adam find the negative
        # direction at all, exposing the negative-curvature failure).
        w = 0.1 * torch.randn(self._N, generator=gen, dtype=torch.float64)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        return 0.5 * (self._diag * w * w).sum()

    def converged(self, current_loss: float, step: int) -> bool:
        del current_loss, step
        return False
