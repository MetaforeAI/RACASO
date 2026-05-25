"""P1 — Off-axis quadratic on a matrix parameter. Validates C1
(rotation matters under off-diagonal curvature).

Quadratic form
    f(W) = 0.5 * vec(W)^T H vec(W) - b^T vec(W)
where ``W ∈ R^{4 × 4}`` and ``H = U Λ U^T`` is a fixed (per-seed) random
orthogonal rotation of the diagonal
``Λ = diag([10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 5e-4,
             2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6])`` (16 eigenvalues for
the 4×4 parameter element-vector). The optimum is
``vec(W*) = H^{-1} b``; we shift the loss by ``f(W*)`` so the
non-negative criterion ``loss < 1e-4`` is meaningful.

Why this problem: axis-aligned momentum (Adam) handles only the diagonal
of the Hessian. The rotation ``U`` breaks alignment, exposing methods
that can model off-diagonal curvature. The matrix parameter shape
exercises RACASO's 2-D rotation pipeline (the 1-D vector version in
``p1_off_axis_quad_v1.py`` falls through to L3 Yogi).

The Hessian is constant; we override ``loss_and_grad`` with the analytic
form ``grad = (H vec(W) - b).reshape(W.shape)``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


_EIGENVALUES: Tuple[float, ...] = (
    10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005,
    1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6,
)
_SHAPE: Tuple[int, int] = (4, 4)  # 16 elements = len(_EIGENVALUES)


class P1OffAxisQuadratic(BenchProblem):
    """4×4 matrix quadratic with a non-axis-aligned eigenbasis."""

    name = "p1_off_axis_quad"
    max_steps = 5000
    converged_tol = 1e-4

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        gen = self._generator
        n = len(_EIGENVALUES)
        # Random orthogonal U via QR of a Gaussian on CPU/float64.
        a = torch.randn(n, n, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        diag_sign = torch.sign(torch.diagonal(r))
        diag_sign[diag_sign == 0] = 1.0
        q = q * diag_sign.unsqueeze(0)

        lam = torch.tensor(_EIGENVALUES, dtype=torch.float64)
        H_cpu = (q * lam.unsqueeze(0)) @ q.t()
        b_cpu = torch.randn(n, generator=gen, dtype=torch.float64)

        # Shift so loss is non-negative with zero at the optimum.
        w_star = torch.linalg.solve(H_cpu, b_cpu)
        self._f_star = 0.5 * float(w_star @ H_cpu @ w_star) - float(
            b_cpu @ w_star
        )
        self._H = H_cpu.to(self.device)
        self._b = b_cpu.to(self.device)

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Seeded init — different seeds produce different starting
        # points (fixes the methodology gap where deterministic
        # baselines + deterministic init = degenerate seed sweep).
        w = torch.randn(*_SHAPE, generator=gen, dtype=torch.float64).to(self.device)
        w.requires_grad_(True)
        return [w]

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (W,) = params
        w_vec = W.detach().reshape(-1)
        hw = self._H @ w_vec
        loss = 0.5 * float(w_vec @ hw) - float(self._b @ w_vec) - self._f_star
        grad_vec = hw - self._b
        return float(loss), [grad_vec.reshape(_SHAPE)]
