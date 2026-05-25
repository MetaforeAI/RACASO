"""P1 — Off-axis quadratic. Validates C1 (rotation matters).

Quadratic form ``f(W) = 0.5 * W^T (U Lambda U^T) W - b^T W`` where ``U`` is
a random orthogonal matrix and ``Lambda`` is the fixed diagonal
``diag([10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005])``. The optimum is
``W* = (U Lambda U^T)^{-1} b`` with ``f(W*) = -0.5 * b^T W*``; we shift the
loss by this constant so the converged criterion ``loss < 1e-4`` is
meaningful (loss is non-negative and zero at the optimum).

Why this problem: axis-aligned momentum (Adam) handles only the diagonal
of the Hessian. The rotation ``U`` breaks alignment, exposing methods that
can model off-diagonal curvature.

The Hessian is constant ``H = U Lambda U^T``, so we override
``loss_and_grad`` with the analytic form
``grad = H W - b`` and ``loss = 0.5 * W^T H W - b^T W - f*``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


_EIGENVALUES: Tuple[float, ...] = (10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005)


class P1OffAxisQuadratic(BenchProblem):
    """8-dim quadratic with a non-axis-aligned eigenbasis.

    Class attributes are set on the subclass (not the instance) because
    ``_registered_problems()`` walks subclasses by their class-level
    ``.name``.
    """

    # ARCHIVE: legacy 1-D vector version. Routed to RACASO's L3 Yogi
    # fallback (no 2-D rotation), so this never exercised the
    # rotation/HVP pipeline. Replaced by the matrix-version
    # ``p1_off_axis_quad`` in p1_off_axis_quad.py. Kept as ``_v1`` for
    # posterity / numerical comparison. Name blanked so the subclass
    # registry does not pick it up.
    name = ""
    max_steps = 5000
    converged_tol = 1e-4

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        # Build a fixed (per-seed) random orthogonal U via QR of a Gaussian.
        # NB: the matrix solve happens on CPU in float64 for numerical
        # stability; we move the final tensors to self.device afterward.
        gen = self._generator
        n = len(_EIGENVALUES)
        a = torch.randn(n, n, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        diag_sign = torch.sign(torch.diagonal(r))
        diag_sign[diag_sign == 0] = 1.0
        q = q * diag_sign.unsqueeze(0)

        lam = torch.tensor(_EIGENVALUES, dtype=torch.float64)
        H_cpu = (q * lam.unsqueeze(0)) @ q.t()
        b_cpu = torch.randn(n, generator=gen, dtype=torch.float64)

        # Constant shift so loss is non-negative with zero at the optimum.
        # Stay in float64 on CPU for the solve, then convert/move.
        w_star = torch.linalg.solve(H_cpu, b_cpu)
        self._f_star = 0.5 * float(w_star @ H_cpu @ w_star) - float(
            b_cpu @ w_star
        )
        self._H = H_cpu.to(self.device)
        self._b = b_cpu.to(self.device)

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        n = len(_EIGENVALUES)
        w = torch.randn(n, generator=gen, dtype=torch.float64).to(self.device)
        w.requires_grad_(True)
        return [w]

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (w,) = params
        wd = w.detach()
        hw = self._H @ wd
        loss = 0.5 * float(wd @ hw) - float(self._b @ wd) - self._f_star
        grad = hw - self._b
        return float(loss), [grad]
