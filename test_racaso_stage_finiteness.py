"""Stage-by-stage finiteness gauge for RACASO under realistic X-organ regime.

After 8 failed v17.5 attempts where the model NaN'd between step 20-40
under RACASO (and identically under SOAP, but not under Muogi), the
suspect surface is the Kronecker-rotation machinery that SOAP and
RACASO share but Muogi does not (covariance EMA, eigh refresh,
rotated-basis update). This test replays a calibrated synthetic X-organ
gradient stream against a 192x192 parameter and asserts each internal
stage of RACASO.step() stays finite for 15 opt-steps × 4 K-windows
(matching the live TBPTT cadence).

Calibration anchors (from v17.4 / v17.5 attempt-8 step-1 telemetry):
  - X-organ raw_grad ≈ 4.22 (L2 over all X params)
  - X-organ lc_mass ≈ 1.83e3 (sum-absolute)
  - 16 2-D X params per micro × 2 blocks → ~16 2-D X params with grads
  - shape (192, 192) for the major cross_branch projections
  - per-param L2 ≈ 1.05 → per-element grad std ≈ 0.005

Failure mode the test isolates: stage trap records the (param_shape,
step, stage, finite-summary) of the first non-finite tensor at each
internal stage. The assertion message then names the exact stage that
broke first — momentum, GG accumulation, eigh, rotation, denom,
clip, spread cap, rotate-back, or parameter update.
"""

from __future__ import annotations

import torch

from racaso import RACASO


# Stage names in execution order — the test asserts each one is clean.
_STAGES = [
    "pre_grad",
    "exp_avg",
    "GG_L", "GG_R",
    "Q_L", "Q_R",
    "m_rot",
    "hessian_diag_rot",
    "denom",
    "update_rot_raw", "update_rot",
    "damp", "update_rot_post_spread",
    "update",
    "p_post_update",
]


def _make_x_param(shape=(192, 192), seed: int = 5) -> torch.nn.Parameter:
    """Match a representative cross_branch 2-D projection."""
    torch.manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape) * (1.0 / shape[1] ** 0.5))


def _synthetic_x_gradient(p: torch.Tensor, step: int, k: int, seed: int = 5):
    """Deterministic synthetic gradient stream matching live X-organ scale.

    Per-element std ≈ 0.005 to match observed per-X-param L2 ≈ 1.05 at
    192x192. Slight variation across step/k to mimic real batch-to-batch
    drift. The gradient is intentionally low-rank-biased (a small-norm
    full-rank perturbation around a rank-r structured component) since
    real attention/MLP gradients are batch-rank-bounded — exactly the
    regime that triggers ill-conditioned GG_L/GG_R.
    """
    g_seed = seed * 1009 + step * 31 + k
    gen = torch.Generator().manual_seed(g_seed)
    m, n = p.shape
    # Rank-r structured component dominates; small full-rank noise floor.
    rank = 16  # batch_size * factor — mimics token-rank bound
    U = torch.randn(m, rank, generator=gen) * 0.05
    V = torch.randn(rank, n, generator=gen) * 0.05
    structured = U @ V                                    # (m, n) low-rank
    noise = torch.randn(m, n, generator=gen) * 0.001       # full-rank noise
    return structured + noise


def test_racaso_stages_stay_finite_under_x_regime():
    """Run RACASO for 15 opt-steps × 4 K-windows on a 192x192 param.

    Arms RACASO._stage_trap; runs synthetic X-organ gradient stream;
    asserts every stage stayed finite for every step+window.

    The test fails with a message naming the first stage to break,
    which (per the live-log diagnosis) is the broken layer.
    """
    p = _make_x_param()
    opt = RACASO(
        [p],
        lr=1e-5,            # matches live --learning-rate
        betas=(0.965, 0.99),
        shampoo_beta=0.95,
        rho=0.04,
        gamma=0.04,
        refresh_freq=10,
        hessian_freq=10,
        eigh_residual_threshold=0.5,
        spread_cap=10.0,
        radam_enabled=True,
    )

    # Arm the stage trap.
    RACASO._stage_trap = {}
    try:
        K = 4
        for step in range(1, 16):
            for k in range(K):
                opt.zero_grad()
                p.grad = _synthetic_x_gradient(p, step, k)
                opt.step()
        trap = dict(RACASO._stage_trap)
    finally:
        RACASO._stage_trap = None

    # Build a readable failure report listing every broken stage.
    if trap:
        report_lines = ["RACASO non-finite at internal stages:"]
        for stage in _STAGES:
            if stage in trap:
                report_lines.append(f"  - {trap[stage]}")
        for stage in trap:
            if stage not in _STAGES:
                report_lines.append(f"  - [UNTRACKED-STAGE] {trap[stage]}")
        assert not trap, "\n".join(report_lines)


def test_racaso_stages_stay_finite_with_bursty_x_regime():
    """Bursty regime: every 7th step, gradient magnitude spikes 10x.

    Real X-organ gradients spike on phase transitions or rare tokens.
    Tests whether the L1 spread cap + L2 eigh-residual gate absorb
    bursts without producing NaN downstream.
    """
    p = _make_x_param(seed=11)
    opt = RACASO(
        [p],
        lr=1e-5, betas=(0.965, 0.99), shampoo_beta=0.95,
        rho=0.04, gamma=0.04,
        refresh_freq=10, hessian_freq=10,
        eigh_residual_threshold=0.5, spread_cap=10.0,
        radam_enabled=True,
    )

    RACASO._stage_trap = {}
    try:
        K = 4
        for step in range(1, 16):
            for k in range(K):
                opt.zero_grad()
                g = _synthetic_x_gradient(p, step, k, seed=11)
                if step % 7 == 0:
                    g = g * 10.0
                p.grad = g
                opt.step()
        trap = dict(RACASO._stage_trap)
    finally:
        RACASO._stage_trap = None

    if trap:
        lines = ["RACASO non-finite under bursty regime:"]
        for stage in _STAGES:
            if stage in trap:
                lines.append(f"  - {trap[stage]}")
        assert not trap, "\n".join(lines)
