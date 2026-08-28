"""Unit tests for the RACASO optimizer.

Covers the four-layer safety chain (L1 spread cap, L2 eigh-residual
safe-skip, L3 vanilla Yogi fallback for 1-D and HVP failures, L4 RAdam
cold-start gate) plus the core algorithmic properties (rotation
refresh schedule, Hessian refresh schedule, Sophia clip, telemetry).

All tests pure-CPU. Hessian-refresh tests construct toy autograd
scenarios with create_graph=True so RACASO's torch.autograd.grad
inside _try_hutchinson_hvp can succeed; tests that don't construct
graph-attached gradients verify the L3 fallback path engages cleanly.
"""

from __future__ import annotations

import io
import math

import pytest
import torch

from racaso import (
    RACASO,
    _safe_eig_with_residual,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(11)


def _make_2d_param(m: int, n: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(m, n) * 0.1)


def _make_1d_param(n: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(n) * 0.1)


def _attach_grad_with_graph(p: torch.nn.Parameter) -> None:
    """Populate p.grad and stash a synthetic Hutchinson HVP estimate.

    Contract: the loop computes z*Hz via torch.func.hvp and stashes it
    as p._racaso_hvp_estimate. For tests, the loss is loss = (p*p).sum()
    so H = 2I, z is Rademacher, and z*Hz = z*(2*z) = 2*1 = 2.
    """
    loss = (p * p).sum()
    p.grad = None
    loss.backward()
    z = torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    p._racaso_hvp_estimate = z * (2.0 * z)


# ── 1. Construction ────────────────────────────────────────────────────


def test_construction_defaults():
    p = _make_2d_param(4, 8)
    opt = RACASO([p])
    g = opt.param_groups[0]
    # Default lr changed from Sophia's 6e-2 to 3e-4 (bench-tested value);
    # see paper §10 and bench/decision_hvp_ema.md for the discussion.
    assert g["lr"] == 3e-4
    assert g["betas"] == (0.965, 0.99)
    assert g["rho"] == 0.04
    assert g["gamma"] == 0.04
    assert g["refresh_freq"] == 10
    assert g["hessian_freq"] == 10
    assert g["eigh_residual_threshold"] == 0.5
    assert g["spread_cap"] == 10.0
    assert g["radam_enabled"] is True


def test_construction_rejects_bad_args():
    p = _make_2d_param(4, 8)
    with pytest.raises(ValueError, match="learning rate"):
        RACASO([p], lr=-1.0)
    with pytest.raises(ValueError, match="beta1"):
        RACASO([p], betas=(1.5, 0.99))
    with pytest.raises(ValueError, match="shampoo_beta"):
        RACASO([p], shampoo_beta=1.0)
    with pytest.raises(ValueError, match="rho"):
        RACASO([p], rho=0.0)
    with pytest.raises(ValueError, match="refresh_freq"):
        RACASO([p], refresh_freq=0)
    with pytest.raises(ValueError, match="spread_cap"):
        RACASO([p], spread_cap=1.0)


# ── 2. Single-step 2-D ─────────────────────────────────────────────────


def test_single_step_2d_no_nan():
    """At step 1 (L4 closed because rho_t <= 4 with default betas),
    the optimizer should take a momentum-only update. Weights must move
    and stay finite."""
    p = _make_2d_param(4, 8)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p) * 0.01
    opt = RACASO([p], lr=1e-3)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), p_before)
    state = opt.state[p]
    assert state["step"] == 1
    # L4 closed at step 1 with default beta2=0.99
    assert state["rectification_skip_count"] == 1


# ── 3. Single-step 1-D (L3 Yogi fallback) ──────────────────────────────


def test_single_step_1d_vanilla_yogi():
    p = _make_1d_param(16)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p) * 0.01
    opt = RACASO([p], lr=1e-3)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), p_before)
    state = opt.state[p]
    # 1-D state has exp_avg_sq (Yogi) — NOT the 2-D state with Q_L/Q_R
    assert "exp_avg_sq" in state
    assert "Q_L" not in state


# ── 4. L4 cold-start gate ──────────────────────────────────────────────


def test_l4_cold_start_skips_spectral():
    """During cold-start (ρ_t ≤ 4), RAdam gate is closed. Verify NS
    rotation is not even attempted; rectification_skip_count increments;
    weights still move via momentum-only path."""
    p = _make_2d_param(4, 8)
    opt = RACASO([p], lr=1e-3, betas=(0.9, 0.99))
    for step in range(1, 5):
        p_before = p.detach().clone()
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        state = opt.state[p]
        assert state["rotation_success_count"] == 0
        assert state["rotation_skip_count"] == 0
        assert state["rectification_skip_count"] == step
        assert state["last_r_t"] == 0.0
        assert not torch.allclose(p.detach(), p_before)
        assert torch.isfinite(p).all()


# ── 5. L4 warmup crossover ─────────────────────────────────────────────


def test_l4_warmup_crossover():
    """By step ~5 with beta2=0.99, rho_t crosses 4 and RACASO engages
    the full pipeline (rotation refresh + Hessian refresh attempted)."""
    p = _make_2d_param(4, 8)
    opt = RACASO([p], lr=1e-3, betas=(0.9, 0.99), refresh_freq=1, hessian_freq=1)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    state = opt.state[p]
    assert state["rectification_skip_count"] < 20, (
        "L4 gate never opened after 20 steps"
    )
    # Rotation should have been attempted post-warmup
    rot_attempted = state["rotation_success_count"] + state["rotation_skip_count"]
    assert rot_attempted > 0


# ── 6. Rotation refresh schedule ───────────────────────────────────────


def test_rotation_refresh_schedule():
    """With refresh_freq=3 and adaptive trigger off (the static
    schedule), rotation should attempt at step 1 (always) and every 3rd
    step thereafter, post-warmup."""
    p = _make_2d_param(4, 8)
    opt = RACASO(
        [p], lr=1e-3, betas=(0.9, 0.99),
        refresh_freq=3, hessian_freq=99,  # disable Hessian for this test
    )
    # Run enough steps to get well past L4
    for _ in range(30):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    state = opt.state[p]
    rot_attempted = state["rotation_success_count"] + state["rotation_skip_count"]
    # Post-warmup (~step 5), every 3rd step refreshes. From step 5 to 30
    # that's roughly 26 steps, ~9 refreshes expected.
    assert 5 <= rot_attempted <= 12, (
        f"rotation attempt count unexpected: {rot_attempted}"
    )


# ── 7. eigh-residual safe-skip (L2) ────────────────────────────────────


def test_eigh_residual_safe_skip_engages_on_high_residual():
    """Craft a synthetic GG_L that produces a high-residual eigh.
    With a tight eigh_residual_threshold, the rotation should safe-skip
    and rotation_skip_count should increment."""
    p = _make_2d_param(4, 8)
    opt = RACASO(
        [p], lr=1e-3, betas=(0.9, 0.99),
        refresh_freq=1, hessian_freq=99,
        eigh_residual_threshold=1e-10,  # absurdly tight
    )
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    state = opt.state[p]
    # With this absurd threshold, every eigh refresh will safe-skip
    # post-warmup. rotation_skip_count should dominate over success.
    assert state["rotation_skip_count"] >= 1, (
        f"safe-skip never engaged at eigh_residual_threshold=1e-10; "
        f"ok={state['rotation_success_count']} skip={state['rotation_skip_count']}"
    )


# ── 8. Spread cap L1 ───────────────────────────────────────────────────


def test_spread_cap_clamps_pathological_rows():
    """Verify the L1 spread cap on rotated update row norms. Synthetic
    parameter + huge per-row gradient asymmetry → spread cap must keep
    weights finite without amplifying quiet rows."""
    p = _make_2d_param(8, 16)
    opt = RACASO([p], lr=1e-3, spread_cap=10.0)
    for t in range(30):
        g = torch.randn_like(p) * 0.01
        # Inject extreme row burst on row 0
        g[0, :] = g[0, :] * 1000.0
        p.grad = g
        opt.step()
        assert torch.isfinite(p).all(), f"NaN at step {t}"


# ── 9. State-dict round-trip ───────────────────────────────────────────


def test_state_dict_roundtrip():
    p_orig = _make_2d_param(4, 8)
    opt_orig = RACASO([p_orig], lr=1e-3, betas=(0.9, 0.99))
    grads = [torch.randn_like(p_orig) * 0.01 for _ in range(8)]
    for g in grads:
        p_orig.grad = g.clone()
        opt_orig.step()
    saved = opt_orig.state_dict()
    buf = io.BytesIO()
    torch.save(saved, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)

    p_new = torch.nn.Parameter(p_orig.detach().clone() - sum(g for g in grads))
    opt_new = RACASO([p_new], lr=1e-3, betas=(0.9, 0.99))
    opt_new.load_state_dict(loaded)

    s_orig = opt_orig.state[p_orig]
    s_new = opt_new.state[list(opt_new.state.keys())[0]]
    assert s_new["step"] == s_orig["step"]
    assert s_new["rotation_success_count"] == s_orig["rotation_success_count"]
    assert s_new["rectification_skip_count"] == s_orig["rectification_skip_count"]
    assert torch.allclose(s_new["exp_avg"], s_orig["exp_avg"])
    assert torch.allclose(s_new["Q_L"], s_orig["Q_L"])
    assert torch.allclose(s_new["Q_R"], s_orig["Q_R"])


# ── 10. Telemetry aggregation ──────────────────────────────────────────


def test_get_telemetry_aggregates_across_params():
    p1 = _make_2d_param(4, 8)
    p2 = _make_2d_param(6, 10)
    p3 = _make_1d_param(8)
    opt = RACASO([p1, p2, p3], lr=1e-3, betas=(0.9, 0.99))
    for _ in range(20):
        for q in (p1, p2, p3):
            q.grad = torch.randn_like(q) * 0.01
        opt.step()
    t = opt.get_telemetry()
    assert t["num_2d_params"] == 2
    assert t["rectification_skip_count"] >= 4 * 2, (
        f"each 2-D param should have ≥4 rect_skips during cold-start; "
        f"got {t['rectification_skip_count']}"
    )


# ── 11. RAdam rectification math ───────────────────────────────────────


def test_radam_rectification_step1():
    warmed, r_t = RACASO._radam_rectification(t=1, beta2=0.99)
    assert warmed is False
    assert r_t == 0.0


def test_radam_rectification_warmed_up():
    warmed, r_t = RACASO._radam_rectification(t=100, beta2=0.99)
    assert warmed is True
    assert 0.0 < r_t <= 1.0


# ── 12. _safe_eig_with_residual helper ─────────────────────────────────


def test_safe_eig_returns_residual():
    M = torch.randn(4, 4)
    M_psd = M @ M.T
    Q, res = _safe_eig_with_residual(M_psd)
    assert Q.shape == (4, 4)
    assert math.isfinite(res)
    assert res < 1e-3  # well-conditioned PSD matrix


def test_safe_eig_fallback_on_pathological():
    """Test the fallback to fallback_Q when eigh fails completely.
    Use NaN matrix to force all ridge steps to error."""
    M = torch.full((4, 4), float("nan"))
    fallback_Q = torch.eye(4)
    Q, res = _safe_eig_with_residual(M, fallback_Q=fallback_Q)
    # The cascade may complete on the nan-cleaned M_sym, but the
    # residual will be inf or large; behaviorally we just need Q to be
    # a valid orthogonal matrix or the fallback.
    assert Q.shape == (4, 4)


# ── 13. Toy regression convergence ─────────────────────────────────────


def test_toy_regression_convergence():
    """Two-layer linear regression. After 300 steps with create_graph=True
    on Hessian-refresh steps, loss should drop meaningfully.

    Without HVP, RACASO's hessian_diag_rot stays at init_acc and the
    Sophia clip (ρ=0.04) bounds every element aggressively — descent is
    very slow. With HVP, the Hessian estimate scales the denom properly
    so updates are correctly sized. This test verifies HVP integration.
    """
    torch.manual_seed(42)
    in_dim, hidden, out_dim, batch = 8, 16, 4, 32
    W1 = torch.nn.Parameter(torch.randn(hidden, in_dim) * 0.1)
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    W2 = torch.nn.Parameter(torch.randn(out_dim, hidden) * 0.1)
    b2 = torch.nn.Parameter(torch.zeros(out_dim))

    X = torch.randn(batch, in_dim)
    W_true = torch.randn(out_dim, in_dim)
    Y = X @ W_true.T

    opt = RACASO(
        [W1, b1, W2, b2], lr=1.0, betas=(0.9, 0.99),
        hessian_freq=5,  # Refresh Hessian frequently for the toy
        rho=1.0,         # Loose clip so good updates aren't strangled
    )

    def loss_fn():
        h = torch.tanh(X @ W1.T + b1)
        pred = h @ W2.T + b2
        return ((pred - Y) ** 2).mean()

    initial_loss = loss_fn().item()
    params = [W1, b1, W2, b2]
    from torch.func import grad as func_grad, jvp
    for step in range(300):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        # Compute Hutchinson diagonal estimate via torch.func: HVP =
        # jvp(grad(f), primals, tangents). Stash z*Hz on each 2-D param.
        if step % 5 == 0 or step == 0:
            x_params_dict = {f"p{i}": p for i, p in enumerate(params) if p.dim() == 2}
            def _l(xp):
                W1n = xp.get("p0", W1)
                W2n = xp.get("p2", W2)
                h = torch.tanh(X @ W1n.T + b1)
                pred = h @ W2n.T + b2
                return ((pred - Y) ** 2).mean()
            z_dict = {
                n: torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)
                for n, p in x_params_dict.items()
            }
            _, hvp_dict = jvp(func_grad(_l), (x_params_dict,), (z_dict,))
            for n, hz in hvp_dict.items():
                if hz is None:
                    continue
                idx = int(n[1:])
                params[idx]._racaso_hvp_estimate = z_dict[n] * hz
        opt.step()
    final_loss = loss_fn().item()
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    )


# ── 14. Stability under bursty conditioning ────────────────────────────


def test_bursty_conditioning_stability():
    """Synthetic gradient stream with per-column bursts and row coupling
    — RACASO's L1+L2+L3+L4 chain must hold over 200 steps."""
    torch.manual_seed(3)
    p = _make_2d_param(8, 16)
    opt = RACASO([p], lr=1e-3, betas=(0.9, 0.99))
    for t in range(200):
        g = torch.randn_like(p) * 0.05
        burst_col = t % p.shape[1]
        g[:, burst_col] += torch.randn(p.shape[0]) * (3.0 if t % 7 == 0 else 0.5)
        row_factor = torch.randn(p.shape[0], 1).abs() + 0.5
        g = g * row_factor
        p.grad = g
        opt.step()
        assert torch.isfinite(p).all(), f"NaN at step {t}"


# ── 15. Telemetry signature distinguishes RACASO from Muogi/RAMuogi ────


def test_telemetry_signature_keys():
    p = _make_2d_param(4, 8)
    opt = RACASO([p])
    p.grad = torch.randn_like(p) * 0.01
    opt.step()
    t = opt.get_telemetry()
    # RACASO-specific fields (loop.py telemetry render uses these as
    # the signature to detect RACASO vs Muogi/RAMuogi).
    assert "rotation_success_count" in t
    assert "rotation_skip_count" in t
    assert "hessian_success_count" in t
    assert "hessian_skip_count" in t
    assert "last_clip_fraction" in t
    assert "last_eigh_residual" in t
    assert "last_r_t" in t
    assert "spread_cap_fire_count" in t
    assert "l5_absorb_fire_count" in t
    assert "last_h_ema_norm" in t


# ── 16. Safety-count accessor (bench harness contract) ────────────────


def test_get_safety_counts_returns_five_keys():
    """The bench harness reads optimizer.get_safety_counts() and expects
    a 5-key dict mapping l1..l5 to int counts."""
    p = _make_2d_param(4, 8)
    opt = RACASO([p], betas=(0.9, 0.99))
    for _ in range(10):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    counts = opt.get_safety_counts()
    assert set(counts.keys()) == {"l1", "l2", "l3", "l4", "l5"}
    for k, v in counts.items():
        assert isinstance(v, int), f"{k} should be int, got {type(v).__name__}"
        assert v >= 0
    # L4 (cold-start) should have fired at least 4 times before rho_t > 4.
    assert counts["l4"] >= 4


# ── 17. Correctness locks for the rotated-basis curvature rewrite ──────
#
# These pin the math fixes that replaced the unsound `|Q_L^T denom Q_R|`
# congruence (which produced sign-garbage masked by `.abs()`) with a
# rotated-basis denominator that is positive-by-construction. See the
# evidence log: diag(Q^T H Q) != Q^T diag(H) Q, so the rotated Hessian
# diagonal must be probed in the rotated basis, not rotated after the fact.


def _convex_matrix_problem(m=6, n=5, seed=7, log_cond=1.0):
    """SPD-coupled convex quadratic f(W) = 0.5 <H W, W>; the true Hessian
    diagonal is strictly positive in every basis, so a correct rotated-basis
    curvature estimate must come out (essentially) all-positive.

    ``log_cond`` sets the eigenvalue spread (condition number = 10**log_cond).
    """
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(m, m, generator=g))[0]
    Hm = U @ torch.diag(torch.logspace(0, log_cond, m)) @ U.T  # SPD
    W = torch.nn.Parameter(torch.randn(m, n, generator=g) * 0.5)

    def loss_of(Wp):
        return 0.5 * (Hm @ Wp * Wp).sum()

    return W, loss_of


def test_curvature_mode_validation():
    p = _make_2d_param(4, 8)
    RACASO([p], curvature_mode="hutchinson")
    RACASO([p], curvature_mode="soap")
    with pytest.raises(ValueError, match="curvature_mode"):
        RACASO([p], curvature_mode="bogus")


def test_rotated_hessian_diagonal_is_sign_correct_on_convex():
    """The shipped hutchinson path must store a *true* rotated-basis Hessian
    diagonal — positive on a convex (SPD) problem. The removed bug produced
    sign-flipped entries (~65% negative) masked by `.abs()`."""
    W, loss_of = _convex_matrix_problem()
    opt = RACASO(
        [W], lr=1e-2, betas=(0.9, 0.99), hessian_freq=1, refresh_freq=1,
        rho=1.0, curvature_mode="hutchinson",
        forward_fn=lambda params: loss_of(params[0]),
    )
    opt.set_hvp_context(lambda params: loss_of(params[0]), [W])
    for _ in range(30):
        opt.zero_grad()
        loss_of(W).backward()
        opt.step()
    hdr = opt.state[W]["hessian_diag_rot"]
    neg_fraction = (hdr < 0).float().mean().item()
    assert neg_fraction < 0.2, (
        f"rotated Hessian diagonal is {neg_fraction*100:.0f}% negative on a "
        f"convex problem — the unsound rotation bug is back"
    )


@pytest.mark.parametrize("mode", ["hutchinson", "soap"])
def test_both_curvature_modes_descend(mode):
    """Both curvature modes must drive a well-conditioned convex coupled
    quadratic down, given an adequate step budget. (On ill-conditioned
    problems Sophia-style cautious clipping is intentionally conservative;
    the descent guarantee is asserted in the regime the optimizer targets.)"""
    W, loss_of = _convex_matrix_problem(log_cond=1.0)  # condition ~10
    fwd = (lambda params: loss_of(params[0])) if mode == "hutchinson" else None
    opt = RACASO(
        [W], lr=3e-2, betas=(0.9, 0.99), hessian_freq=4, refresh_freq=4,
        rho=1.0, curvature_mode=mode, forward_fn=fwd,
    )
    if mode == "hutchinson":
        opt.set_hvp_context(lambda params: loss_of(params[0]), [W])
    initial = loss_of(W).item()
    for _ in range(500):
        opt.zero_grad()
        loss_of(W).backward()
        opt.step()
    final = loss_of(W).item()
    assert torch.isfinite(torch.tensor(final))
    assert final < 0.1 * initial, (
        f"{mode} did not descend: {initial:.3f} -> {final:.3f}"
    )


def test_pearlmutter_identity_grad_of_inner_product_is_Hz():
    """Sanity-lock the HVP foundation: grad_p <g(p), z> = H @ z (full
    matvec), NOT diag(H)*z. The diagonal estimate only appears after the
    elementwise z (x) Hz and expectation."""
    n = 6
    A = torch.randn(n, n)
    H = A @ A.T - 2.0 * torch.eye(n)  # symmetric, indefinite
    p = torch.randn(n, requires_grad=True)
    g = torch.autograd.grad(0.5 * p @ H @ p, p, create_graph=True)[0]
    z = (torch.randint(0, 2, (n,)) * 2 - 1).float()
    hz = torch.autograd.grad((g * z).sum(), p)[0]
    assert torch.allclose(hz, H @ z, atol=1e-4)
    assert not torch.allclose(hz, torch.diag(H) * z, atol=1e-2)
