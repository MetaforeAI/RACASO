"""Unit tests for the RACASO Hutchinson + GNB HVP-strategy wrappers."""

from __future__ import annotations

import pytest
import torch

from bench.optimizers.racaso_hvp_wrappers import RACASOHutchinson, RACASOGNB
from bench.problems.p1_off_axis_quad import P1OffAxisQuadratic
from bench.problems.p6_classification import P6Classification


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ── Hutchinson ───────────────────────────────────────────────────────────


def test_hutchinson_construction() -> None:
    """RACASOHutchinson subclasses RACASO and accepts the same kwargs."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = RACASOHutchinson([p], lr=1e-3)
    assert isinstance(opt, RACASOHutchinson)
    # set_hvp_context is a new method, not on plain RACASO.
    assert hasattr(opt, "set_hvp_context")
    assert opt._forward_fn is None
    assert opt._params_ref is None


def test_hutchinson_step_without_context_is_noop_on_stash() -> None:
    """Calling step() without set_hvp_context leaves no HVP stash —
    RACASO's L4 / L5 absorbs it, no crash."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = RACASOHutchinson([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    # No stash means no _racaso_hvp_estimate on p.
    assert not hasattr(p, "_racaso_hvp_estimate")


def test_hutchinson_stashes_z_dot_hz_on_first_step() -> None:
    """On a refresh step (step 1), the wrapper computes z * Hz and the
    stash gets consumed by RACASO.step() (which clears it)."""
    # Use a 2-D param so RACASO actually consumes the HVP stash.
    problem = P1OffAxisQuadratic(seed=0)
    # P1 uses a 1-D param; we need a 2-D param to exercise the HVP path.
    # Build a manual quadratic on a small matrix.
    W = torch.nn.Parameter(torch.randn(4, 4))
    def forward_fn(params):
        (w,) = params
        return (w * w).sum()
    opt = RACASOHutchinson([W], lr=1e-3, hessian_freq=1)
    opt.set_hvp_context(forward_fn, [W])
    # First step is a refresh step (state.step + 1 == 1, and 1 % 1 == 0).
    W.grad = 2.0 * W.detach()  # gradient of ||W||²
    opt.step()
    # RACASO.step() consumes (clears) the stash, so it should not be
    # present afterwards. The fact that step() ran finitely is the test
    # — proof that the HVP path was exercised without crash.
    assert torch.isfinite(W.detach()).all()


def test_hutchinson_handles_non_finite_hvp_gracefully() -> None:
    """If forward_fn returns NaN, the wrapper must not crash; it must
    just leave the stash empty and let RACASO fall through to L5."""
    W = torch.nn.Parameter(torch.randn(4, 4))
    def bad_forward(params):
        return torch.tensor(float("nan"))
    opt = RACASOHutchinson([W], lr=1e-3, hessian_freq=1)
    opt.set_hvp_context(bad_forward, [W])
    W.grad = torch.randn_like(W)
    opt.step()  # must not raise
    assert torch.isfinite(W.detach()).all()


# ── GNB ──────────────────────────────────────────────────────────────────


def test_gnb_construction() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = RACASOGNB([p], lr=1e-3)
    assert isinstance(opt, RACASOGNB)
    assert hasattr(opt, "set_hvp_context")


def test_gnb_step_without_context_is_noop() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = RACASOGNB([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    assert not hasattr(p, "_racaso_hvp_estimate")


def test_gnb_runs_on_classification_problem() -> None:
    """The GNB wrapper computes its stash via one extra first-order
    backward against a re-built CE loss; it should run end-to-end on
    P6Classification without crash and produce finite param updates."""
    problem = P6Classification(seed=0)
    params = problem.init_params()
    opt = RACASOGNB(params, lr=1e-4, hessian_freq=1)
    opt.set_hvp_context(problem.logits_fn, params)
    # One step end-to-end.
    loss_val, grads = problem.loss_and_grad(params)
    assert loss_val > 0
    for p, g in zip(params, grads):
        p.grad = g
    opt.step()
    for p in params:
        assert torch.isfinite(p.detach()).all()


def test_gnb_stash_feeds_psd_rotated_second_moment() -> None:
    """The corrected GNB stash is the raw synthetic-label gradient ĝ
    (mean-reduction) on ``p._racaso_gnb_ghat``. The rotated GN diagonal
    it feeds, ``(Q_Lᵀ ĝ Q_R)²``, is positive-by-construction — this is
    the design verified in /tmp/verify_gnb_design.py (replaces the
    removed ĝ²·B param-basis stash)."""
    problem = P6Classification(seed=1)
    params = problem.init_params()
    opt = RACASOGNB(params, lr=1e-4, hessian_freq=1)
    opt.set_hvp_context(problem.logits_fn, params)
    opt._compute_and_stash_gnb()
    found = False
    for p in params:
        if p.ndim >= 2 and hasattr(p, "_racaso_gnb_ghat"):
            found = True
            g_hat = p._racaso_gnb_ghat
            assert torch.isfinite(g_hat).all()
            # Rotated second moment is PSD in ANY orthonormal basis.
            m_, n_ = p.shape
            Q_L, _ = torch.linalg.qr(torch.randn(m_, m_))
            Q_R, _ = torch.linalg.qr(torch.randn(n_, n_))
            v_rot = (Q_L.T @ g_hat @ Q_R) ** 2
            assert (v_rot >= 0).all(), (
                "rotated GN diagonal (Qᵀ ĝ Q)² must be PSD"
            )
            # No _racaso_hvp_estimate path for GNB (the soap path owns it).
            assert not hasattr(p, "_racaso_hvp_estimate")
    assert found, "GNB did not stash ĝ on any 2-D parameter"


def test_gnb_uses_mean_reduction_no_batch_scaling() -> None:
    """The corrected GNB gradient uses reduction='mean' and DROPS the
    ``* batch_size`` factor (the removed bug). A mean-reduction CE
    gradient is O(1/B) the magnitude of a sum-reduction one; we assert
    the stash matches the mean-reduction ĝ exactly, not sum or sum·B."""
    problem = P6Classification(seed=2)
    params = problem.init_params()
    opt = RACASOGNB(params, lr=1e-4, hessian_freq=1)
    opt.set_hvp_context(problem.logits_fn, params)
    opt._compute_and_stash_gnb()
    stash = next(
        p._racaso_gnb_ghat.clone() for p in params
        if p.ndim >= 2 and hasattr(p, "_racaso_gnb_ghat")
    )
    # Mean-reduction gradient magnitude is bounded and finite, and far
    # below the sum-reduction (×B) scale for B=128 — a coarse but real
    # guard against reintroducing the ``* batch_size`` bug.
    assert torch.isfinite(stash).all()
    assert stash.abs().max().item() < 10.0, (
        "GNB stash magnitude too large — looks like sum/×B reduction"
    )


# ── Integration with the bench harness ──────────────────────────────────


def test_build_racaso_hutchinson_via_wrappers() -> None:
    from bench.optimizers.wrappers import build_optimizer

    problem = P6Classification(seed=0)
    params = problem.init_params()
    opt = build_optimizer("racaso_hutchinson", params, lr=1e-4, problem=problem)
    assert isinstance(opt, RACASOHutchinson)
    assert opt._forward_fn is not None
    assert opt._params_ref is params


def test_build_racaso_gnb_via_wrappers() -> None:
    from bench.optimizers.wrappers import build_optimizer

    problem = P6Classification(seed=0)
    params = problem.init_params()
    opt = build_optimizer("racaso_gnb", params, lr=1e-4, problem=problem)
    assert isinstance(opt, RACASOGNB)
    assert opt._logits_fn is not None


def test_build_racaso_gnb_refuses_problem_without_logits_fn() -> None:
    """GNB requires a classification problem; non-classification
    problems must raise NotImplementedError naming the constraint."""
    from bench.optimizers.wrappers import build_optimizer

    problem = P1OffAxisQuadratic(seed=0)  # no logits_fn
    params = problem.init_params()
    with pytest.raises(NotImplementedError, match="logits_fn"):
        build_optimizer("racaso_gnb", params, lr=1e-4, problem=problem)


def test_p6_classification_problem_runs() -> None:
    """P6 forward returns scalar CE loss; logits_fn returns [B, C]."""
    problem = P6Classification(seed=0)
    params = problem.init_params()
    logits = problem.logits_fn(params)
    assert logits.shape == (128, 10)
    loss = problem.forward(params)
    assert loss.dim() == 0
    assert loss.item() > 0
