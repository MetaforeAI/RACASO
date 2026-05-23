"""RACASO — Rotation-Aligned Cautious Approximately Second-Order Optimizer.

by Richard I Christopher, 2026

A composition of three published methods, engineered with a four-layer
safety chain in the Muogi/RAMuogi lineage:

    R(otation-Aligned A)xes — CASPR/Shampoo eigenbasis Q_L, Q_R, the
        privileged coordinate system aligned with the parameter's
        natural curvature axes (eigenbasis of g·gᵀ and gᵀ·g).

    (C)autious — Sophia's per-element clip ±ρ applied IN THE ROTATED
        BASIS. Bounds per-eigendirection step magnitude regardless of
        Hessian estimate quality. The safety net for residual cross-
        coupling the rotation cannot fully diagonalize.

    (A)pproximately (S)econd-(O)rder — Hutchinson HVP diagonal Hessian
        estimate in the rotated basis. Periodic refresh (every
        ``hessian_freq`` steps) via ``torch.autograd.grad`` on the
        existing ``p.grad`` graph (caller must have backpropped with
        ``create_graph=True`` on those steps).

    (O)ptimizer — composes the above with momentum, weight decay, and
        the four-layer safety chain.

## Why RACASO for cross-branch aggregation specifically

In architectures with joint-norm denominators that couple gradients
across rows and columns (cross-branch aggregation surfaces, shared-
denominator normalization layers, multi-stream attention output
projections with a shared scaling factor), the gradient covariance
violates SOAP's Kronecker assumption (Σ ≈ Σ_L ⊗ Σ_R, row dependencies
independent of column dependencies) — eigh refresh on
``GG_L``/``GG_R`` hits progressively ill-conditioned matrices and the
fallback chain eventually collapses. RAMuogi handled the coupling
numerically but
over-empowered X's spectral side, producing register collapse in
qualitative samples by step 4000+ despite continued loss descent.

RACASO addresses both: keep CASPR/Shampoo's rotation (the spectral-
balancing argument that motivated trying second-order methods), but
run Sophia's cautious step in the rotated basis (per-element clip
ρ catches the residual off-diagonal energy the rotation cannot
absorb). Hutchinson HVP gives a real Hessian estimate, not just a
gradient-squared proxy.

## The four-layer safety chain

  L1 — spread cap on rotated update's row norms. Bounds per-
       eigendirection step spread after Sophia's clip.
  L2 — eigh residual threshold on rotation refresh. If a refresh
       produces an eigh result whose residual ‖M·Q − Q·Λ‖_F exceeds
       threshold, keep previous Q_L/Q_R.
  L3 — vanilla Yogi fallback for 1-D params AND for HVP failures
       (missing graph, autograd.grad raised). Always produces a
       finite weight update.
  L4 — RAdam variance-confidence gate. When ρ_t ≤ 4 (RAdam math),
       skip rotation + Hessian + clip; apply momentum-only update.
       Mirrors RAMuogi's L4 exactly.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


def _safe_eig_with_residual(
    M: torch.Tensor,
    fallback_Q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Eigendecomposition of symmetric PSD M with progressive ridge
    fallback. Returns ``(Q, residual)`` where residual is the Frobenius
    error of the eigendecomposition ``‖M_sym − Q·diag(λ)·Qᵀ‖_F``.

    Caller's L2 uses the residual to decide whether to accept the new
    Q. Progressive ridges tried: ``0, 1e-6, 1e-3, 1e-1``. If all fail
    OR produce NaN/Inf in Q or eigvals, returns ``(fallback_Q, inf)``
    to signal "do not trust this refresh."

    NaN trap: PyTorch's ``linalg.eigh`` does NOT always raise on
    rank-deficient or near-singular inputs — it can silently return
    eigenvectors with NaN columns for the null space. We explicitly
    check finiteness and continue to the next ridge if so.
    """
    n = M.shape[-1]
    M_sym = 0.5 * (M + M.T)
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    for ridge_scale in (0.0, 1e-6, 1e-3, 1e-1):
        try:
            eigvals, Q = torch.linalg.eigh(M_sym + ridge_scale * eye)
        except Exception:
            continue
        # Hard NaN/Inf gate — eigh can return non-finite Q without raising.
        if not (torch.isfinite(eigvals).all() and torch.isfinite(Q).all()):
            continue
        recon = Q @ torch.diag(eigvals) @ Q.T
        residual = float((M_sym - recon).norm().item())
        if not math.isfinite(residual):
            continue
        return Q, residual
    if fallback_Q is not None:
        return fallback_Q, float("inf")
    return eye, float("inf")


class RACASO(Optimizer):
    """Rotation-Aligned Cautious Approximately Second-Order Optimizer.

    See module docstring for the algorithm composition and safety
    chain. For 2-D parameters, runs the full pipeline (rotation +
    Hessian + clip). For 1-D parameters (norms, biases, learned
    scalars), falls back to vanilla Yogi via L3.

    Constructor defaults match Sophia's reference paper (lr=6e-2,
    betas=(0.965, 0.99), rho=0.04, gamma=0.04) plus SOAP-style
    shampoo_beta=0.95 for the Kronecker covariance EMA.

    Class-level contract flag ``_optimizer_handles_own_clip = True``:
    the loop's per-organ pre-step soft-clip block (which would
    in-place-mutate ``p.grad`` and break the autograd graph RACASO
    needs for HVP) skips this optimizer's organ. RACASO's L1 spread
    cap inside step() handles the equivalent magnitude bounding —
    on the rotated update's row norms, not on the raw gradient norm.
    """

    # Loop contract: this optimizer handles its own gradient clipping
    # inside step() and the loop should NOT pre-mutate p.grad with the
    # per-organ soft-clip. The pre-clip would otherwise break the graph
    # RACASO needs for Hutchinson HVP via torch.autograd.grad on p.grad.
    # L1 (spread cap on rotated update row norms inside RACASO.step) is
    # the magnitude bound that replaces the loop's per-organ clip.
    _optimizer_handles_own_clip: bool = True


    def __init__(
        self,
        params,
        lr: float = 6e-2,
        betas: Tuple[float, float] = (0.965, 0.99),
        shampoo_beta: float = 0.95,
        eps: float = 1e-12,
        eps_adam: float = 1e-8,
        eps_yogi: float = 1e-3,
        rho: float = 0.04,
        gamma: float = 0.04,
        weight_decay: float = 0.0,
        refresh_freq: int = 10,
        hessian_freq: int = 10,
        eigh_residual_threshold: float = 0.5,
        spread_cap: float = 10.0,
        radam_enabled: bool = True,
        initial_accumulator: float = 1e-6,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid RACASO learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid RACASO beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid RACASO beta2: {betas[1]}")
        if not 0.0 <= shampoo_beta < 1.0:
            raise ValueError(f"Invalid RACASO shampoo_beta: {shampoo_beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if eps_adam < 0.0:
            raise ValueError(f"Invalid eps_adam: {eps_adam}")
        if eps_yogi <= 0.0:
            raise ValueError(f"Invalid eps_yogi: {eps_yogi}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        if gamma <= 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if refresh_freq < 1:
            raise ValueError(f"Invalid refresh_freq: {refresh_freq}")
        if hessian_freq < 1:
            raise ValueError(f"Invalid hessian_freq: {hessian_freq}")
        if eigh_residual_threshold <= 0.0:
            raise ValueError(
                f"Invalid eigh_residual_threshold: {eigh_residual_threshold}"
            )
        if spread_cap <= 1.0:
            raise ValueError(f"Invalid spread_cap (must be > 1): {spread_cap}")
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            eps_adam=eps_adam,
            eps_yogi=eps_yogi,
            rho=rho,
            gamma=gamma,
            weight_decay=weight_decay,
            refresh_freq=refresh_freq,
            hessian_freq=hessian_freq,
            eigh_residual_threshold=eigh_residual_threshold,
            spread_cap=spread_cap,
            radam_enabled=radam_enabled,
            initial_accumulator=initial_accumulator,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _radam_rectification(t: int, beta2: float) -> Tuple[bool, float]:
        """RAdam variance-confidence gate (L4). Identical to RAMuogi.

        Returns ``(warmed_up, r_t)``. With β2=0.99 (Sophia default),
        ρ_∞ ≈ 199 and ρ_t crosses 4 at step ≈ 5.
        """
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        beta2_t = beta2 ** t
        rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)
        if rho_t <= 4.0:
            return False, 0.0
        r_t = (
            ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
        ) ** 0.5
        return True, r_t

    def _try_hutchinson_hvp(
        self,
        p: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Read a pre-computed Hutchinson HVP estimate from ``p``.

        Contract: on Hessian-refresh steps, the loop calls
        ``_compute_and_stash_racaso_hvp`` which uses ``torch.func.hvp``
        + ``functional_call`` to compute ``z * Hz`` (Hutchinson diagonal
        estimate) for each X-organ 2-D param and stashes it on the
        param as ``p._racaso_hvp_estimate``. We just read it.

        Returns ``z * Hz`` if stashed, ``None`` if missing
        (non-refresh step, or loop's hvp call failed). Always clears
        the stash on read so a stale estimate from a refresh step can't
        bleed into the next non-refresh step.

        The earlier autograd.grad approach hit a chain of PyTorch eager
        autograd issues (leaf-grad with no grad_fn → flash-CPU-SDPA has
        no 2nd derivative → saved tensor version mismatch from
        view+inplace ops). torch.func.hvp builds its own functional
        graph and sidesteps all three.
        """
        if not hasattr(RACASO, "_diag_skip_reasons"):
            RACASO._diag_skip_reasons = {
                "no_grad": 0, "no_grad_fn": 0,
                "runtime_err": 0, "hz_none": 0, "success": 0,
            }
        hvp_estimate = getattr(p, "_racaso_hvp_estimate", None)
        if hvp_estimate is None:
            RACASO._diag_skip_reasons["no_grad"] += 1
            return None
        try:
            delattr(p, "_racaso_hvp_estimate")
        except AttributeError:
            pass
        RACASO._diag_skip_reasons["success"] += 1
        return hvp_estimate

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            eps_adam = group["eps_adam"]
            eps_yogi = group["eps_yogi"]
            rho = group["rho"]
            gamma_scale = group["gamma"]
            wd = group["weight_decay"]
            refresh_freq = group["refresh_freq"]
            hessian_freq = group["hessian_freq"]
            eigh_res_threshold = group["eigh_residual_threshold"]
            spread_cap = group["spread_cap"]
            radam_enabled = group["radam_enabled"]
            init_acc = group["initial_accumulator"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("RACASO does not support sparse gradients")

                state = self.state[p]
                use_rotation = (g.ndim == 2)

                # ── Lazy init ────────────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    if use_rotation:
                        m, n = p.shape
                        state["hessian_diag_rot"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format
                        )
                        state["GG_L"] = torch.zeros(
                            m, m, device=p.device, dtype=p.dtype
                        )
                        state["GG_R"] = torch.zeros(
                            n, n, device=p.device, dtype=p.dtype
                        )
                        state["Q_L"] = torch.eye(
                            m, device=p.device, dtype=p.dtype
                        )
                        state["Q_R"] = torch.eye(
                            n, device=p.device, dtype=p.dtype
                        )
                        state["rotation_success_count"] = 0
                        state["rotation_skip_count"] = 0
                        state["hessian_success_count"] = 0
                        state["hessian_skip_count"] = 0
                        state["last_eigh_residual"] = 0.0
                        state["last_clip_fraction"] = 0.0
                        state["last_h_estimate_norm"] = 0.0
                    else:
                        state["exp_avg_sq"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format
                        )
                    state["rectification_skip_count"] = 0
                    state["last_r_t"] = 0.0

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]

                # ── L3: 1-D fallback path: vanilla Yogi ──────────────
                if not use_rotation:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    grad_sq = g * g
                    exp_avg_sq.addcmul_(
                        torch.sign(exp_avg_sq - grad_sq),
                        grad_sq,
                        value=-(1.0 - beta2),
                    )
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    denom = (exp_avg_sq / bc2).sqrt().clamp_(min=eps_yogi).add_(eps_adam)
                    p.addcdiv_(exp_avg / bc1, denom, value=-lr)
                    continue

                # ── 2-D path: full RACASO pipeline ───────────────────
                # Stage trip-wire: when RACASO._stage_trap is a dict,
                # record the (param_shape, step, stage, finite-summary)
                # of the first non-finite value found at each stage.
                # Zero-overhead when _stage_trap is None (default).
                _trap = getattr(RACASO, "_stage_trap", None)
                def _check(stage: str, tensor: torch.Tensor) -> None:
                    if _trap is None:
                        return
                    if torch.isfinite(tensor).all():
                        return
                    if stage in _trap:
                        return
                    n_nan = int(torch.isnan(tensor).sum().item())
                    n_inf = int(torch.isinf(tensor).sum().item())
                    n_tot = int(tensor.numel())
                    finite = tensor[torch.isfinite(tensor)]
                    fmax = float(finite.abs().max().item()) if finite.numel() else float("nan")
                    _trap[stage] = (
                        f"shape={tuple(p.shape)} step={t} stage={stage} "
                        f"nan={n_nan}/{n_tot} inf={n_inf}/{n_tot} "
                        f"finite_absmax={fmax:.3e}"
                    )

                # Sophia momentum (Adam-style β1 EMA on raw gradient)
                _check("pre_grad", g)
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                _check("exp_avg", exp_avg)

                # ── L4: RAdam cold-start gate ────────────────────────
                if radam_enabled:
                    warmed_up, r_t = self._radam_rectification(t, beta2)
                    state["last_r_t"] = r_t
                    if not warmed_up:
                        state["rectification_skip_count"] += 1
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        bc1 = 1.0 - beta1 ** t
                        p.add_(exp_avg, alpha=-lr / bc1)
                        continue
                else:
                    r_t = 1.0

                # ── Update Kronecker covariance EMAs every step ──────
                GG_L = state["GG_L"]
                GG_R = state["GG_R"]
                GG_L.mul_(shampoo_beta).addmm_(g, g.T, alpha=1.0 - shampoo_beta)
                GG_R.mul_(shampoo_beta).addmm_(g.T, g, alpha=1.0 - shampoo_beta)
                _check("GG_L", GG_L)
                _check("GG_R", GG_R)

                # ── L2: rotation refresh with eigh-residual safe-skip
                # Gate L and R independently — a bad eigh on one side
                # mustn't reject the other's good refresh, and a NaN
                # residual on one side mustn't pollute the max-of-both
                # comparison (max(nan, x) is order-dependent in Python).
                Q_L = state["Q_L"]
                Q_R = state["Q_R"]
                if t % refresh_freq == 0 or t == 1:
                    Q_L_new, eigh_res_L = _safe_eig_with_residual(
                        GG_L, fallback_Q=Q_L,
                    )
                    Q_R_new, eigh_res_R = _safe_eig_with_residual(
                        GG_R, fallback_Q=Q_R,
                    )
                    # NaN-safe gate: math.isfinite + bounded comparison.
                    L_ok = (math.isfinite(eigh_res_L) and
                            eigh_res_L < eigh_res_threshold and
                            torch.isfinite(Q_L_new).all().item())
                    R_ok = (math.isfinite(eigh_res_R) and
                            eigh_res_R < eigh_res_threshold and
                            torch.isfinite(Q_R_new).all().item())
                    # Telemetry: take the larger finite residual; if both
                    # are inf/nan, use inf so the diagnostic shows rejection.
                    if math.isfinite(eigh_res_L) and math.isfinite(eigh_res_R):
                        state["last_eigh_residual"] = max(eigh_res_L, eigh_res_R)
                    elif math.isfinite(eigh_res_L):
                        state["last_eigh_residual"] = eigh_res_L
                    elif math.isfinite(eigh_res_R):
                        state["last_eigh_residual"] = eigh_res_R
                    else:
                        state["last_eigh_residual"] = float("inf")
                    if L_ok:
                        state["Q_L"] = Q_L_new
                        Q_L = Q_L_new
                        _check("Q_L", Q_L)
                    if R_ok:
                        state["Q_R"] = Q_R_new
                        Q_R = Q_R_new
                        _check("Q_R", Q_R)
                    if L_ok and R_ok:
                        state["rotation_success_count"] += 1
                    else:
                        state["rotation_skip_count"] += 1

                # ── Rotate momentum into the privileged basis ────────
                m_rot = Q_L.T @ exp_avg @ Q_R
                _check("m_rot", m_rot)

                # ── Hessian-vector product diagonal estimate ─────────
                # Caller must have run backward with create_graph=True
                # on this step for the HVP to succeed. L3 catches None.
                hessian_diag_rot = state["hessian_diag_rot"]
                if t % hessian_freq == 0 or t == 1:
                    h_diag_param = self._try_hutchinson_hvp(p)
                    if h_diag_param is not None:
                        h_rot = Q_L.T @ h_diag_param @ Q_R
                        hessian_diag_rot.mul_(beta2).addcmul_(
                            h_rot, h_rot, value=1.0 - beta2,
                        )
                        state["last_h_estimate_norm"] = float(h_rot.norm().item())
                        state["hessian_success_count"] += 1
                    else:
                        state["hessian_skip_count"] += 1

                # ── Sophia cautious step in the rotated basis ────────
                bc1 = 1.0 - beta1 ** t
                m_hat_rot = m_rot / bc1
                _check("hessian_diag_rot", hessian_diag_rot)
                denom = (gamma_scale * hessian_diag_rot.abs()).clamp_(min=eps)
                _check("denom", denom)
                update_rot_raw = m_hat_rot / denom
                _check("update_rot_raw", update_rot_raw)
                update_rot = update_rot_raw.clamp(min=-rho, max=rho)
                _check("update_rot", update_rot)
                state["last_clip_fraction"] = float(
                    (update_rot_raw.abs() > rho).float().mean().item()
                )

                # ── L1: spread cap on rotated update's row norms ─────
                # Bound the per-eigendirection step magnitude spread at
                # spread_cap by damping loud rows toward row_floor.
                # Damp factor = row_floor / max(row_norm, eps), clamped
                # at 1.0 so quiet rows pass through unchanged (we never
                # AMPLIFY a quiet row's update — that would be spurious).
                row_norms = update_rot.norm(dim=-1)
                row_max = row_norms.max()
                safe_max = row_max.clamp(min=eps_adam)
                row_floor = safe_max / spread_cap
                row_norm_safe = row_norms.clamp(min=eps_adam)
                damp = (row_floor / row_norm_safe).clamp(max=1.0)
                _check("damp", damp)
                update_rot = update_rot * damp.unsqueeze(-1)
                _check("update_rot_post_spread", update_rot)

                # ── Rotate update back to parameter basis ────────────
                update = Q_L @ update_rot @ Q_R.T
                _check("update", update)

                # ── Final NaN/Inf guard: refuse to write garbage to p.
                # If anything upstream (eigh, momentum drift, rotated
                # division) produced a non-finite update, skip THIS
                # parameter's update for THIS step. Logged via trap so
                # the cause is still surfaced.
                if not torch.isfinite(update).all():
                    state.setdefault("update_skip_count", 0)
                    state["update_skip_count"] += 1
                    continue

                # ── Apply with weight decay and RAdam r_t scaling ────
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * r_t)
                _check("p_post_update", p)

        return loss

    # ── Telemetry ────────────────────────────────────────────────────
    def get_telemetry(self) -> dict:
        """Aggregate per-organ RACASO counters across all parameters.

        Returns dict with:
          - ``rotation_success_count``, ``rotation_skip_count``
          - ``hessian_success_count``, ``hessian_skip_count``
          - ``rectification_skip_count``
          - ``last_r_t``, ``last_eigh_residual``, ``last_clip_fraction``,
            ``last_h_estimate_norm``
          - ``num_2d_params``
        """
        rot_ok = 0
        rot_skip = 0
        hess_ok = 0
        hess_skip = 0
        rect_skip = 0
        last_r_t = 0.0
        last_eigh_res = 0.0
        last_clip = 0.0
        last_h_norm = 0.0
        last_step = 0
        num_2d = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if not state:
                    continue
                if p.ndim != 2:
                    continue
                num_2d += 1
                rot_ok += state.get("rotation_success_count", 0)
                rot_skip += state.get("rotation_skip_count", 0)
                hess_ok += state.get("hessian_success_count", 0)
                hess_skip += state.get("hessian_skip_count", 0)
                rect_skip += state.get("rectification_skip_count", 0)
                if state.get("step", 0) > last_step:
                    last_step = state["step"]
                    last_r_t = state.get("last_r_t", 0.0)
                    last_eigh_res = state.get("last_eigh_residual", 0.0)
                    last_clip = state.get("last_clip_fraction", 0.0)
                    last_h_norm = state.get("last_h_estimate_norm", 0.0)
        return {
            "rotation_success_count": rot_ok,
            "rotation_skip_count": rot_skip,
            "hessian_success_count": hess_ok,
            "hessian_skip_count": hess_skip,
            "rectification_skip_count": rect_skip,
            "last_r_t": last_r_t,
            "last_eigh_residual": last_eigh_res,
            "last_clip_fraction": last_clip,
            "last_h_estimate_norm": last_h_norm,
            "num_2d_params": num_2d,
            "_diag_skip_reasons": dict(getattr(RACASO, "_diag_skip_reasons", {})),
        }
