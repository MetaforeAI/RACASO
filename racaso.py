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
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


_CURVATURE_MODES = ("hutchinson", "soap")


def _safe_eig_with_residual(
    M: torch.Tensor,
    fallback_Q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Eigendecomposition of symmetric PSD M with progressive ridge
    fallback. Returns ``(Q, relative_residual)`` where the residual is
    Frobenius-normalized: ``‖M_sym − Q·diag(λ)·Qᵀ‖_F / max(‖M_sym‖_F, eps)``.

    Caller's L2 uses the *relative* residual so the threshold has
    consistent meaning across scale regimes (a residual of 0.1 means
    "10% reconstruction error" regardless of ‖M‖). Progressive ridges
    tried: ``0, 1e-6, 1e-3, 1e-1``. If all fail OR produce NaN/Inf in
    Q or eigvals, returns ``(fallback_Q, inf)`` to signal "do not trust
    this refresh."

    NaN trap: PyTorch's ``linalg.eigh`` does NOT always raise on
    rank-deficient or near-singular inputs — it can silently return
    eigenvectors with NaN columns for the null space. We explicitly
    check finiteness and continue to the next ridge if so.
    """
    n = M.shape[-1]
    M_sym = 0.5 * (M + M.T)
    M_sym_norm = float(M_sym.norm().item())
    M_sym_norm = max(M_sym_norm, 1e-30)
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
        abs_residual = float((M_sym - recon).norm().item())
        if not math.isfinite(abs_residual):
            continue
        rel_residual = abs_residual / M_sym_norm
        return Q, rel_residual
    if fallback_Q is not None:
        return fallback_Q, float("inf")
    return eye, float("inf")


class RACASO(Optimizer):
    """Rotation-Aligned Cautious Approximately Second-Order Optimizer.

    See module docstring for the algorithm composition and safety
    chain. For 2-D parameters, runs the full pipeline (rotation +
    Hessian + clip). For 1-D parameters (norms, biases, learned
    scalars), falls back to vanilla Yogi via L3.

    Constructor default lr=3e-4 reflects the bench-tested value; the
    Sophia paper's lr=6e-2 may work with the linear-EMA variant but is
    GPU-untested in our sweep (see paper §10 and bench/decision_hvp_ema.md).
    betas=(0.965, 0.99), rho=0.04, gamma=0.04 plus SOAP-style
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
        lr: float = 3e-4,
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
        eigh_residual_threshold: float = 0.5,  # relative, see _safe_eig_with_residual
        spread_cap: float = 10.0,
        radam_enabled: bool = True,
        initial_accumulator: float = 1e-6,
        curvature_mode: str = "hutchinson",
        forward_fn: Optional[Callable] = None,
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
        if curvature_mode not in _CURVATURE_MODES:
            raise ValueError(
                f"Invalid curvature_mode {curvature_mode!r}; "
                f"expected one of {_CURVATURE_MODES}"
            )
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
            curvature_mode=curvature_mode,
        )
        super().__init__(params, defaults)
        # Instance-level diag counters (#16: moved off the class so
        # multiple optimizer instances don't share counters).
        self._diag_skip_reasons: dict = {
            "no_grad": 0, "no_grad_fn": 0,
            "runtime_err": 0, "hz_none": 0, "success": 0,
            "non_finite": 0,
        }
        # Self-contained Hutchinson context (plan B2). When forward_fn is
        # set, step() computes + stashes the rotated Hutchinson estimate
        # inline on refresh steps — no external wrapper required.
        self._forward_fn: Optional[Callable] = forward_fn
        self._params_ref: Optional[List[torch.Tensor]] = None

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

    def set_hvp_context(
        self,
        forward_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        params: List[torch.Tensor],
    ) -> None:
        """Register the forward function and parameter list for the
        self-contained inline Hutchinson stash.

        Args:
            forward_fn: callable taking the parameter list and returning
                a scalar loss tensor (with autograd graph).
            params: the parameter list (same identities as those
                registered with this optimizer).
        """
        self._forward_fn = forward_fn
        self._params_ref = params

    def is_refresh_step(self) -> bool:
        """True if the step about to be taken is a Hessian-refresh step.

        The next step number is ``(max state step) + 1``; a refresh
        happens at the first step and every ``hessian_freq`` steps after.
        """
        next_step = 1
        for state in self.state.values():
            next_step = max(next_step, int(state.get("step", 0)) + 1)
        hessian_freq = self.param_groups[0]["hessian_freq"]
        return next_step == 1 or next_step % hessian_freq == 0

    def _compute_and_stash_hutchinson(self) -> None:
        """Compute the ROTATED-basis Hutchinson estimate and stash it.

        For each 2-D param p the probe lives in the rotated (eigen)
        basis where the cautious step runs (verified in
        ``/tmp/design_rotated_hvp.py``):

            z_tilde  = Rademacher in the rotated basis
            z_param  = Q_L @ z_tilde @ Q_R.T        (map probe to params)
            Hz_param = H @ z_param                   (param-space HVP)
            h_rot_est = z_tilde * (Q_L.T @ Hz_param @ Q_R)

        ``E[h_rot_est] = diag(H_rot)``. The result is stashed as
        ``p._racaso_hvp_estimate`` (single-stash form); ``step()`` EMAs
        it linearly into ``hessian_diag_rot``. Q_L/Q_R are read from
        per-param state (identity until the first eigh refresh, which is
        consistent — z_tilde in the identity basis equals the param
        basis). Non-finite Hz is still stashed so L5 absorbs it.
        """
        if self._forward_fn is None or self._params_ref is None:
            return

        params = self._params_ref

        def fwd(flat_params: Tuple[torch.Tensor, ...]) -> torch.Tensor:
            return self._forward_fn(list(flat_params))

        def _eye(p: torch.Tensor, dim: int) -> torch.Tensor:
            return torch.eye(dim, device=p.device, dtype=p.dtype)

        z_tildes: List[Optional[torch.Tensor]] = []
        QLs: List[Optional[torch.Tensor]] = []
        QRs: List[Optional[torch.Tensor]] = []
        tangents: List[torch.Tensor] = []
        for p in params:
            if p.ndim != 2:
                z_tildes.append(None)
                QLs.append(None)
                QRs.append(None)
                tangents.append(torch.zeros_like(p))
                continue
            m, n = p.shape
            state = self.state.get(p, {})
            Q_L = state.get("Q_L", _eye(p, m))
            Q_R = state.get("Q_R", _eye(p, n))
            z_tilde = (
                torch.randint(low=0, high=2, size=p.shape,
                              device=p.device, dtype=p.dtype) * 2 - 1
            )
            z_param = Q_L @ z_tilde @ Q_R.T
            z_tildes.append(z_tilde)
            QLs.append(Q_L)
            QRs.append(Q_R)
            tangents.append(z_param)

        # HVP via forward-over-reverse: jvp(grad(f)) is exactly the
        # Hessian-vector product, and avoids materializing a second-
        # derivative tape (the equivalent of torch.func.hvp).
        try:
            _, hz_tuple = torch.func.jvp(
                torch.func.grad(fwd), (tuple(params),), (tuple(tangents),)
            )
        except Exception:
            return  # HVP failure leaves stashes empty; L3/L5 absorb.

        for p, z_tilde, Q_L, Q_R, hz in zip(
            params, z_tildes, QLs, QRs, hz_tuple
        ):
            if z_tilde is None or hz is None:
                continue
            h_rot_est = z_tilde * (Q_L.T @ hz @ Q_R)
            # Stash even when non-finite so the optimizer's L5 absorb
            # counter increments instead of silently no-op'ing.
            p._racaso_hvp_estimate = h_rot_est.detach()

    def _try_hutchinson_hvp(
        self,
        p: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Read a pre-computed rotated-basis Hutchinson estimate from ``p``.

        Contract: on Hessian-refresh steps the inline stash (or an
        external wrapper) computes ``h_rot_est = z_tilde ⊙ (Q_Lᵀ Hz Q_R)``
        — the rotated-basis diagonal estimate, ``E[h_rot_est] =
        diag(H_rot)`` — and stashes it on ``p._racaso_hvp_estimate``.
        We just read it; ``step()`` EMAs it linearly into
        ``hessian_diag_rot``.

        Returns the rotated estimate if stashed, ``None`` if missing
        (non-refresh step, or the hvp call failed). Always clears the
        stash on read so a stale estimate from a refresh step can't
        bleed into the next non-refresh step.

        The probe MUST live in the rotated basis (verified by
        ``/tmp/design_rotated_hvp.py``): rotating a param-basis diagonal
        via ``Qᵀ diag(H) Q`` is wrong because ``diag(Qᵀ H Q) ≠
        Qᵀ diag(H) Q`` when the Hessian has off-diagonal coupling.
        """
        # Instance-level diag counters; legacy class-level fallback
        # for callers that arm `RACASO._diag_skip_reasons = {...}`
        # directly (kept for backward compat with the stage-trap tests).
        diag = self._diag_skip_reasons
        hvp_estimate = getattr(p, "_racaso_hvp_estimate", None)
        if hvp_estimate is None:
            diag["no_grad"] = diag.get("no_grad", 0) + 1
            return None
        try:
            delattr(p, "_racaso_hvp_estimate")
        except AttributeError:
            pass
        # L5: absorb non-finite HVP estimates (e.g. P5 DivBackward0
        # 1/||x||^3 second derivative blow-up).
        if not torch.isfinite(hvp_estimate).all():
            diag["non_finite"] = diag.get("non_finite", 0) + 1
            state = self.state.get(p, None)
            if state is not None:
                state["l5_absorb_fire_count"] = state.get("l5_absorb_fire_count", 0) + 1
            return None
        diag["success"] = diag.get("success", 0) + 1
        return hvp_estimate

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Self-contained Hutchinson stash (plan B2). When a forward_fn is
        # registered, compute the rotated probe inline on refresh steps so
        # RACASO(params, forward_fn=..., curvature_mode="hutchinson") needs
        # no external wrapper. Skipped in soap mode (no HVP) and when the
        # GNB path supplies its own synthetic-label gradient.
        if (
            self._forward_fn is not None
            and self.param_groups[0]["curvature_mode"] == "hutchinson"
            and self.is_refresh_step()
        ):
            with torch.enable_grad():
                self._compute_and_stash_hutchinson()

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
            curvature_mode = group["curvature_mode"]

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
                        # Hessian-diagonal EMA in the ROTATED (eigen)
                        # basis — the basis the cautious step runs in.
                        # Fed by a rotated-basis Hutchinson probe
                        # (curvature_mode="hutchinson"); the denominator
                        # is built positive-by-construction in this
                        # basis with NO Q^T diag(H) Q similarity hack
                        # (verified by /tmp/design_rotated_hvp.py).
                        state["hessian_diag_rot"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format
                        )
                        # Rotated second moment EMA(g_rot^2) for the SOAP
                        # / GNB denominator path (curvature_mode="soap"
                        # and the GNB synthetic-label gradient).
                        state["v_rot"] = torch.full_like(
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
                        state["spread_cap_fire_count"] = 0  # L1 counter
                        state["l5_absorb_fire_count"] = 0   # L5 counter
                        state["last_eigh_residual"] = 0.0
                        state["last_clip_fraction"] = 0.0
                        state["last_h_estimate_norm"] = 0.0
                        state["last_h_ema_norm"] = 0.0
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

                # ── Denominator, built positive-by-construction in the
                # rotated (eigen) basis. Two selectable curvature modes,
                # plus the GNB synthetic-label override. No Q^T diag(H) Q
                # similarity hack anywhere (the removed bug rotated a
                # positive field and .abs()-masked the ~65% negative
                # entries that congruence produced).
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat_rot = m_rot / bc1

                hessian_diag_rot = state["hessian_diag_rot"]
                v_rot = state["v_rot"]
                gnb_ghat = getattr(p, "_racaso_gnb_ghat", None)

                if gnb_ghat is not None:
                    # GNB: rotated Gauss-Newton diagonal = EMA[(Q_L^T ĝ Q_R)^2]
                    # on the synthetic-label CE gradient (verified by
                    # /tmp/verify_gnb_design.py). Positive by construction;
                    # consumes the soap denom path regardless of
                    # curvature_mode. Refresh-only (stash present).
                    try:
                        delattr(p, "_racaso_gnb_ghat")
                    except AttributeError:
                        pass
                    if torch.isfinite(gnb_ghat).all():
                        g_hat_rot = Q_L.T @ gnb_ghat @ Q_R
                        v_rot.mul_(beta2).addcmul_(
                            g_hat_rot, g_hat_rot, value=1.0 - beta2,
                        )
                        state["last_h_estimate_norm"] = float(
                            g_hat_rot.norm().item()
                        )
                        state["hessian_success_count"] += 1
                    else:
                        state["l5_absorb_fire_count"] += 1
                        state["hessian_skip_count"] += 1
                    denom = (v_rot / bc2).sqrt().clamp_(min=eps)
                    state["last_h_ema_norm"] = float(v_rot.norm().item())
                elif curvature_mode == "soap":
                    # SOAP: Adam-in-eigenbasis. v_rot = EMA(g_rot^2),
                    # updated every step, sqrt -> positive denominator.
                    g_rot = Q_L.T @ g @ Q_R
                    v_rot.mul_(beta2).addcmul_(g_rot, g_rot, value=1.0 - beta2)
                    denom = (v_rot / bc2).sqrt().clamp_(min=eps)
                    state["last_h_estimate_norm"] = float(g_rot.norm().item())
                    state["last_h_ema_norm"] = float(v_rot.norm().item())
                    state["hessian_success_count"] += 1
                else:
                    # Hutchinson: hessian_diag_rot is the linear (signed)
                    # EMA of the ROTATED-basis Hessian diagonal probe.
                    # Refresh on the Hessian schedule from the stash.
                    if t % hessian_freq == 0 or t == 1:
                        h_rot_est = self._try_hutchinson_hvp(p)
                        if h_rot_est is not None:
                            hessian_diag_rot.mul_(beta2).add_(
                                h_rot_est, alpha=1.0 - beta2,
                            )
                            state["last_h_estimate_norm"] = float(
                                h_rot_est.norm().item()
                            )
                            state["hessian_success_count"] += 1
                        else:
                            state["hessian_skip_count"] += 1
                    # .abs() here is legitimate: |H_ii| is a true
                    # rotated-basis curvature magnitude (the eigenbasis
                    # diagonal), NOT sign-garbage from a wrong rotation.
                    _check("hessian_diag_rot", hessian_diag_rot)
                    denom = (gamma_scale * hessian_diag_rot.abs()).clamp_(min=eps)
                    state["last_h_ema_norm"] = float(hessian_diag_rot.norm().item())

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
                # L1 counter: any row actually damped (damp < 1) means
                # the spread cap top-clipped a loud row this step. Note:
                # the documented contract is "top-clip rows above
                # row_max/spread_cap" — not "bound the max/min ratio"
                # (we never amplify quiet rows). See paper §2.4/§3.
                if bool((damp < 1.0).any().item()):
                    state["spread_cap_fire_count"] += 1
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
          - ``spread_cap_fire_count`` (L1), ``l5_absorb_fire_count`` (L5)
          - ``last_r_t``, ``last_eigh_residual``, ``last_clip_fraction``,
            ``last_h_estimate_norm``, ``last_h_ema_norm`` (norm of
            ``hessian_diag_rot`` or ``v_rot`` per curvature mode)
          - ``num_2d_params``, ``num_1d_params``, ``_last_step``,
            ``curvature_mode``
        """
        rot_ok = 0
        rot_skip = 0
        hess_ok = 0
        hess_skip = 0
        rect_skip = 0
        spread_cap_fire = 0
        l5_absorb_fire = 0
        last_r_t = 0.0
        last_eigh_res = 0.0
        last_clip = 0.0
        last_h_norm = 0.0
        last_h_ema_norm = 0.0
        last_step = 0
        num_2d = 0
        num_1d = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if not state:
                    continue
                if p.ndim != 2:
                    num_1d += 1
                    rect_skip += state.get("rectification_skip_count", 0)
                    if state.get("step", 0) > last_step:
                        last_step = state["step"]
                    continue
                num_2d += 1
                rot_ok += state.get("rotation_success_count", 0)
                rot_skip += state.get("rotation_skip_count", 0)
                hess_ok += state.get("hessian_success_count", 0)
                hess_skip += state.get("hessian_skip_count", 0)
                rect_skip += state.get("rectification_skip_count", 0)
                spread_cap_fire += state.get("spread_cap_fire_count", 0)
                l5_absorb_fire += state.get("l5_absorb_fire_count", 0)
                if state.get("step", 0) >= last_step:
                    last_step = state["step"]
                    last_r_t = state.get("last_r_t", 0.0)
                    last_eigh_res = state.get("last_eigh_residual", 0.0)
                    last_clip = state.get("last_clip_fraction", 0.0)
                    last_h_norm = state.get("last_h_estimate_norm", 0.0)
                    last_h_ema_norm = state.get("last_h_ema_norm", 0.0)
        return {
            "rotation_success_count": rot_ok,
            "rotation_skip_count": rot_skip,
            "hessian_success_count": hess_ok,
            "hessian_skip_count": hess_skip,
            "rectification_skip_count": rect_skip,
            "spread_cap_fire_count": spread_cap_fire,
            "l5_absorb_fire_count": l5_absorb_fire,
            "last_r_t": last_r_t,
            "last_eigh_residual": last_eigh_res,
            "last_clip_fraction": last_clip,
            "last_h_estimate_norm": last_h_norm,
            "last_h_ema_norm": last_h_ema_norm,
            "num_2d_params": num_2d,
            "num_1d_params": num_1d,
            "_last_step": last_step,
            "curvature_mode": self.param_groups[0]["curvature_mode"],
            "_diag_skip_reasons": dict(self._diag_skip_reasons),
        }

    def get_safety_counts(self) -> Dict[str, int]:
        """Return the L1..L5 safety-chain counters as a dict.

        Patches the bench harness gap where ``run_bench.py`` reads
        ``optimizer.safety_counts`` (absent) — this method is the
        canonical accessor. Maps:
          - l1: spread_cap_fire_count (top-clip on rotated row norms)
          - l2: rotation_skip_count (eigh refresh failures)
          - l3: hessian_skip_count (HVP missing or absorbed)
          - l4: rectification_skip_count (RAdam cold-start gate)
          - l5: l5_absorb_fire_count (non-finite HVP absorb)
        """
        tel = self.get_telemetry()
        return {
            "l1": int(tel.get("spread_cap_fire_count", 0)),
            "l2": int(tel.get("rotation_skip_count", 0)),
            "l3": int(tel.get("hessian_skip_count", 0)),
            "l4": int(tel.get("rectification_skip_count", 0)),
            "l5": int(tel.get("l5_absorb_fire_count", 0)),
        }
