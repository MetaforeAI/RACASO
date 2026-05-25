"""Empirical decision experiment: squared vs linear HVP-EMA in RACASO.

Question: §2.2.1 claims RACASO "preserves both positive and negative
curvature." The current code does
    EMA <- beta2 * EMA + (1 - beta2) * h * h
which tracks E[h²] (a magnitude/variance estimator). The sign of h is
destroyed before the denominator. Linear-EMA tracks E[h] (sign-preserving).

This script:
  1. Constructs a matrix saddle problem
       f(W) = 0.5 * sum_ij D_ij * W_ij^2,    D = diag-style ±1 reshaped
  2. Runs RACASO with (A) squared EMA and (B) linear EMA over a LR sweep
     and 3 seeds. Both variants have the SOAP-style rotation fix applied
     (no Q.T h Q similarity transform; HVP used directly).
  3. Compares saddle escape depth ||W||² and final loss.

Decision criterion: which variant uses negative-curvature info more
strongly — i.e., escapes the saddle faster / deeper? If linear-EMA
diverges or fails to improve, ship squared. Otherwise ship linear.

Run as:
    python -m bench.decision_hvp_ema_run
"""
from __future__ import annotations

import math
import statistics
from typing import List, Tuple

import torch

import racaso as racaso_mod
from racaso import RACASO


# ---------- linear-EMA variant (B) ----------
# We patch RACASO.step() at module level via a flag instead of duplicating
# the file. The flag is read inside step() — see the patched method below.

def _patched_step_factory(use_linear_ema: bool):
    """Return a bound step method using either squared or linear HVP-EMA.

    Both variants apply the SOAP-style rotation fix: the HVP estimate
    is *not* rotated by Q_L^T (...) Q_R. The denominator uses an EMA in
    the parameter basis (rotated denom is built by reading the EMA
    after EMA update).
    """
    # We re-build the step() to keep this experiment self-contained.
    # Import locals from the module so the closure stays clean.
    import math
    import torch

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
                state = self.state[p]
                use_rotation = (g.ndim == 2)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format)
                    if use_rotation:
                        m, n = p.shape
                        # Param-basis Hessian-diag EMA (no rotation)
                        state["hessian_diag_param"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format)
                        state["GG_L"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
                        state["GG_R"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                        state["Q_L"] = torch.eye(m, device=p.device, dtype=p.dtype)
                        state["Q_R"] = torch.eye(n, device=p.device, dtype=p.dtype)
                        state["rotation_success_count"] = 0
                        state["rotation_skip_count"] = 0
                        state["hessian_success_count"] = 0
                        state["hessian_skip_count"] = 0
                    else:
                        state["exp_avg_sq"] = torch.full_like(
                            p, init_acc, memory_format=torch.preserve_format)
                    state["rectification_skip_count"] = 0
                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                if not use_rotation:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    grad_sq = g * g
                    exp_avg_sq.addcmul_(
                        torch.sign(exp_avg_sq - grad_sq), grad_sq,
                        value=-(1.0 - beta2))
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    denom = (exp_avg_sq / bc2).sqrt().clamp_(min=eps_yogi).add_(eps_adam)
                    p.addcdiv_(exp_avg / bc1, denom, value=-lr)
                    continue

                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)

                if radam_enabled:
                    warmed_up, r_t = self._radam_rectification(t, beta2)
                    if not warmed_up:
                        state["rectification_skip_count"] += 1
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        bc1 = 1.0 - beta1 ** t
                        p.add_(exp_avg, alpha=-lr / bc1)
                        continue
                else:
                    r_t = 1.0

                GG_L = state["GG_L"]
                GG_R = state["GG_R"]
                GG_L.mul_(shampoo_beta).addmm_(g, g.T, alpha=1.0 - shampoo_beta)
                GG_R.mul_(shampoo_beta).addmm_(g.T, g, alpha=1.0 - shampoo_beta)
                Q_L = state["Q_L"]
                Q_R = state["Q_R"]
                if t % refresh_freq == 0 or t == 1:
                    from racaso import _safe_eig_with_residual
                    Q_L_new, res_L = _safe_eig_with_residual(GG_L, fallback_Q=Q_L)
                    Q_R_new, res_R = _safe_eig_with_residual(GG_R, fallback_Q=Q_R)
                    L_ok = math.isfinite(res_L) and res_L < eigh_res_threshold
                    R_ok = math.isfinite(res_R) and res_R < eigh_res_threshold
                    if L_ok:
                        state["Q_L"] = Q_L_new; Q_L = Q_L_new
                    if R_ok:
                        state["Q_R"] = Q_R_new; Q_R = Q_R_new
                    if L_ok and R_ok:
                        state["rotation_success_count"] += 1
                    else:
                        state["rotation_skip_count"] += 1

                # SOAP-style fix: rotate gradient/momentum; HVP stays
                # in the parameter basis.
                m_rot = Q_L.T @ exp_avg @ Q_R

                hessian_diag_param = state["hessian_diag_param"]
                if t % hessian_freq == 0 or t == 1:
                    h_diag_param = self._try_hutchinson_hvp(p)
                    if h_diag_param is not None:
                        if use_linear_ema:
                            # Linear EMA — preserves sign of h.
                            hessian_diag_param.mul_(beta2).add_(
                                h_diag_param, alpha=1.0 - beta2)
                        else:
                            # Squared EMA — tracks E[h^2].
                            hessian_diag_param.mul_(beta2).addcmul_(
                                h_diag_param, h_diag_param, value=1.0 - beta2)
                        state["hessian_success_count"] += 1
                    else:
                        state["hessian_skip_count"] += 1

                # Denom = |EMA| in param basis, then rotated for the
                # rotated-update division. SOAP-style: rotate the
                # element-wise denom into the same basis as m_rot.
                denom_param = (gamma_scale * hessian_diag_param.abs()).clamp_(min=eps)
                # Rotate denom into m_rot basis (same Kronecker rotation).
                denom_rot = (Q_L.T @ denom_param @ Q_R).abs().clamp_(min=eps)

                bc1 = 1.0 - beta1 ** t
                m_hat_rot = m_rot / bc1
                update_rot = (m_hat_rot / denom_rot).clamp(min=-rho, max=rho)

                # L1 spread cap
                row_norms = update_rot.norm(dim=-1)
                row_max = row_norms.max()
                safe_max = row_max.clamp(min=eps_adam)
                row_floor = safe_max / spread_cap
                row_norm_safe = row_norms.clamp(min=eps_adam)
                damp = (row_floor / row_norm_safe).clamp(max=1.0)
                update_rot = update_rot * damp.unsqueeze(-1)

                update = Q_L @ update_rot @ Q_R.T
                if not torch.isfinite(update).all():
                    state.setdefault("update_skip_count", 0)
                    state["update_skip_count"] += 1
                    continue
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * r_t)
        return loss
    return step


def _saddle_loss(W: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return 0.5 * (D * W * W).sum()


def _make_saddle_problem(seed: int, shape=(5, 4)) -> Tuple[torch.Tensor, torch.Tensor]:
    """Matrix saddle with mixed-magnitude eigenvalues. D has a mix of
    positive and negative entries with varying magnitudes, so the
    distinction between |h| (linear-EMA-absolute) and h² (squared-EMA)
    is non-trivial: |D_ij| vs D_ij² produces *different* denominators
    when the magnitudes vary.

    Layout: half positive, half negative, magnitudes drawn from a
    deterministic geometric series."""
    n = shape[0] * shape[1]
    half = n // 2
    mags = torch.tensor([2.0 ** k for k in range(n)], dtype=torch.float64) / (2.0 ** ((n - 1) / 2))
    signs = torch.cat([torch.ones(half), -torch.ones(n - half)])
    diag = signs * mags
    # Shuffle so saddle directions are interspersed with positive ones,
    # increasing rotation pressure.
    perm_gen = torch.Generator(); perm_gen.manual_seed(7)
    perm = torch.randperm(n, generator=perm_gen)
    diag = diag[perm]
    D = diag.reshape(shape)
    gen = torch.Generator(); gen.manual_seed(seed)
    W = (0.1 * torch.randn(shape, generator=gen, dtype=torch.float64)).clone()
    W.requires_grad_(True)
    return W, D


def _hutchinson_estimate(W: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """For f(W) = 0.5 * sum D_ij W_ij^2, the Hessian is element-wise D.
    With Rademacher z, z*Hz = z*(D*z) = D*z*z = D since z^2 = 1.
    Return that exactly (no torch.func needed; this isolates the EMA
    decision from autograd noise)."""
    return D.clone()  # deterministic exact Hessian diagonal


def run_one(use_linear: bool, lr: float, seed: int, steps: int = 500) -> Tuple[float, float]:
    """Returns (final_loss, final_param_norm_sq)."""
    W, D = _make_saddle_problem(seed)
    # rho loose so denom (not the clip) drives behavior.
    opt = RACASO([W], lr=lr, betas=(0.9, 0.99), refresh_freq=10, hessian_freq=10,
                 rho=10.0)
    # Patch the step method on this instance.
    import types
    opt.step = types.MethodType(_patched_step_factory(use_linear), opt)
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        # Analytic grad: D * W
        with torch.no_grad():
            g = D * W.detach()
            W.grad = g
            # Inject HVP estimate (exact, since H is diagonal D)
            if (t + 1) % 10 == 0 or t == 0:
                W._racaso_hvp_estimate = _hutchinson_estimate(W, D)
        opt.step()
        if not torch.isfinite(W).all():
            return float("nan"), float("nan")
    with torch.no_grad():
        final_loss = float(_saddle_loss(W, D).item())
        final_norm_sq = float((W * W).sum().item())
    return final_loss, final_norm_sq


def main() -> None:
    LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    SEEDS = (0, 1, 2)
    STEPS = 500
    print("=" * 70)
    print("HVP-EMA decision experiment: matrix saddle (5x4), 10+/10- eigenvalues")
    print(f"steps={STEPS}, seeds={SEEDS}")
    print("=" * 70)
    results = {"squared": {}, "linear": {}}
    for variant_name, use_linear in (("squared", False), ("linear", True)):
        for lr in LRS:
            losses, norms = [], []
            for s in SEEDS:
                loss, nrm = run_one(use_linear, lr, s, STEPS)
                if math.isfinite(loss):
                    losses.append(loss); norms.append(nrm)
            if losses:
                results[variant_name][lr] = (
                    statistics.mean(losses), statistics.mean(norms),
                    min(losses), max(norms))
            else:
                results[variant_name][lr] = (float("nan"),) * 4

    # Print table
    print(f"\n{'lr':>10} | {'squared loss':>14} {'squared ||W||²':>16} | "
          f"{'linear loss':>14} {'linear ||W||²':>16}")
    print("-" * 80)
    for lr in LRS:
        s_loss, s_nrm, _, _ = results["squared"][lr]
        l_loss, l_nrm, _, _ = results["linear"][lr]
        print(f"{lr:>10.2e} | {s_loss:>14.3e} {s_nrm:>16.3e} | "
              f"{l_loss:>14.3e} {l_nrm:>16.3e}")
    print()
    # Best per variant
    best_sq = min(results["squared"].values(), key=lambda v: v[0] if math.isfinite(v[0]) else float("inf"))
    best_li = min(results["linear"].values(), key=lambda v: v[0] if math.isfinite(v[0]) else float("inf"))
    print(f"Best squared: loss={best_sq[0]:.3e}, ||W||²={best_sq[1]:.3e}")
    print(f"Best linear : loss={best_li[0]:.3e}, ||W||²={best_li[1]:.3e}")

    # Decision
    print("\nDecision criterion: lower loss => deeper saddle escape.")
    if math.isfinite(best_li[0]) and best_li[0] < best_sq[0]:
        print("=> linear-EMA wins (use signed h in EMA).")
    elif math.isfinite(best_sq[0]) and best_sq[0] < best_li[0]:
        print("=> squared-EMA wins (variant A).")
    else:
        print("=> inconclusive.")


if __name__ == "__main__":
    main()
