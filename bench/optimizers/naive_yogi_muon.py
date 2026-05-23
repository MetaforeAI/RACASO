"""Naive Yogi → Muon composition — the anti-baseline.

This is the optimizer the Muogi paper explicitly argues *fails*.

Construction (the "naive" composition):
    1. Compute Yogi moments:
           m_t = β₁·m + (1-β₁)·g
           v_t = v - (1-β₂)·sign(v - g²)·g²
    2. Bias-correct:
           m_hat = m_t / (1 - β₁^t)
           v_hat = v_t / (1 - β₂^t)
    3. Form Yogi's full preconditioned direction:
           D = m_hat / (sqrt(v_hat) + eps_yogi)
    4. For 2-D params: feed D into NS5 (5-iteration Newton-Schulz with
       Jordan coefficients). No row-scale injection. No spread cap. No
       safety chain. **This is the composition Muogi's paper says
       destroys Yogi's variance signal** — NS5 sees a near-orthogonal
       input and averages it to the spectral mean, erasing the
       burst-aware variance that Yogi's update encoded.
    5. For 1-D params: skip NS5, use D directly.
    6. Step: W ← W - lr · output

Contrast with Muogi's "cheater's choice":
    Muogi calls NS5 on ``R · m_hat`` (row-scaled momentum where R encodes
    the per-row Yogi variance). NS5 orthogonalizes the row-scaled input,
    so the output is ``polar(R · m_hat)`` — the variance signal enters
    via row-scale and gets preserved in the rotation. The naive
    composition here orthogonalizes the already-divided-by-sqrt(v_hat)
    direction, losing the variance signal in the spectral mean.

This optimizer must be a real ``torch.optim.Optimizer`` with state_dict
support so the harness treats it identically to Muogi. NS5 is inlined
(not imported from Muogi's internal helper) to keep the naive
composition truly free of Muogi machinery.

Used as the anti-baseline for Muogi-paper claim M1 (variance preservation).
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


# Newton-Schulz iteration coefficients from Keller Jordan's reference
# implementation (the 5-iteration polynomial used in Muon and Muogi).
# Per-iteration update:   X ← a·X + b·A·X + c·A²·X     where A = X X^T
# Standard Jordan values: a = 3.4445,  b = -4.7750,  c = 2.0315
# (b is negative — the polynomial is monotonic toward orthogonality only
# with the correct sign. Earlier docstring used "a*X - b*XX^TX" with b
# stored positive; that mismatched the implementation. Both the constant
# and the iteration body now use the consistent sign convention.)
_NS5_A = 3.4445
_NS5_B = -4.7750
_NS5_C = 2.0315


@torch.no_grad()
def _naive_ns5(X: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Pure 5-iteration Newton-Schulz orthogonalization.

    No row-scale injection. No spread cap. No safety chain. This is
    intentionally the bare polynomial: the naive composition fed into
    NS5 is what the Muogi paper says fails.

    Operates on 2-D input X (treats wide matrices as the standard case
    and transposes M > N matrices so the longer dim is the K-axis,
    matching Muon's convention).
    """
    if X.dim() != 2:
        raise ValueError(f"NS5 requires 2-D input, got shape {tuple(X.shape)}")
    # Normalize spectral norm to ≤ 1 by dividing by Frobenius norm
    # (cheap overestimate of the spectral norm; standard Muon practice).
    norm = X.norm().clamp_min(1e-12)
    X = X / norm
    # Transpose if M > N so the inner product XX^T is the smaller square.
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    for _ in range(iters):
        A = X @ X.T
        B = _NS5_B * A + _NS5_C * (A @ A)
        X = _NS5_A * X + B @ X
    if transposed:
        X = X.T
    return X


class NaiveYogiMuon(Optimizer):
    """Naive Yogi → Muon composition (anti-baseline for Muogi paper M1).

    Args:
        params: iterable of parameters to optimize.
        lr: learning rate.
        betas: (β₁, β₂) for Yogi moments.
        eps_yogi: ε in the ``m_hat / (sqrt(v_hat) + ε)`` denominator.
        ns5_iters: number of NS5 iterations on 2-D params.
        initial_accumulator: starting value for Yogi state (matches
            Yogi's convention to avoid divide-by-zero in early steps).
        weight_decay: decoupled weight decay (set to 0.0 by default to
            keep the naive composition's behavior pure).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps_yogi: float = 1e-3,
        ns5_iters: int = 5,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps_yogi <= 0.0:
            raise ValueError(f"Invalid eps_yogi: {eps_yogi}")
        if ns5_iters < 1:
            raise ValueError(f"Invalid ns5_iters: {ns5_iters}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps_yogi=eps_yogi,
            ns5_iters=ns5_iters,
            initial_accumulator=initial_accumulator,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps_yogi = group["eps_yogi"]
            ns5_iters = group["ns5_iters"]
            init_acc = group["initial_accumulator"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError(
                        "NaiveYogiMuon does not support sparse gradients"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Yogi moment updates.
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                grad_sq = g * g
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_sq),
                    grad_sq,
                    value=-(1.0 - beta2),
                )

                # Bias correction.
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Form Yogi's full preconditioned direction.
                D = m_hat / (v_hat.sqrt() + eps_yogi)

                # NAIVE COMPOSITION: feed Yogi's already-normalized
                # direction into NS5. NS5 orthogonalizes a near-orthogonal
                # input → averages to spectral mean → Yogi's variance
                # signal is gone.
                if p.dim() == 2 and p.shape[0] >= 2 and p.shape[1] >= 2:
                    update = _naive_ns5(D, iters=ns5_iters)
                else:
                    update = D

                p.add_(update, alpha=-lr)

        return loss
