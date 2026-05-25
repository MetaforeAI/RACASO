"""RACASO HVP-strategy wrappers: Hutchinson and Gauss-Newton-Bartlett.

The RACASO optimizer reads its Hessian-diagonal estimate from
``p._racaso_hvp_estimate``, which must be stashed on each parameter
before each ``step()`` call. The paper (§2.2) documents two strategies
for producing that stash:

  - **Hutchinson HVP** — `z * Hz` from a single Rademacher probe,
    computed via ``torch.func.hvp``. Captures the true Hessian diagonal,
    including negative-curvature directions. Works against any twice-
    differentiable forward function.

  - **Gauss-Newton-Bartlett (GNB)** — for a classification problem with
    softmax output and cross-entropy loss, sample one synthetic label
    per row from the model's own softmax, compute ``ĝ = ∂CE(logits, ŷ)/∂p``,
    stash ``ĝ² * B`` (Sophia §3.2 reweighting). Drops the negative-
    definite component of the Hessian, always positive semidefinite.
    Cost: one extra first-order backward. No second derivative.

Both wrappers subclass ``RACASO`` and override ``step()`` to compute and
stash the HVP estimate before delegating to ``super().step()``. The
bench harness can treat them as drop-in optimizers — no harness-side
changes needed.

For the Hutchinson wrapper, the upstream caller must provide the
problem's ``forward(params) -> scalar loss`` function. For the GNB
wrapper, the caller must provide a ``logits_fn(params) -> [B, C]
logits`` function. The bench harness wires these up automatically from
the active ``BenchProblem`` instance.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from bench.optimizers.racaso import RACASO


# ── Hutchinson wrapper ───────────────────────────────────────────────────


class RACASOHutchinson(RACASO):
    """RACASO with Hutchinson HVP stash computed inline before each step.

    Reads the ``forward_fn`` and ``params_ref`` attached by the bench
    harness (``opt.set_hvp_context(forward_fn, params)``) and computes
    ``z * Hz`` via ``torch.func.hvp`` for each 2-D parameter on every
    refresh step (controlled by RACASO's ``hessian_freq``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._forward_fn: Optional[Callable] = None
        self._params_ref: Optional[List[torch.Tensor]] = None

    def set_hvp_context(
        self,
        forward_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        params: List[torch.Tensor],
    ) -> None:
        """Register the forward function for HVP computation.

        Args:
            forward_fn: callable that takes the parameter list and
                returns a scalar loss tensor (with autograd graph).
            params: the parameter list (same identities as the ones
                registered with this optimizer).
        """
        self._forward_fn = forward_fn
        self._params_ref = params

    def _compute_and_stash_hutchinson(self) -> None:
        """Compute ``z ⊙ Hz`` for each 2-D param and stash on ``p._racaso_hvp_estimate``.

        Uses a single Rademacher probe ``z ∈ {-1, +1}^shape(p)`` per
        param. ``torch.func.hvp`` evaluates the HVP without building a
        second-derivative tape, which sidesteps the eager-autograd
        pathologies documented in §6 of the paper.
        """
        if self._forward_fn is None or self._params_ref is None:
            return  # context not set — RACASO falls back to L4/L5 paths

        params = self._params_ref
        # Build a closure of forward over the params list.
        def fwd(*flat_params: torch.Tensor) -> torch.Tensor:
            return self._forward_fn(list(flat_params))

        # Rademacher probes per param (same dtype as the param so the
        # HVP arithmetic stays in the param's numerical regime).
        probes = [
            (torch.randint(low=0, high=2, size=p.shape,
                           device=p.device, dtype=p.dtype) * 2 - 1)
            if p.ndim >= 2 else None
            for p in params
        ]
        # Only need HVP at 2-D params; pass zero probes for the rest so
        # torch.func.hvp gets a tuple of the right shape.
        probe_tuple = tuple(
            z if z is not None else torch.zeros_like(p)
            for z, p in zip(probes, params)
        )
        # Pass the LIVE params directly (no .clone()). torch.func.hvp is
        # functional — it does not in-place modify its primal argument,
        # so handing it the live tensors is safe and avoids the
        # duplicate forward pass the previous .clone() route caused
        # (see paper §4.1 — the "monolithic engine" claim depends on
        # *not* running forward twice per refresh step).
        param_tuple = tuple(params)

        try:
            _, hz_tuple = torch.func.hvp(fwd, param_tuple, probe_tuple)
        except Exception:
            # Any HVP failure leaves stashes empty; RACASO's L5 absorbs.
            return

        for p, z, hz in zip(params, probes, hz_tuple):
            if z is None or hz is None:
                continue
            if not torch.isfinite(hz).all():
                # Stash NaN/Inf so the optimizer's L5 absorb counter
                # increments instead of silently being a no-op.
                p._racaso_hvp_estimate = (z * hz).detach()
                continue
            p._racaso_hvp_estimate = (z * hz).detach()

    @torch.no_grad()
    def step(self, closure=None):
        # Only compute the HVP on refresh steps to amortize cost.
        # RACASO's internal hessian_freq controls when it actually
        # consumes the stash; we mirror that here so we don't waste
        # compute on non-refresh steps.
        any_param_state = next(iter(self.state.values()), None)
        next_step = (any_param_state.get("step", 0) + 1) if any_param_state else 1
        hessian_freq = self.param_groups[0].get("hessian_freq", 10)
        if next_step == 1 or next_step % hessian_freq == 0:
            with torch.enable_grad():
                self._compute_and_stash_hutchinson()
        return super().step(closure)


# ── GNB wrapper ──────────────────────────────────────────────────────────


class RACASOGNB(RACASO):
    """RACASO with Gauss-Newton-Bartlett HVP stash.

    For a classification problem, samples one synthetic label per row
    from the model's softmax, recomputes the CE-loss gradient, squares
    it elementwise, scales by the batch size, and stashes the result on
    ``p._racaso_hvp_estimate``.

    Mathematical justification (paper §2.2.2):

        H ≈ G_N = E_{ŷ ~ p(y|x)} [ ∇log p(ŷ|x) · ∇log p(ŷ|x)^T ]

    Operationally, ``diag(G_N) ≈ ĝ² · B``, where ``ĝ`` is the
    per-parameter gradient of the synthetic-label CE loss and ``B`` is
    the batch size (Sophia §3.2 reweighting). GNB drops the negative-
    definite component of the Hessian, so the Sophia clip in RACASO
    never sees a sign-flip; the optimizer's behavior is strictly more
    conservative than Hutchinson.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._logits_fn: Optional[Callable] = None
        self._params_ref: Optional[List[torch.Tensor]] = None

    def set_hvp_context(
        self,
        logits_fn: Callable[[List[torch.Tensor]], torch.Tensor],
        params: List[torch.Tensor],
    ) -> None:
        """Register the logits function for GNB computation.

        Args:
            logits_fn: callable that takes the parameter list and
                returns a ``[B, C]`` logits tensor (with autograd graph).
            params: the parameter list.
        """
        self._logits_fn = logits_fn
        self._params_ref = params

    def _compute_and_stash_gnb(self) -> None:
        """Sample synthetic labels, compute ĝ² · B, stash on each 2-D param."""
        if self._logits_fn is None or self._params_ref is None:
            return

        params = self._params_ref
        # Need a fresh forward pass with autograd graph.
        # The bench harness owns the data; we just call logits_fn(params).
        logits = self._logits_fn(params)  # [B, C]
        if logits.dim() != 2:
            return  # contract violation; bail
        batch_size = logits.shape[0]

        # Sample synthetic labels from the model's own softmax.
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)
            synthetic_y = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # CE-loss against the synthetic labels.
        ce_loss = torch.nn.functional.cross_entropy(
            logits, synthetic_y, reduction="sum"
        )
        # Compute gradients w.r.t. each parameter.
        grads = torch.autograd.grad(
            ce_loss,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for p, g in zip(params, grads):
            if g is None or p.ndim < 2:
                continue
            if not torch.isfinite(g).all():
                continue
            # diag(G_N) ≈ ĝ² · B.
            p._racaso_hvp_estimate = (g.detach() ** 2) * float(batch_size)

    @torch.no_grad()
    def step(self, closure=None):
        any_param_state = next(iter(self.state.values()), None)
        next_step = (any_param_state.get("step", 0) + 1) if any_param_state else 1
        hessian_freq = self.param_groups[0].get("hessian_freq", 10)
        if next_step == 1 or next_step % hessian_freq == 0:
            with torch.enable_grad():
                self._compute_and_stash_gnb()
        return super().step(closure)
