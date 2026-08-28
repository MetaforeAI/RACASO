"""RACASO HVP-strategy wrappers: Hutchinson and Gauss-Newton-Bartlett.

The RACASO optimizer builds its rotated-basis denominator in one of two
selectable curvature modes, plus a GNB synthetic-label override:

  - **Hutchinson HVP** (``curvature_mode="hutchinson"``) — a Rademacher
    probe in the ROTATED (eigen) basis gives an unbiased estimate of the
    rotated-basis Hessian diagonal ``diag(H_rot)``
    (see ``/tmp/design_rotated_hvp.py``). Captures true negative
    curvature. Works against any twice-differentiable forward function.

  - **Gauss-Newton-Bartlett (GNB)** — for a softmax + cross-entropy
    classification head, sample one synthetic label per row from the
    model's own softmax and compute the CE gradient ``ĝ`` with
    ``reduction="mean"``. The rotated GN diagonal is
    ``EMA[(Q_L^T ĝ Q_R)^2]`` — SOAP's second moment on the
    synthetic-label gradient instead of the real gradient
    (see ``/tmp/verify_gnb_design.py``). Positive by construction,
    rotates correctly, no ``.abs()`` hack. Drops the negative-definite
    component of the Hessian. Cost: one extra first-order backward.

``RACASOHutchinson`` is a thin subclass — the base optimizer owns the
inline rotated-probe stash via ``set_hvp_context(forward_fn, params)``.
``RACASOGNB`` keeps its own logits path: it stashes the synthetic-label
gradient on ``p._racaso_gnb_ghat`` and the base ``step()`` routes it
through the rotated second-moment (soap) denominator.

Both wrappers are drop-in optimizers for the bench harness — no
harness-side changes needed.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from racaso import RACASO


# ── Hutchinson wrapper ───────────────────────────────────────────────────


class RACASOHutchinson(RACASO):
    """RACASO with the rotated-basis Hutchinson stash.

    Thin subclass: ``set_hvp_context``, ``is_refresh_step`` and the
    inline rotated-probe stash live on the base class. Registering a
    ``forward_fn`` (constructor kwarg or ``set_hvp_context``) makes the
    optimizer self-contained — it computes ``h_rot_est`` on each refresh
    step and EMAs it into ``hessian_diag_rot``.
    """


# ── GNB wrapper ──────────────────────────────────────────────────────────


class RACASOGNB(RACASO):
    """RACASO with the Gauss-Newton-Bartlett synthetic-label denominator.

    For a classification head, samples one synthetic label per row from
    the model's softmax, recomputes the mean-reduction CE-loss gradient
    ``ĝ``, and stashes it (param basis) on ``p._racaso_gnb_ghat``. The
    base ``step()`` forms the rotated GN diagonal
    ``v_rot = EMA[(Q_L^T ĝ Q_R)^2]`` and uses the soap denominator
    ``sqrt(v_rot / (1 - β2^t))`` — unifying GNB with SOAP on one
    rotated-second-moment code path that differs only in which gradient
    feeds ``v_rot`` (real ``g`` for SOAP, synthetic ``ĝ`` for GNB).
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
            logits_fn: callable taking the parameter list and returning a
                ``[B, C]`` logits tensor (with autograd graph).
            params: the parameter list (same identities as those
                registered with this optimizer).
        """
        self._logits_fn = logits_fn
        self._params_ref = params

    def _compute_and_stash_gnb(self) -> None:
        """Stash the synthetic-label CE gradient ``ĝ`` on each 2-D param.

        ``ĝ`` uses ``reduction="mean"`` (the GNB design — NO ``*
        batch_size``); the base ``step()`` squares its rotated form into
        ``v_rot``.
        """
        if self._logits_fn is None or self._params_ref is None:
            return

        params = self._params_ref
        logits = self._logits_fn(params)  # [B, C]
        if logits.dim() != 2:
            return  # contract violation; bail

        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)
            synthetic_y = torch.multinomial(probs, num_samples=1).squeeze(-1)

        ce_loss = torch.nn.functional.cross_entropy(
            logits, synthetic_y, reduction="mean"
        )
        grads = torch.autograd.grad(
            ce_loss,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        for p, g_hat in zip(params, grads):
            if g_hat is None or p.ndim < 2:
                continue
            p._racaso_gnb_ghat = g_hat.detach()

    @torch.no_grad()
    def step(self, closure=None):
        if self.is_refresh_step():
            with torch.enable_grad():
                self._compute_and_stash_gnb()
        return super().step(closure)
