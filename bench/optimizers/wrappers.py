"""Canonical optimizer-construction entry point for the RACASO benchmark suite.

``build_optimizer(name, params, lr, problem=None)`` returns a fully
configured ``torch.optim.Optimizer``. All hyperparameters other than the
learning rate are pinned here. The optional ``problem`` argument is used
by the two RACASO HVP-strategy wrappers (``racaso_hutchinson`` and
``racaso_gnb``) to wire up the forward / logits function the wrapper
needs to compute its Hessian-diagonal stash.

Every optimizer is vendored as a standalone source file in this
``bench/optimizers/`` directory — no sibling-repo imports, no sys.path
gymnastics. Sibling-project optimizers (Liger, Muogi, RAMuogi) are
treated exactly like external baselines (Lion, Yogi): the source file
is copied here.

Canonical configs (single source of truth — see ``README.md``):

    adam              : torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8)
    adamw             : torch.optim.AdamW(lr, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0.01)
    yogi              : Yogi(lr, betas=(0.9, 0.999), eps=1e-3,
                             initial_accumulator=1e-6, weight_decay=0.0)
                        — bench/optimizers/yogi.py (Zaheer et al. 2018)
    lion              : Lion(lr, betas=(0.9, 0.99), weight_decay=0.0)
                        — bench/optimizers/lion.py (Chen et al. 2023)
    naive_yogi_muon   : NaiveYogiMuon(lr, betas=(0.9, 0.999), eps_yogi=1e-3,
                                      ns5_iters=5)
                        — bench/optimizers/naive_yogi_muon.py
    racaso_hutchinson : RACASOHutchinson(lr) — Hutchinson HVP stash
                        — bench/optimizers/racaso_hvp_wrappers.py
                        — requires problem.forward
    racaso_gnb        : RACASOGNB(lr) — Gauss-Newton-Bartlett HVP stash
                        — bench/optimizers/racaso_hvp_wrappers.py
                        — requires problem.logits_fn (classification problem)
    liger             : Liger(lr, betas=(0.9, 0.99), eps_yogi=1e-3, wd=0.0)
                        — bench/optimizers/liger.py (sibling repo, vendored)
    muogi             : Muogi(lr, default Muogi config)
                        — bench/optimizers/muogi.py (sibling repo, vendored)
    ramuogi           : RAMuogi(lr, default RAMuogi config)
                        — bench/optimizers/ramuogi.py (sibling repo, vendored)
    muon              : NotImplementedError until vendored from the
                        Keller Jordan reference implementation
    sophia            : NotImplementedError until vendored from the
                        official Sophia repo
    soap              : NotImplementedError until vendored from Vyas et al.
"""

from __future__ import annotations

from typing import List, Optional

import torch


KNOWN_OPTIMIZERS = (
    "adam",
    "adamw",
    "yogi",
    "lion",
    "naive_yogi_muon",
    "racaso_hutchinson",
    "racaso_gnb",
    "liger",
    "muogi",
    "ramuogi",
    "muon",
    "sophia",
    "soap",
)


def _vendor_pointer(opt_name: str) -> str:
    return (
        f"{opt_name} is not vendored yet; "
        "see bench/optimizers/README.md for the canonical source and "
        "pinned version to drop into this wrapper before benchmarks run."
    )


# ── Constructors ─────────────────────────────────────────────────────────


def _build_adam(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_adamw(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )


def _build_yogi(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    from bench.optimizers.yogi import Yogi

    return Yogi(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0.0,
    )


def _build_lion(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    from bench.optimizers.lion import Lion

    return Lion(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)


def _build_naive_yogi_muon(
    params: List[torch.Tensor], lr: float, **_
) -> torch.optim.Optimizer:
    from bench.optimizers.naive_yogi_muon import NaiveYogiMuon

    return NaiveYogiMuon(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps_yogi=1e-3,
        ns5_iters=5,
    )


def _build_racaso_hutchinson(
    params: List[torch.Tensor], lr: float, problem=None, **_
) -> torch.optim.Optimizer:
    from bench.optimizers.racaso_hvp_wrappers import RACASOHutchinson

    opt = RACASOHutchinson(params, lr=lr)
    if problem is not None:
        opt.set_hvp_context(problem.forward, params)
    return opt


def _build_racaso_gnb(
    params: List[torch.Tensor], lr: float, problem=None, **_
) -> torch.optim.Optimizer:
    from bench.optimizers.racaso_hvp_wrappers import RACASOGNB

    opt = RACASOGNB(params, lr=lr)
    if problem is not None:
        logits_fn = getattr(problem, "logits_fn", None)
        if logits_fn is None:
            raise NotImplementedError(
                f"racaso_gnb requires the problem to expose a "
                f"logits_fn(params) -> [B, C] method; "
                f"{type(problem).__name__} does not. Use a classification "
                f"problem (e.g. p6_classification) for GNB."
            )
        opt.set_hvp_context(logits_fn, params)
    return opt


def _build_liger(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    from bench.optimizers.liger import Liger

    return Liger(params, lr=lr, betas=(0.9, 0.99), eps_yogi=1e-3, weight_decay=0.0)


def _build_muogi(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    from bench.optimizers.muogi import Muogi

    return Muogi(params, lr=lr)


def _build_ramuogi(params: List[torch.Tensor], lr: float, **_) -> torch.optim.Optimizer:
    from bench.optimizers.ramuogi import RAMuogi

    return RAMuogi(params, lr=lr)


def build_optimizer(
    name: str,
    params: List[torch.Tensor],
    lr: float,
    problem=None,
) -> torch.optim.Optimizer:
    """Construct a baseline optimizer by canonical short name.

    Args:
        name: one of ``KNOWN_OPTIMIZERS``.
        params: list of parameter tensors with ``requires_grad=True``.
        lr: learning rate.
        problem: optional ``BenchProblem`` instance, only used by the
            two RACASO HVP-strategy wrappers (``racaso_hutchinson`` /
            ``racaso_gnb``). Other builders ignore it.

    Returns:
        A fully constructed ``torch.optim.Optimizer``.

    Raises:
        ValueError: if ``name`` is not in ``KNOWN_OPTIMIZERS``.
        NotImplementedError: for baselines whose implementation has not
            been vendored yet (muon / sophia / soap), or for
            ``racaso_gnb`` when ``problem`` does not provide
            ``logits_fn``.
    """
    if name not in KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer name '{name}'; "
            f"known: {sorted(KNOWN_OPTIMIZERS)}"
        )
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not params:
        raise ValueError("params must be a non-empty list of tensors")

    builders = {
        "adam": _build_adam,
        "adamw": _build_adamw,
        "yogi": _build_yogi,
        "lion": _build_lion,
        "naive_yogi_muon": _build_naive_yogi_muon,
        "racaso_hutchinson": _build_racaso_hutchinson,
        "racaso_gnb": _build_racaso_gnb,
        "liger": _build_liger,
        "muogi": _build_muogi,
        "ramuogi": _build_ramuogi,
    }
    if name in builders:
        return builders[name](params, lr, problem=problem)
    if name in ("muon", "sophia", "soap"):
        raise NotImplementedError(_vendor_pointer(name))

    raise AssertionError(f"unreachable: optimizer {name} not dispatched")
