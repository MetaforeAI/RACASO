"""Canonical optimizer-construction entry point for the benchmark suite.

``build_optimizer(name, params, lr)`` returns a fully configured
``torch.optim.Optimizer``. All hyperparameters other than the learning
rate are pinned here, identical across RACASO/bench and Muogi/bench so
that head-to-head comparisons share configuration.

Canonical configs (the single source of truth — see ``README.md``):

    adam              : torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8)
    adamw             : torch.optim.AdamW(lr, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0.01)
    yogi              : Yogi(lr, betas=(0.9, 0.999), eps=1e-3,
                             initial_accumulator=1e-6, weight_decay=0.0)
                        — vendored from morpheus/training/optimizers/yogi.py
    muon              : NotImplementedError until vendored from the
                        Keller Jordan reference implementation
    lion              : NotImplementedError until vendored / pip-installed
    sophia            : NotImplementedError until vendored from the
                        official Sophia repo
    soap              : NotImplementedError until vendored from Vyas et al.
    racaso_hutchinson : RACASO(lr, curvature='hutchinson', ...)
                        — imported from RACASO/racaso.py
    racaso_gnb        : RACASO(lr, curvature='gnb', ...)
                        — imported from RACASO/racaso.py

For Phase 1, baselines that need external code raise ``NotImplementedError``
with a pointer to ``bench/optimizers/README.md``. Phase 2 will vendor or
pip-install each missing baseline. The RACASO entries also raise
``NotImplementedError`` if ``racaso.py`` does not yet expose the
expected interface; the harness must be able to construct them once the
RACASO class accepts the documented arguments.
"""

from __future__ import annotations

from typing import List

import torch


KNOWN_OPTIMIZERS = (
    "adam",
    "adamw",
    "yogi",
    "muon",
    "lion",
    "sophia",
    "soap",
    "racaso_hutchinson",
    "racaso_gnb",
)


def _vendor_pointer(opt_name: str) -> str:
    return (
        f"{opt_name} is not vendored yet; "
        "see bench/optimizers/README.md for the canonical source and "
        "pinned version to drop into this wrapper before Phase 3 runs."
    )


def _build_adam(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_adamw(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )


def _build_yogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    # Vendored Yogi from morpheus/training/optimizers/yogi.py — copy of the
    # implementation lives in this directory to keep the bench self-contained
    # without requiring a Morpheus checkout on the path.
    from bench.optimizers.yogi import Yogi

    return Yogi(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0.0,
    )


def _build_racaso(
    params: List[torch.Tensor], lr: float, curvature: str
) -> torch.optim.Optimizer:
    try:
        from racaso import RACASO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotImplementedError(
            "racaso.RACASO is not importable from this environment; "
            "ensure RACASO/ is on sys.path before constructing "
            f"racaso_{curvature}. Original error: {exc}"
        ) from exc

    return RACASO(params, lr=lr, curvature=curvature)


def build_optimizer(
    name: str, params: List[torch.Tensor], lr: float
) -> torch.optim.Optimizer:
    """Construct a baseline optimizer by canonical short name.

    Args:
        name: one of ``KNOWN_OPTIMIZERS``.
        params: list of parameter tensors with ``requires_grad=True``.
        lr: learning rate.

    Returns:
        A fully constructed ``torch.optim.Optimizer``.

    Raises:
        ValueError: if ``name`` is not in ``KNOWN_OPTIMIZERS``.
        NotImplementedError: for baselines whose implementation has not
            been vendored / installed yet (see ``README.md``).
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

    if name == "adam":
        return _build_adam(params, lr)
    if name == "adamw":
        return _build_adamw(params, lr)
    if name == "yogi":
        return _build_yogi(params, lr)
    if name == "muon":
        raise NotImplementedError(_vendor_pointer("muon"))
    if name == "lion":
        raise NotImplementedError(_vendor_pointer("lion"))
    if name == "sophia":
        raise NotImplementedError(_vendor_pointer("sophia"))
    if name == "soap":
        raise NotImplementedError(_vendor_pointer("soap"))
    if name == "racaso_hutchinson":
        return _build_racaso(params, lr, curvature="hutchinson")
    if name == "racaso_gnb":
        return _build_racaso(params, lr, curvature="gnb")

    raise AssertionError(f"unreachable: optimizer {name} not dispatched")
