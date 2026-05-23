"""P6 — Tiny Classification.

A 2-layer MLP on a synthetic 10-class classification task with softmax
+ cross-entropy. Exists so the GNB (Gauss-Newton-Bartlett) HVP strategy
has a problem to run on — GNB requires a softmax output to sample
synthetic labels from, which the regression problems P1–P5 don't have.

Architecture:
    W1: [D, H]  matrix params (Lion-route equivalent for non-RACASO opts)
    W2: [H, C]  classification head
    b1: [H]     vector params
    b2: [C]

with D=64 input dim, H=32 hidden, C=10 classes, B=128 batch.

Loss is mean cross-entropy. Target labels come from a held-out random
linear classifier on the inputs so the optimum is well-defined.

Convergence tolerance: loss < log(10) - 1.0 (i.e. cross-entropy meaningfully
below the uniform-prior 2.303 baseline).
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


_INPUT_DIM = 64
_HIDDEN = 32
_NUM_CLASSES = 10
_BATCH = 128


class P6Classification(BenchProblem):
    """2-layer MLP on synthetic 10-class classification.

    Exposes a ``logits_fn(params)`` method that the GNB wrapper needs.
    The Hutchinson wrapper just uses ``forward(params)`` (which returns
    scalar CE loss); both run against the same problem instance.
    """

    name = "p6_classification"
    max_steps = 2000
    converged_tol = 1.3  # below ~log(10) = 2.303

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        gen = self._generator
        # Target classifier: a fixed random linear map from input to class logits.
        self._classifier_W = (torch.randn(
            _INPUT_DIM, _NUM_CLASSES, generator=gen
        ) * 0.5).to(self.device)
        # Fixed batch (stationary loss surface across steps).
        self._x = torch.randn(_BATCH, _INPUT_DIM, generator=gen).to(self.device)
        with torch.no_grad():
            true_logits = self._x @ self._classifier_W
            self._y = true_logits.argmax(dim=-1)

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        # Xavier-ish init.
        W1 = (torch.randn(_INPUT_DIM, _HIDDEN, generator=gen) * (1.0 / _INPUT_DIM) ** 0.5).to(self.device)
        b1 = torch.zeros(_HIDDEN, device=self.device)
        W2 = (torch.randn(_HIDDEN, _NUM_CLASSES, generator=gen) * (1.0 / _HIDDEN) ** 0.5).to(self.device)
        b2 = torch.zeros(_NUM_CLASSES, device=self.device)
        for p in (W1, b1, W2, b2):
            p.requires_grad_(True)
        return [W1, b1, W2, b2]

    def logits_fn(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Return [B, C] logits with autograd graph attached.

        This is the entry point the GNB wrapper calls to sample
        synthetic labels and compute its single first-order backward.
        """
        W1, b1, W2, b2 = params
        h = torch.relu(self._x @ W1 + b1)
        logits = h @ W2 + b2
        return logits

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        logits = self.logits_fn(params)
        return torch.nn.functional.cross_entropy(logits, self._y, reduction="mean")
