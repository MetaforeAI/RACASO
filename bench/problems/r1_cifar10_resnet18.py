"""R1 — CIFAR-10 ResNet-18.

Industry-standard "does this optimizer train a real model" benchmark.
Trains a ResNet-18 (~11.2M params) on CIFAR-10 for ``max_steps`` steps,
reports final train loss + final test accuracy.

The model is created in ``__init__`` and stashed on ``self._model``;
``init_params()`` returns ``list(self._model.parameters())``. ``forward``
samples the next batch from an infinite-cycling train iterator and
returns the cross-entropy loss tensor with autograd graph attached.

This problem also exposes ``logits_fn`` so RACASO's GNB strategy can
sample synthetic labels from the model's softmax output.

Convergence tolerance: train loss < 0.5 (well-converged CIFAR-10
ResNet-18 typically reaches ~0.1-0.2 train loss in 5k steps with a
sane optimizer).
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.datasets.cifar10_loader import get_cifar10_loaders
from bench.models.resnet18 import ResNet18
from bench.problems.base import BenchProblem


class R1Cifar10ResNet18(BenchProblem):
    """ResNet-18 on CIFAR-10 — industry-credible training benchmark."""

    name = "r1_cifar10_resnet18"
    max_steps = 5000
    converged_tol = 0.5  # train loss

    _BATCH_SIZE = 128

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self._model = ResNet18(num_classes=10).to(self.device)
        self._train_loader, self._test_loader = get_cifar10_loaders(
            batch_size=self._BATCH_SIZE,
            num_workers=0,
            device=device,
        )
        self._train_iter: Iterator = self._make_train_iter()
        # Track last batch for logits_fn() — GNB wrapper needs to be able
        # to compute logits on the same input that was just stepped on.
        self._last_x: torch.Tensor | None = None
        self._last_y: torch.Tensor | None = None

    def _make_train_iter(self) -> Iterator:
        """Infinite iterator over the train loader (re-shuffles per epoch)."""
        while True:
            for batch in self._train_loader:
                yield batch

    def init_params(self) -> List[torch.Tensor]:
        return list(self._model.parameters())

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = next(self._train_iter)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self._last_x, self._last_y = x, y
        return x, y

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        del params  # model parameters are already inside self._model
        x, y = self._next_batch()
        logits = self._model(x)
        return F.cross_entropy(logits, y)

    def logits_fn(self, params: List[torch.Tensor]) -> torch.Tensor:
        """For RACASO GNB: return logits on the most recent batch.

        If no batch has been processed yet, draw a fresh one. The GNB
        wrapper samples synthetic labels from these logits, so they need
        an autograd graph.
        """
        del params
        if self._last_x is None:
            self._next_batch()
        assert self._last_x is not None
        return self._model(self._last_x)

    def converged(self, current_loss: float, step: int) -> bool:
        # Reach converged_tol at any point counts; the harness records
        # the step at which we first cleared it.
        return current_loss < self.converged_tol

    def evaluate_test_accuracy(self) -> float:
        """Compute test accuracy across the full CIFAR-10 test set."""
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self._test_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                preds = self._model(x).argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += y.numel()
        self._model.train()
        return correct / max(1, total)
