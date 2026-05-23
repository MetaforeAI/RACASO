"""R3 — NanoGPT (byte-level) on WikiText-2.

Industry-credible LM benchmark at NanoGPT scale (~30M params). Byte-
level tokenization (vocab 256) keeps the bench dependency-free — no
transformers/tokenizers libraries, no merges JSON to vendor. Byte-level
modeling is a legitimate published practice (ByT5, MEGABYTE).

Specs:
    - Model: 6-layer transformer, hidden=384, heads=6, seq_len=256.
    - Dataset: WikiText-2 raw train split (~10MB of bytes).
    - Training: 1000 steps, batch 8 (= 2048 bytes/step).
    - Convergence tol: train loss < 5.0 (uniform 256-class baseline ≈ 5.55).

This problem also exposes ``logits_fn`` for RACASO GNB.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from bench.datasets.wikitext2_loader import get_wikitext2_bytes
from bench.models.nanogpt import NanoGPT
from bench.problems.base import BenchProblem


_VOCAB_SIZE = 256
_SEQ_LEN = 256
_BATCH_SIZE = 8


class R3NanoGPTWikitext2(BenchProblem):
    """Byte-level NanoGPT on WikiText-2 (industry-credible LM benchmark)."""

    name = "r3_nanogpt_wikitext2"
    max_steps = 1000
    converged_tol = 5.0

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self._model = NanoGPT(
            vocab_size=_VOCAB_SIZE,
            hidden=384,
            heads=6,
            layers=6,
            seq_len=_SEQ_LEN,
        ).to(self.device)
        self._data = get_wikitext2_bytes("train").to(self.device)
        self._last_x: torch.Tensor | None = None
        self._last_y: torch.Tensor | None = None

    def init_params(self) -> List[torch.Tensor]:
        return list(self._model.parameters())

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self._data) - _SEQ_LEN - 1
        idx = torch.randint(0, n, (_BATCH_SIZE,))
        x = torch.stack([self._data[i : i + _SEQ_LEN] for i in idx])
        y = torch.stack([self._data[i + 1 : i + 1 + _SEQ_LEN] for i in idx])
        self._last_x, self._last_y = x, y
        return x, y

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        del params
        x, y = self._next_batch()
        logits = self._model(x)
        return F.cross_entropy(
            logits.view(-1, _VOCAB_SIZE), y.view(-1), reduction="mean"
        )

    def logits_fn(self, params: List[torch.Tensor]) -> torch.Tensor:
        del params
        if self._last_x is None:
            self._next_batch()
        assert self._last_x is not None
        return self._model(self._last_x).view(-1, _VOCAB_SIZE)

    def converged(self, current_loss: float, step: int) -> bool:
        return current_loss < self.converged_tol
