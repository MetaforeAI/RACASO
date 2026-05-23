"""R2 — Char-level LM on tiny-shakespeare.

Karpathy-canonical lightweight LM benchmark. Trains a ~3M-param
char-level transformer on tiny-shakespeare for ``max_steps`` steps,
reports final train loss.

Vocab is ASCII (128 chars); any byte > 127 in the corpus would be
clamped at construction (tiny-shakespeare is pure ASCII so this is a
no-op in practice). Sequence length 128, batch 32 = 4096 tokens/step,
which fits comfortably in 4GB of GPU memory.

This problem also exposes ``logits_fn`` so RACASO's GNB strategy works.

Convergence tolerance: train loss < 1.5 (uniform-prior char baseline
is ~log(128) ≈ 4.85; a well-trained char-LM reaches ~1.0-1.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from bench.models.charlm import CharLM
from bench.problems.base import BenchProblem


_VOCAB_SIZE = 128
_SEQ_LEN = 128
_BATCH_SIZE = 32
_DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"


def _load_corpus() -> torch.Tensor:
    """Load tiny-shakespeare as a flat tensor of byte indices."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"tinyshakespeare not found at {_DATA_PATH}. "
            "It should be vendored in bench/datasets/."
        )
    text = _DATA_PATH.read_text(encoding="utf-8")
    # Clamp non-ASCII bytes to 0; tiny-shakespeare is pure ASCII so this
    # is structurally a no-op but documented.
    data = torch.tensor(
        [min(ord(c), _VOCAB_SIZE - 1) for c in text], dtype=torch.long
    )
    return data


class R2CharLMShakespeare(BenchProblem):
    """Char-level transformer LM on tiny-shakespeare."""

    name = "r2_charlm_shakespeare"
    max_steps = 3000
    converged_tol = 1.5

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        self._model = CharLM(
            vocab_size=_VOCAB_SIZE,
            hidden=256,
            heads=4,
            layers=4,
            seq_len=_SEQ_LEN,
        ).to(self.device)
        self._data = _load_corpus().to(self.device)
        # 10% held out as the eval split (used by future logits-eval).
        n = len(self._data)
        cut = int(0.9 * n)
        self._train_data = self._data[:cut]
        self._eval_data = self._data[cut:]
        self._last_x: torch.Tensor | None = None
        self._last_y: torch.Tensor | None = None

    def init_params(self) -> List[torch.Tensor]:
        return list(self._model.parameters())

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self._train_data) - _SEQ_LEN - 1
        # Use CPU sampling to keep determinism cheap and the generator
        # state aligned across CPU/CUDA runs.
        idx = torch.randint(0, n, (_BATCH_SIZE,))
        # Gather slices of length seq_len; targets are shifted by 1.
        x = torch.stack([self._train_data[i : i + _SEQ_LEN] for i in idx])
        y = torch.stack(
            [self._train_data[i + 1 : i + 1 + _SEQ_LEN] for i in idx]
        )
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
        """For RACASO GNB: return [B*T, V] flattened logits on last batch."""
        del params
        if self._last_x is None:
            self._next_batch()
        assert self._last_x is not None
        logits = self._model(self._last_x)
        return logits.view(-1, _VOCAB_SIZE)

    def converged(self, current_loss: float, step: int) -> bool:
        return current_loss < self.converged_tol
