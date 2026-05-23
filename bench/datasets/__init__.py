"""Bench-vendored dataset helpers.

  - ``cifar10_loader.py``: CIFAR-10 via torchvision (downloads on first call).
  - ``tinyshakespeare.txt``: 1MB Shakespeare corpus, vendored verbatim
    (Karpathy's char-LM dataset, public domain).
  - ``wikitext2_loader.py``: WikiText-2-raw downloader (~13MB, salesforce
    research mirror), with SHA256 verification.

Shared data root: ``BENCH_DATA_DIR`` env var or ``./bench/datasets/_cache``.
"""

import os
from pathlib import Path


def get_data_root() -> Path:
    """Resolve the dataset cache directory.

    Order of resolution:
      1. ``$BENCH_DATA_DIR`` if set.
      2. Per-repo fallback at ``bench/datasets/_cache``.
    """
    env = os.environ.get("BENCH_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = Path(__file__).resolve().parent / "_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
