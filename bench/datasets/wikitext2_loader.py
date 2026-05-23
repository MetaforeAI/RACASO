"""WikiText-2 raw loader.

Downloads to ``BENCH_DATA_DIR`` (or per-repo cache) on first call. The
file lives at the canonical Salesforce-research URL.

Byte-level: returns the raw bytes of the train split as a flat torch
tensor of dtype torch.long, vocab range [0, 256).
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import torch

from bench.datasets import get_data_root


_URL = "https://wikitext.smerity.com/wikitext-2-v1.zip"
_ZIP_NAME = "wikitext-2-v1.zip"
# SHA may vary across mirrors; left empty intentionally — if the
# upstream Salesforce S3 ever comes back, we can pin this and verify.
_ZIP_SHA256 = ""


def _download(target_zip: Path) -> None:
    """Download the WikiText-2 zip.

    Sends a Mozilla user-agent because some mirrors (smerity.com)
    refuse the default ``Python-urllib/3.x`` UA with a 403.
    """
    print(f"[bench] downloading WikiText-2 from {_URL} to {target_zip}")
    req = urllib.request.Request(
        _URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; bench-loader/1.0)"},
    )
    with urllib.request.urlopen(req) as resp, target_zip.open("wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    if _ZIP_SHA256:
        digest = hashlib.sha256(target_zip.read_bytes()).hexdigest()
        if digest != _ZIP_SHA256:
            print(
                f"[bench] WARN: WikiText-2 SHA mismatch (expected {_ZIP_SHA256}, "
                f"got {digest}); proceeding anyway."
            )


def get_wikitext2_bytes(split: str = "train") -> torch.Tensor:
    """Return WikiText-2 ``split`` as a flat tensor of byte indices.

    Splits: ``train`` (~10MB), ``valid`` (~1MB), ``test`` (~1MB).
    """
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be train|valid|test, got {split!r}")
    root = get_data_root()
    extracted_dir = root / "wikitext-2"
    train_path = extracted_dir / f"wiki.{split}.tokens"
    if not train_path.exists():
        zip_path = root / _ZIP_NAME
        if not zip_path.exists():
            _download(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    if not train_path.exists():
        raise FileNotFoundError(
            f"after download+extract, expected {train_path} but it is missing"
        )
    raw = train_path.read_bytes()
    return torch.tensor(list(raw), dtype=torch.long)
