"""RACASO optimizer — re-export of the single source of truth.

The canonical implementation lives in the top-level ``racaso.py``. This
module used to be a vendored copy; it is now a thin re-export so there is
exactly one RACASO implementation in the repo (no drift between two files).

Import either way — both resolve to the same class:

    from racaso import RACASO                      # canonical
    from bench.optimizers.racaso import RACASO     # via this re-export
"""

from __future__ import annotations

from racaso import RACASO, _safe_eig_with_residual

__all__ = ["RACASO", "_safe_eig_with_residual"]
