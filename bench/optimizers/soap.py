"""SOAP wrapper for bench use.

Re-exports `SOAP` from the verbatim upstream copy
(`_soap_upstream.py`). Provenance header lives in the upstream file.

Reference: Vyas et al. 2024, "SOAP: Improving and Stabilizing Shampoo
using Adam" (arXiv:2409.11321). Upstream:
https://github.com/nikhilvyas/SOAP (MIT License, © 2024 Nikhil Vyas).
"""
from __future__ import annotations

from bench.optimizers._soap_upstream import SOAP

__all__ = ["SOAP"]
