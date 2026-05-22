"""Plotting harness — reads ``bench_results.csv`` and emits the figures
documented in ``plans/sharded-plotting-haven.md`` (RACASO benchmark
section).

Phase 1: ``load_results`` is fully implemented (parses the CSV, splits
the ``loss_trajectory`` column back into a list of floats). All plotting
functions are stubbed with ``NotImplementedError`` plus a docstring
describing what the plot should show; Phase 4 fills them in.

Plotting backend choice (matplotlib vs plotly) is deferred to Phase 4.
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd


def _parse_trajectory(s: object) -> List[float]:
    """Parse the ``loss_trajectory`` column back to a Python list of floats.

    Empty / NaN cells return an empty list. Non-finite literals like
    ``nan`` and ``inf`` are preserved via ``float()``.
    """
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []
    text = str(s).strip()
    if not text:
        return []
    out: List[float] = []
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def load_results(csv_path: str) -> pd.DataFrame:
    """Load ``bench_results.csv`` into a DataFrame.

    Parses ``loss_trajectory`` from its semicolon-encoded string form
    back into a ``list[float]`` column. All other columns are loaded
    with their natural pandas dtypes.

    Args:
        csv_path: path to the CSV produced by ``run_bench.py``.

    Returns:
        A DataFrame with one row per (problem, optimizer, lr, seed) run.
    """
    df = pd.read_csv(csv_path)
    if "loss_trajectory" in df.columns:
        df["loss_trajectory"] = df["loss_trajectory"].map(_parse_trajectory)
    return df


def plot_loss_curves(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """Loss-vs-step plot: 8 curves (one per optimizer), log-y, median +
    IQR shading across seeds, all on the same axes. One figure per
    problem; filtered by ``problem``.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_wall_clock_pareto(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """Pareto curve: wall-clock-to-loss-X (x-axis) vs final-loss (y-axis)
    across optimizers. Highlights which optimizer is fastest to reach
    each target loss; one figure per problem.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_lr_sensitivity(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """LR-sensitivity plot: small multiples (one panel per optimizer)
    showing best-final-loss as a function of LR. Sweeps across the
    documented LR set; one figure per problem.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_optimizer_vs_problem_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Heatmap: optimizer (rows) × problem (cols) → relative final loss
    normalized so Adam=1.0 on each problem. RACASO should appear as the
    consistent leader on P1-P4; GNB path comparable to Hutchinson on
    most, divergent on P3 saddle.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_safety_chain_activations(df: pd.DataFrame, out_path: str) -> None:
    """Stacked bar chart per problem showing what fraction of steps
    triggered each RACASO safety-chain layer (L1/L2/L3/L4/L5). Validates
    the engineering provenance claims in the RACASO paper.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_hutchinson_vs_gnb_curvature(
    df: pd.DataFrame, problem: str, out_path: str
) -> None:
    """Scatter plot comparing per-step rotated-Hessian-diagonal
    (Hutchinson) against the per-step GNB estimate on the SAME problem
    (P1 PSD region: tight correlation; P3 saddle: divergent because
    Hutchinson sees negative curvature, GNB does not).
    """
    raise NotImplementedError("Phase 4 implements plotting")
