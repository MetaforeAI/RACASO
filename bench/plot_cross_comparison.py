"""Cross-comparison figure: all 8 optimizers on R1/R2/R3 real-task problems.

Produces a single multi-panel figure suitable for inclusion as
"Figure X: head-to-head comparison" in each of the 3 papers (Liger,
Muogi/RAMuogi, RACASO).

Layout: three rows (R1, R2, R3) × two columns (loss curve, final-metric
bar chart). Each row's loss-curve panel shows mean-over-seeds with shaded
±std band; the bar chart shows the best final-metric per optimizer.

Reads ``results.csv`` produced by ``run_bench.py --sweep``; expects rows
for problems ``r1_cifar10_resnet18``, ``r2_charlm_shakespeare``,
``r3_nanogpt_wikitext2`` with all 8 optimizers represented.

Usage:
    python bench/plot_cross_comparison.py --input bench/results.csv \\
        --output bench/figs/cross_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import sys
csv.field_size_limit(sys.maxsize)
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_OPT_ORDER = (
    "adam", "adamw", "yogi", "lion",
    "liger", "muogi", "ramuogi",
    "racaso", "racaso_hutchinson", "racaso_gnb",
)

_OPT_COLOR = {
    "adam":              "#888888",
    "adamw":             "#444444",
    "yogi":              "#1f77b4",
    "lion":              "#ff7f0e",
    "liger":             "#d62728",
    "muogi":             "#2ca02c",
    "ramuogi":           "#17becf",
    "racaso":            "#9467bd",
    "racaso_hutchinson": "#9467bd",
    "racaso_gnb":        "#7f4cba",
}

_PROBLEM_LABELS = {
    "r1_cifar10_resnet18":    "R1 — CIFAR-10 ResNet-18",
    "r2_charlm_shakespeare":  "R2 — Char-LM (tiny-shakespeare)",
    "r3_nanogpt_wikitext2":   "R3 — NanoGPT (WikiText-2, byte-level)",
}


def _read_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _parse_trajectory(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(";"):
        try:
            out.append(float(tok))
        except ValueError:
            out.append(float("nan"))
    return out


def _mean_band(
    trajectories: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad-to-max-length and compute mean ± std across seeds."""
    if not trajectories:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(t) for t in trajectories)
    padded = []
    for t in trajectories:
        if not t:
            continue
        v = np.array(t + [t[-1]] * (max_len - len(t)), dtype=float)
        padded.append(v)
    if not padded:
        return np.array([]), np.array([]), np.array([])
    mat = np.stack(padded)
    return mat.mean(axis=0), mat.mean(axis=0) - mat.std(axis=0), mat.mean(axis=0) + mat.std(axis=0)


def _present_opts(rows: List[dict]) -> List[str]:
    """Return the list of optimizers actually present in this CSV, in
    canonical order."""
    seen = {r["optimizer"] for r in rows}
    return [o for o in _OPT_ORDER if o in seen]


def _best_lr_rows(rows: List[dict], optimizer: str) -> List[dict]:
    """For a single optimizer, find the LR whose mean final loss across
    seeds is lowest; return all rows for that (optimizer, lr) pair."""
    candidates = [r for r in rows if r["optimizer"] == optimizer]
    if not candidates:
        return []
    by_lr: Dict[str, List[dict]] = defaultdict(list)
    for r in candidates:
        by_lr[r["lr"]].append(r)
    def _score(lst: List[dict]) -> float:
        finals = []
        for r in lst:
            try:
                v = float(r["final_loss"])
                if np.isfinite(v):
                    finals.append(v)
            except (TypeError, ValueError):
                continue
        return float(np.mean(finals)) if finals else float("inf")
    best_lr = min(by_lr, key=lambda lr: _score(by_lr[lr]))
    return by_lr[best_lr]


def _plot_problem(
    ax_curve: plt.Axes,
    ax_bar: plt.Axes,
    rows: List[dict],
    problem: str,
) -> None:
    sub = [r for r in rows if r["problem"] == problem]
    opts = _present_opts(sub)

    ax_curve.set_title(f"{_PROBLEM_LABELS.get(problem, problem)} — loss")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("loss")
    ax_curve.set_yscale("log")
    ax_curve.grid(True, alpha=0.3, which="both")

    bar_finals: Dict[str, float] = {}
    for opt in opts:
        chosen = _best_lr_rows(sub, opt)
        trajectories = [_parse_trajectory(r["loss_trajectory"]) for r in chosen]
        trajectories = [t for t in trajectories if t]
        if not trajectories:
            continue
        mean, lo, hi = _mean_band(trajectories)
        x = np.arange(1, len(mean) + 1)
        color = _OPT_COLOR.get(opt, "#000")
        ax_curve.plot(x, mean, color=color, label=opt, linewidth=1.5)
        ax_curve.fill_between(x, lo, hi, color=color, alpha=0.15)
        # Final-loss bar uses mean of the final point across seeds.
        bar_finals[opt] = float(mean[-1])
    ax_curve.legend(loc="upper right", fontsize=8)

    if bar_finals:
        order = [o for o in opts if o in bar_finals]
        vals = [bar_finals[o] for o in order]
        ax_bar.bar(
            order, vals, color=[_OPT_COLOR.get(o, "#000") for o in order]
        )
        ax_bar.set_title(f"{_PROBLEM_LABELS.get(problem, problem)} — final loss")
        ax_bar.set_ylabel("final loss")
        ax_bar.grid(True, axis="y", alpha=0.3)
        ax_bar.tick_params(axis="x", rotation=30, labelsize=8)


def render(input_csv: Path, output_png: Path) -> None:
    rows = _read_rows(input_csv)
    problems = [
        "r1_cifar10_resnet18",
        "r2_charlm_shakespeare",
        "r3_nanogpt_wikitext2",
    ]
    have = {r["problem"] for r in rows}
    problems = [p for p in problems if p in have]
    if not problems:
        print("[plot_cross_comparison] no R1/R2/R3 rows present; nothing to plot.")
        return
    nrows = len(problems)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    for i, prob in enumerate(problems):
        _plot_problem(axes[i, 0], axes[i, 1], rows, prob)
    fig.suptitle(
        "Cross-comparison: 8 optimizers on real-task benchmarks",
        fontsize=14,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cross_comparison] wrote {output_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="results.csv")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("bench/figs/cross_comparison.png"),
        help="output PNG path",
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.input, args.output)


if __name__ == "__main__":
    main()
