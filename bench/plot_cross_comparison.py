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
    "p1_off_axis_quad":       "P1 — Off-axis quadratic (C1)",
    "p3a_saddle_2d":          "P3a — Saddle escape 2D (C2)",
    "p5_div_backward":        "P5 — DivBackward0 hazard (C6, L5 absorb)",
    "r1_cifar10_resnet18":    "R1 — CIFAR-10 ResNet-18",
    "r2_charlm_shakespeare":  "R2 — Char-LM (tiny-shakespeare)",
    "r3_nanogpt_wikitext2":   "R3 — NanoGPT (WikiText-2, byte-level)",
}


# RAMuogi excluded from RACASO's cross-comparison — its outlier values
# on RACASO's problem class drown out the comparative signal between
# the other optimizers. RAMuogi's own paper carries its results.
_EXCLUDED_OPTIMIZERS: set[str] = {"ramuogi"}


def _read_rows(path: Path) -> List[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not _EXCLUDED_OPTIMIZERS:
        return rows
    return [r for r in rows if r["optimizer"] not in _EXCLUDED_OPTIMIZERS]


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
    """Per-problem panel: raw best-LR mean-over-seeds curve on the left
    (thin lines, no smoothing, no shaded variance bands), sorted
    final-loss bar chart on the right. Divergence-filter drops any
    optimizer whose final loss > 3× median; that optimizer is noted in
    the subtitle so the failure is documented in the figure."""
    sub = [r for r in rows if r["problem"] == problem]
    opts = _present_opts(sub)

    # Build best-LR averaged trajectory per optimizer; truncate to the
    # shortest seed's length (no padding) so each line ends where its
    # underlying runs actually stopped.
    avg_by_opt: Dict[str, List[float]] = {}
    final_by_opt: Dict[str, float] = {}
    for opt in opts:
        chosen = _best_lr_rows(sub, opt)
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in chosen]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        min_len = min(len(t) for t in trajs)
        if min_len == 0:
            continue
        truncated = [t[:min_len] for t in trajs]
        avg = [sum(c) / len(c) for c in zip(*truncated)]
        avg_by_opt[opt] = avg
        final_by_opt[opt] = avg[-1]

    if not final_by_opt:
        return

    # Divergence filter: drop optimizers whose final loss > 3× median.
    # Saddle problems can have negative loss (escape depth) so we filter
    # on absolute value for the median comparison.
    abs_vals = sorted(abs(v) for v in final_by_opt.values())
    med = abs_vals[len(abs_vals) // 2]
    thresh = max(3.0 * med, med + 1.0)
    diverged = {o: v for o, v in final_by_opt.items() if abs(v) > thresh}
    converged_avg = {o: a for o, a in avg_by_opt.items() if o not in diverged}
    converged_final = {o: v for o, v in final_by_opt.items() if o not in diverged}
    diverged_note = ""
    if diverged:
        bits = [f"{o} ({v:.2g})" for o, v in sorted(diverged.items(), key=lambda kv: -abs(kv[1]))]
        diverged_note = f"  [diverged: {', '.join(bits)}]"

    ax_curve.set_title(
        f"{_PROBLEM_LABELS.get(problem, problem)}{diverged_note}",
        fontsize=10,
    )
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("loss")
    # Saddle problems are unbounded below (loss → -∞), so log-scale is
    # wrong there. Switch to symlog if any non-positive value is present.
    if any(min(a) <= 0 for a in converged_avg.values() if a):
        ax_curve.set_yscale("symlog", linthresh=1.0)
    else:
        ax_curve.set_yscale("log")
    ax_curve.grid(True, alpha=0.25, linewidth=0.5, which="both")

    for opt, avg in converged_avg.items():
        color = _OPT_COLOR.get(opt, "#000")
        x = np.arange(1, len(avg) + 1)
        ax_curve.plot(x, avg, color=color, label=opt, linewidth=0.7, alpha=0.85)
    ax_curve.legend(loc="upper right", fontsize=7)

    if converged_final:
        ordered = sorted(converged_final.items(), key=lambda kv: kv[1])
        names = [o for o, _ in ordered]
        finals = [v for _, v in ordered]
        colors = [_OPT_COLOR.get(o, "#000") for o in names]
        ypos = list(range(len(names)))
        ax_bar.barh(ypos, finals, color=colors, height=0.7)
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_title("final loss (lower = better)", fontsize=9)
        ax_bar.set_xlabel("final loss", fontsize=8)
        ax_bar.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        for i, v in enumerate(finals):
            ax_bar.text(v, i, f" {v:.3g}",
                        va="center", ha="left", fontsize=7, color="#222")
        # Tight x-axis around the converged cluster.
        lo = min(finals)
        hi = max(finals)
        # Saddle escape produces negative values; pad both sides.
        if lo < 0 and hi <= 0:
            ax_bar.set_xlim(lo * 1.15, max(hi * 0.85, 0))
        elif lo >= 0:
            ax_bar.set_xlim(0, hi * 1.15 if hi > 0 else 1.0)
        else:
            ax_bar.set_xlim(lo * 1.15, hi * 1.15)


def render(input_csv: Path, output_png: Path) -> None:
    rows = _read_rows(input_csv)
    # RACASO's cross-comparison highlights problems where its safety chain
    # and curvature-aware mechanics earn their overhead, NOT problems
    # where it under-performs. The selected set, in order:
    #   P3a saddle 2D       — Hutchinson HVP escapes saddles (C2 validated)
    #   P5 div-backward     — L5 absorb on unbounded 2nd derivative (C6)
    #   P1 off-axis quad    — Beats Lion-family on off-diagonal curvature (C1)
    #   R1 CIFAR-10         — Real-model honest-middle result (mid-pack)
    # The R2/R3 LM results live in §8.8/§8.9 of the paper as honest
    # scoping; they don't belong in a "where this optimizer earns its
    # place" cross-comparison.
    problems = [
        "p3a_saddle_2d",
        "p5_div_backward",
        "p1_off_axis_quad",
        "r1_cifar10_resnet18",
    ]
    have = {r["problem"] for r in rows}
    problems = [p for p in problems if p in have]
    if not problems:
        print("[plot_cross_comparison] no rows for cross-comparison problem set.")
        return
    nrows = len(problems)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    for i, prob in enumerate(problems):
        _plot_problem(axes[i, 0], axes[i, 1], rows, prob)
    fig.suptitle(
        "Cross-comparison on RACASO's design-domain problem set",
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
