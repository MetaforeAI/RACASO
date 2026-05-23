"""Render the paper figures from a bench_results.csv produced by run_bench.

Synthetic problem figures:
  fig_p1_off_axis_quad.png        — P1, loss-vs-step
  fig_p2_rosenbrock.png           — P2a + P2b combined
  fig_p3_saddle.png               — P3a + P3b combined
  fig_p4_row_spread.png           — P4, loss-vs-step
  fig_p5_div_backward.png         — P5, loss-vs-step
  fig_p6_classification.png       — P6 classification (Hutchinson/GNB)

Real-task figures (if present):
  fig_r1_cifar10.png              — R1, loss-vs-step
  fig_r2_charlm.png               — R2, loss-vs-step
  fig_r3_nanogpt.png              — R3, loss-vs-step

Safety-counter bar chart:
  fig_safety_counters.png         — L1..L5 totals per optimizer

Usage:
    python bench/plot_bench.py --input bench/results.csv --output bench/figs/
"""

from __future__ import annotations

import argparse
import csv
import sys

# Long loss_trajectory cells can exceed Python's default CSV field limit.
csv.field_size_limit(sys.maxsize)
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_OPTIMIZER_COLORS = {
    "adam":              "#888888",
    "adamw":             "#444444",
    "yogi":              "#1f77b4",
    "lion":              "#ff7f0e",
    "liger":             "#d62728",
    "muogi":             "#2ca02c",
    "ramuogi":           "#17becf",
    "racaso_hutchinson": "#9467bd",
    "racaso_gnb":        "#7f4cba",
    "naive_yogi_muon":   "#bcbd22",
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
            v = float(tok)
        except ValueError:
            v = float("nan")
        out.append(v)
    return out


def _filter(rows: List[dict], **kw) -> List[dict]:
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == str(v) or r.get(k) == v]
    return out


def _loss_curves(
    rows: List[dict],
    problems: List[str],
    title: str,
    out: Path,
) -> None:
    """Loss-vs-step overlay across optimizers.

    Accepts a list of problem names so combined-subproblem plots
    (e.g. P2a + P2b) can be rendered together. The first matching
    problem provides the trajectory length scale.
    """
    sub = [r for r in rows if r["problem"] in problems]
    if not sub:
        print(f"  no {problems} data; skipping {out}")
        return
    opts = sorted({r["optimizer"] for r in sub})
    fig, ax = plt.subplots(figsize=(10, 5))
    for opt in opts:
        cand = [r for r in sub if r["optimizer"] == opt]
        by_lr: Dict[str, List[dict]] = defaultdict(list)
        for r in cand:
            by_lr[r["lr"]].append(r)
        def _score(lst: List[dict]) -> float:
            vals = []
            for r in lst:
                try:
                    v = float(r["final_loss"])
                    if math.isfinite(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    continue
            return sum(vals) / len(vals) if vals else float("inf")
        if not by_lr:
            continue
        best_lr = min(by_lr, key=lambda k: _score(by_lr[k]))
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in by_lr[best_lr]]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        avg = [sum(c) / len(c) for c in zip(*padded)]
        ax.plot(
            range(1, len(avg) + 1), avg,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            label=f"{opt} (lr={best_lr})",
            linewidth=1.5,
        )
    ax.set_yscale("symlog")  # P3 saddle goes negative; use symlog
    ax.set_xlabel("step")
    ax.set_ylabel("loss (symlog)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _safety_counters(rows: List[dict], out: Path) -> None:
    counters: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {f"l{i}": 0 for i in range(1, 6)}
    )
    for r in rows:
        opt = r["optimizer"]
        for i in range(1, 6):
            try:
                counters[opt][f"l{i}"] += int(r.get(f"l{i}_count", 0) or 0)
            except (TypeError, ValueError):
                continue
    opts = sorted(counters)
    if not opts:
        print(f"  no safety-counter data; skipping {out}")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = [0] * len(opts)
    layer_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, layer in enumerate(("l1", "l2", "l3", "l4", "l5")):
        heights = [counters[o][layer] for o in opts]
        ax.bar(opts, heights, bottom=bottoms,
               color=layer_colors[i], label=layer.upper())
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("total safety-counter firings")
    ax.set_title("Safety chain (L1-L5) firing counts per optimizer (all problems)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(title="layer", loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="results.csv")
    ap.add_argument("--output", type=Path, default=Path("bench/figs"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input)
    print(f"loaded {len(rows)} rows from {args.input}")

    _loss_curves(rows, ["p1_off_axis_quad"],
                 "P1 — Off-axis quadratic (C1)",
                 args.output / "fig_p1_off_axis_quad.png")
    _loss_curves(rows, ["p2a_rosenbrock_2d", "p2b_rosenbrock_n100"],
                 "P2 — Rosenbrock (2D + N=100)",
                 args.output / "fig_p2_rosenbrock.png")
    _loss_curves(rows, ["p3a_saddle_2d", "p3b_saddle_n20"],
                 "P3 — Saddle (2D + N=20) (C2 + C3)",
                 args.output / "fig_p3_saddle.png")
    _loss_curves(rows, ["p4_row_spread"],
                 "P4 — Row-spread pathology (C4 + C5)",
                 args.output / "fig_p4_row_spread.png")
    _loss_curves(rows, ["p5_div_backward"],
                 "P5 — DivBackward0 hazard (C6)",
                 args.output / "fig_p5_div_backward.png")
    _loss_curves(rows, ["p6_classification"],
                 "P6 — Tiny classification (Hutchinson vs GNB)",
                 args.output / "fig_p6_classification.png")

    _loss_curves(rows, ["r1_cifar10_resnet18"],
                 "R1 — CIFAR-10 ResNet-18: training loss",
                 args.output / "fig_r1_cifar10.png")
    _loss_curves(rows, ["r2_charlm_shakespeare"],
                 "R2 — Char-LM on tiny-shakespeare: training loss",
                 args.output / "fig_r2_charlm.png")
    _loss_curves(rows, ["r3_nanogpt_wikitext2"],
                 "R3 — NanoGPT on WikiText-2: training loss",
                 args.output / "fig_r3_nanogpt.png")

    _safety_counters(rows, args.output / "fig_safety_counters.png")


if __name__ == "__main__":
    main()
