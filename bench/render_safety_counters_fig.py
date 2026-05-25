"""Generate bench/figs/fig_safety_counters.png.

Runs RACASO + GNB variant on P1..P6 matrix problems for ~200 steps and
renders a stacked bar chart of L1..L5 safety-chain activation counts
per (problem, optimizer). Non-RACASO baselines have zero counters by
construction (only RACASO exposes get_safety_counts), so the figure
spotlights when each safety layer fired across the design-domain
problems.

CPU-runnable. Output: bench/figs/fig_safety_counters.png.
"""
from __future__ import annotations

import os
import sys

import torch

from bench.problems.p1_off_axis_quad import P1OffAxisQuadratic
from bench.problems.p2_rosenbrock import P2aRosenbrock2D, P2bRosenbrockN100
from bench.problems.p3_saddle import P3aSaddle2D, P3bSaddleN20
from bench.problems.p4_row_spread import P4RowSpread
from bench.problems.p5_div_backward import P5DivBackward
from bench.problems.p6_classification import P6Classification

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBLEMS = [
    ("P1", P1OffAxisQuadratic),
    ("P2a", P2aRosenbrock2D),
    ("P2b", P2bRosenbrockN100),
    ("P3a", P3aSaddle2D),
    ("P3b", P3bSaddleN20),
    ("P4", P4RowSpread),
    ("P5", P5DivBackward),
    ("P6", P6Classification),
]

L_LABELS = ("L1", "L2", "L3", "L4", "L5")
L_COLORS = {
    "L1": "#ff7f0e",  # spread cap
    "L2": "#2ca02c",  # eigh skip
    "L3": "#d62728",  # HVP skip
    "L4": "#9467bd",  # RAdam gate
    "L5": "#8c564b",  # HVP absorb
}

OPTS = [
    ("racaso_hutchinson", "racaso\nHutchinson"),
    ("racaso_gnb", "racaso\nGNB"),
]

N_STEPS = 200


def _run_one(problem_name, problem_cls, opt_name) -> dict:
    from bench.optimizers.wrappers import build_optimizer
    problem = problem_cls(seed=0)
    params = problem.init_params()
    # GNB requires logits_fn; skip non-classification problems.
    if opt_name == "racaso_gnb" and not hasattr(problem, "logits_fn"):
        return {k: 0 for k in L_LABELS}
    try:
        opt = build_optimizer(opt_name, params, lr=1e-4, problem=problem)
    except NotImplementedError:
        return {k: 0 for k in L_LABELS}
    for step in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        try:
            loss_val, grads = problem.loss_and_grad(params)
        except Exception:
            break
        for p, g in zip(params, grads):
            if isinstance(g, torch.Tensor):
                p.grad = g.detach()
        try:
            opt.step()
        except Exception:
            break
    counts = opt.get_safety_counts() if hasattr(opt, "get_safety_counts") else {k.lower(): 0 for k in L_LABELS}
    return {f"L{i}": int(counts.get(f"l{i}", 0)) for i in range(1, 6)}


def main() -> int:
    rows = []  # (problem_label, opt_label, {L1..L5: int})
    for label, cls in PROBLEMS:
        for opt_name, opt_label in OPTS:
            print(f"[fig] running {label} × {opt_name} ...", file=sys.stderr, flush=True)
            counts = _run_one(label, cls, opt_name)
            rows.append((label, opt_label, counts))
            print(f"      counts={counts}", file=sys.stderr)

    # Render: x = (problem, optimizer) pair, stacked bars of L1..L5.
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = list(range(len(rows)))
    bottoms = [0] * len(rows)
    for L in L_LABELS:
        vals = [r[2][L] for r in rows]
        ax.bar(xs, vals, bottom=bottoms, color=L_COLORS[L], label=L,
               edgecolor="white", linewidth=0.3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    labels = [f"{r[0]}\n{r[1]}" for r in rows]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.set_ylabel("Safety-chain activation count (200 CPU steps)")
    ax.set_title("RACASO L1–L5 safety-chain activations per problem × HVP strategy")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_safety_counters.png")
    fig.savefig(out_path, dpi=130)
    print(f"[fig] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
