"""Phase 1 sanity tests for the benchmark infrastructure.

These tests do NOT depend on Phase 2 problem modules. They use small,
self-contained ``BenchProblem`` subclasses defined inline to verify the
contract, the wrapper dispatch, the CSV schema, the plot-stub error
messages, and reproducibility under a fixed seed.

Run with::

    pytest RACASO/bench/tests/test_infrastructure.py

Hard rule (per CLAUDE.md): tests that ``import torch`` must run in a
clean Python process — pytest qualifies (no heavyweight/Triton import).
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd
import pytest
import torch

from bench.optimizers.wrappers import KNOWN_OPTIMIZERS, build_optimizer
from bench.problems.base import BenchProblem
from bench.run_bench import CSV_COLUMNS, run_one
from bench import plot_bench


# ---------------------------------------------------------------------------
# Inline test problem — never registered globally; used only to exercise the
# contract. Defined as a fresh class on each test invocation that needs it.
# ---------------------------------------------------------------------------


class _TinyQuadratic(BenchProblem):
    """Minimal axis-aligned quadratic: f(w) = 0.5 * sum(w**2).

    Used to exercise the contract in tests. Not a Phase 2 problem.
    """

    name = "test_tiny_quadratic"
    max_steps = 50
    converged_tol = 1e-6

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        w = torch.randn(4, generator=gen, dtype=torch.float64)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        return 0.5 * (w * w).sum()


# ---------------------------------------------------------------------------
# Contract — BenchProblem cannot be instantiated abstractly.
# ---------------------------------------------------------------------------


def test_benchproblem_is_abstract():
    with pytest.raises(TypeError):
        BenchProblem(seed=0)  # type: ignore[abstract]


def test_benchproblem_subclass_missing_init_params_fails():
    class _BadProblem(BenchProblem):
        name = "bad"
        max_steps = 1
        converged_tol = 0.0

    with pytest.raises(TypeError):
        _BadProblem(seed=0)  # type: ignore[abstract]


def test_benchproblem_default_loss_and_grad_uses_autograd():
    problem = _TinyQuadratic(seed=0)
    params = problem.init_params()
    loss, grads = problem.loss_and_grad(params)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert len(grads) == 1
    assert grads[0].shape == params[0].shape
    # Gradient of 0.5 * w^2 is w; verify analytically.
    assert torch.allclose(grads[0], params[0].detach())


def test_benchproblem_converged_default():
    problem = _TinyQuadratic(seed=0)
    assert problem.converged(1e-9, step=10) is True
    assert problem.converged(1.0, step=10) is False


# ---------------------------------------------------------------------------
# build_optimizer — dispatch table
# ---------------------------------------------------------------------------


_IMPLEMENTED_BASELINES = ("adam", "adamw", "yogi")


@pytest.mark.parametrize("name", _IMPLEMENTED_BASELINES)
def test_build_optimizer_returns_optimizer(name: str):
    params = [torch.randn(3, requires_grad=True)]
    opt = build_optimizer(name, params, lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.parametrize(
    # All four (muon, lion, sophia, soap) have since been vendored.
    # Parametrize empty so the test is a no-op until a new
    # not-yet-vendored baseline appears. We keep the function so the
    # naming convention is preserved for future additions.
    "name", ()
)
def test_build_optimizer_not_vendored_raises(name: str):
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(NotImplementedError) as excinfo:
        build_optimizer(name, params, lr=1e-3)
    assert "README" in str(excinfo.value) or "vendored" in str(excinfo.value)


def test_build_optimizer_unknown_name_raises():
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(ValueError):
        build_optimizer("nonexistent", params, lr=1e-3)


def test_build_optimizer_rejects_empty_params():
    with pytest.raises(ValueError):
        build_optimizer("adam", [], lr=1e-3)


def test_build_optimizer_rejects_nonpositive_lr():
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(ValueError):
        build_optimizer("adam", params, lr=0.0)


def test_known_optimizers_includes_all_eight():
    """KNOWN_OPTIMIZERS should be a superset of the eight original
    baselines (adam, adamw, yogi, muon, lion, sophia, soap, racaso).
    Additional sibling optimizers (liger, muogi, ramuogi,
    naive_yogi_muon, racaso_gnb) may also be present."""
    required = {
        "adam", "adamw", "yogi", "muon", "lion", "sophia", "soap",
        "racaso_hutchinson", "racaso_gnb",
    }
    assert required.issubset(set(KNOWN_OPTIMIZERS))


# ---------------------------------------------------------------------------
# CSV schema — run_one returns exactly the documented columns.
# ---------------------------------------------------------------------------


def test_run_one_csv_schema():
    problem = _TinyQuadratic(seed=0)
    row = run_one(problem, "adam", lr=1e-2, seed=0)
    assert set(row.keys()) == set(CSV_COLUMNS)
    # Spot-check types.
    assert isinstance(row["problem"], str)
    assert isinstance(row["optimizer"], str)
    assert isinstance(row["lr"], float)
    assert isinstance(row["seed"], int)
    assert isinstance(row["steps"], int)
    assert isinstance(row["loss_trajectory"], str)
    # Trajectory string parses back to floats.
    parsed = plot_bench._parse_trajectory(row["loss_trajectory"])
    assert len(parsed) == row["steps"]


# ---------------------------------------------------------------------------
# Reproducibility — same (problem, optimizer, lr, seed) yields identical
# loss trajectories across two runs.
# ---------------------------------------------------------------------------


def test_run_one_reproducibility():
    p1 = _TinyQuadratic(seed=123)
    row1 = run_one(p1, "adam", lr=1e-2, seed=123)

    p2 = _TinyQuadratic(seed=123)
    row2 = run_one(p2, "adam", lr=1e-2, seed=123)

    traj1 = plot_bench._parse_trajectory(row1["loss_trajectory"])
    traj2 = plot_bench._parse_trajectory(row2["loss_trajectory"])
    assert traj1 == traj2
    assert row1["final_loss"] == row2["final_loss"]
    assert row1["steps"] == row2["steps"]


# ---------------------------------------------------------------------------
# Plot stubs — NotImplementedError with the right pointer.
# ---------------------------------------------------------------------------


def test_plot_stubs_raise_not_implemented():
    df = pd.DataFrame(columns=list(CSV_COLUMNS))
    stubs = [
        lambda: plot_bench.plot_loss_curves(df, "p1", "out.png"),
        lambda: plot_bench.plot_wall_clock_pareto(df, "p1", "out.png"),
        lambda: plot_bench.plot_lr_sensitivity(df, "p1", "out.png"),
        lambda: plot_bench.plot_optimizer_vs_problem_heatmap(df, "out.png"),
        lambda: plot_bench.plot_safety_chain_activations(df, "out.png"),
        lambda: plot_bench.plot_hutchinson_vs_gnb_curvature(df, "p1", "out.png"),
    ]
    for stub in stubs:
        with pytest.raises(NotImplementedError) as excinfo:
            stub()
        assert "Phase 4" in str(excinfo.value)


# ---------------------------------------------------------------------------
# load_results — parses trajectory column back.
# ---------------------------------------------------------------------------


def test_load_results_parses_trajectory(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    header = ",".join(CSV_COLUMNS)
    # One row with a 3-step trajectory.
    traj = "1.0;0.5;0.25"
    row = ",".join(
        [
            "test_tiny_quadratic",  # problem
            "adam",  # optimizer
            "0.01",  # lr
            "0",  # seed
            "3",  # steps
            "-1",  # convergence_step
            "0.25",  # final_loss
            "1.0",  # wall_clock_per_step_us
            "0",  # nan_count
            "0",  # l1_count
            "0",  # l2_count
            "0",  # l3_count
            "0",  # l4_count
            "0",  # l5_count
            traj,
        ]
    )
    csv_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    df = plot_bench.load_results(str(csv_path))
    assert len(df) == 1
    assert df.loc[0, "loss_trajectory"] == [1.0, 0.5, 0.25]
