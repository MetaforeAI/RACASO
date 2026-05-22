"""Phase-2 smoke tests for the five problem modules.

For each registered problem we run 50 Adam steps at ``lr=1e-3`` via the
existing ``run_one`` harness and assert:

* the class instantiates cleanly,
* the trajectory contains a finite first-step loss,
* no NaN losses are observed across the 50 steps,
* the returned row matches the documented CSV schema,
* ``final_loss`` is finite.

These tests do not exercise convergence — Phase 3 owns that. They only
guard against import-time / instantiation regressions and confirm that
each problem produces a well-formed signal under a mainstream baseline.
"""

from __future__ import annotations

import math
from typing import Dict, Type

import pytest

from bench.problems import (  # noqa: F401  (registers subclasses)
    p1_off_axis_quad,
    p2_rosenbrock,
    p3_saddle,
    p4_row_spread,
    p5_div_backward,
)
from bench.problems.base import BenchProblem
from bench.run_bench import CSV_COLUMNS, _registered_problems, run_one


_EXPECTED_NAMES = (
    "p1_off_axis_quad",
    "p2a_rosenbrock_2d",
    "p2b_rosenbrock_n100",
    "p3a_saddle_2d",
    "p3b_saddle_n20",
    "p4_row_spread",
    "p5_div_backward",
)


def test_all_seven_problems_registered():
    registry = _registered_problems()
    for name in _EXPECTED_NAMES:
        assert name in registry, (
            f"problem {name!r} not registered; have {sorted(registry)}"
        )


@pytest.fixture(scope="module")
def registry() -> Dict[str, Type[BenchProblem]]:
    return _registered_problems()


@pytest.mark.parametrize("problem_name", _EXPECTED_NAMES)
def test_problem_smokes_50_adam_steps(
    problem_name: str,
    registry: Dict[str, Type[BenchProblem]],
):
    cls = registry[problem_name]
    original_max = cls.max_steps
    cls.max_steps = 50
    try:
        problem = cls(seed=0)
        row = run_one(problem, "adam", lr=1e-3, seed=0)
    finally:
        cls.max_steps = original_max

    assert set(row.keys()) == set(CSV_COLUMNS)
    traj = [float(x) for x in row["loss_trajectory"].split(";") if x]
    assert len(traj) > 0, f"{problem_name}: trajectory empty"
    assert math.isfinite(traj[0]), f"{problem_name}: first loss not finite"
    assert row["nan_count"] == 0, f"{problem_name}: NaN observed"
    assert math.isfinite(row["final_loss"]), (
        f"{problem_name}: final_loss not finite ({row['final_loss']})"
    )
    # Surface the init/final loss in the pytest stdout for human review.
    print(
        f"\n[smoke] {problem_name}: init={traj[0]:+.4e} "
        f"final={row['final_loss']:+.4e} steps={row['steps']}"
    )


def test_saddle_classes_have_saddle_flag(
    registry: Dict[str, Type[BenchProblem]],
):
    for name in ("p3a_saddle_2d", "p3b_saddle_n20"):
        cls = registry[name]
        assert getattr(cls, "saddle", False) is True, (
            f"{name}: expected class attribute saddle=True"
        )
