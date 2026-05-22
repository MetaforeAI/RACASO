"""Benchmark problems live here.

Phase 2: the five problem modules P1-P5 land here. Importing this
package imports every problem module so that ``BenchProblem``'s
subclass registry is populated and ``run_bench.py`` can discover
them via ``_registered_problems()``.

Problem set (validates RACASO paper claims C1-C7):
    p1_off_axis_quad   — C1       (rotation matters under off-diagonal curvature)
    p2_rosenbrock      — C1       (registers p2a_rosenbrock_2d + p2b_rosenbrock_n100)
    p3_saddle          — C2 + C3  (registers p3a_saddle_2d + p3b_saddle_n20)
    p4_row_spread      — C4 + C5  (spread cap + eigh safe-skip)
    p5_div_backward    — C6       (L5 safe-skip on unbounded second derivative)
"""

from bench.problems import (
    p1_off_axis_quad,
    p2_rosenbrock,
    p3_saddle,
    p4_row_spread,
    p5_div_backward,
)

__all__ = [
    "p1_off_axis_quad",
    "p2_rosenbrock",
    "p3_saddle",
    "p4_row_spread",
    "p5_div_backward",
]
