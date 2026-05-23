"""Benchmark problems live here.

Importing this package imports every problem module so that
``BenchProblem``'s subclass registry is populated and ``run_bench.py``
can discover them via ``_registered_problems()``.

Problem set (validates RACASO paper claims C1-C7 + GNB):
    p1_off_axis_quad   — C1       (rotation matters under off-diagonal curvature)
    p2_rosenbrock      — C1       (registers p2a_rosenbrock_2d + p2b_rosenbrock_n100)
    p3_saddle          — C2 + C3  (registers p3a_saddle_2d + p3b_saddle_n20)
    p4_row_spread      — C4 + C5  (spread cap + eigh safe-skip)
    p5_div_backward    — C6       (L5 safe-skip on unbounded second derivative)
    p6_classification  — GNB      (softmax CE so the Gauss-Newton-Bartlett
                                    HVP strategy from paper §2.2.2 has a
                                    problem to run on)
"""

from bench.problems import (
    p1_off_axis_quad,
    p2_rosenbrock,
    p3_saddle,
    p4_row_spread,
    p5_div_backward,
    p6_classification,
    r1_cifar10_resnet18,
    r2_charlm_shakespeare,
    r3_nanogpt_wikitext2,
)

__all__ = [
    "p1_off_axis_quad",
    "p2_rosenbrock",
    "p3_saddle",
    "p4_row_spread",
    "p5_div_backward",
    "p6_classification",
    "r1_cifar10_resnet18",
    "r2_charlm_shakespeare",
    "r3_nanogpt_wikitext2",
]
