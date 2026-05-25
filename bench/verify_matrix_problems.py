"""Smoke-verify the new matrix-version problems activate RACASO's 2-D path.

Runs each (P1, P2b, P3b, P5) for a small number of steps under RACASO,
then asserts the telemetry shows rotation_success_count > 0 (i.e., the
2-D rotation pipeline engaged at least once).
"""
from __future__ import annotations

import torch
from bench.problems.p1_off_axis_quad import P1OffAxisQuadratic
from bench.problems.p2_rosenbrock import P2bRosenbrockN100
from bench.problems.p3_saddle import P3bSaddleN20
from bench.problems.p5_div_backward import P5DivBackward
from bench.optimizers.racaso_hvp_wrappers import RACASOHutchinson


PROBLEMS = [
    ("p1_off_axis_quad",   P1OffAxisQuadratic),
    ("p2b_rosenbrock_n100", P2bRosenbrockN100),
    ("p3b_saddle_n20",     P3bSaddleN20),
    ("p5_div_backward",    P5DivBackward),
]


def main() -> None:
    for name, cls in PROBLEMS:
        problem = cls(seed=0)
        params = problem.init_params()
        # Verify 2-D shape on at least one param.
        shapes = [tuple(p.shape) for p in params]
        has_2d = any(len(s) == 2 for s in shapes)
        opt = RACASOHutchinson(params, lr=1e-4, betas=(0.9, 0.99),
                               refresh_freq=2, hessian_freq=2)
        opt.set_hvp_context(problem.forward, params)
        for step in range(40):
            opt.zero_grad(set_to_none=True)
            loss_val, grads = problem.loss_and_grad(params)
            for p, g in zip(params, grads):
                p.grad = g.detach() if isinstance(g, torch.Tensor) else None
            opt.step()
        tel = opt.get_telemetry()
        counts = opt.get_safety_counts()
        rot_ok = tel.get("rotation_success_count", 0)
        rot_skip = tel.get("rotation_skip_count", 0)
        print(f"{name:24s} | shapes={shapes} 2d={has_2d} "
              f"rot_ok={rot_ok} rot_skip={rot_skip} "
              f"l1={counts['l1']} l3={counts['l3']} l5={counts['l5']} "
              f"loss={loss_val:.3e}")


if __name__ == "__main__":
    main()
