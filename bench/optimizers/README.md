# Benchmark Optimizer Wrappers — Canonical Configs

All optimizers are constructed via the single entry point
`bench.optimizers.wrappers.build_optimizer(name, params, lr)`. Everything
except `lr` is pinned here. The pinning is mirrored verbatim in
`Muogi/bench/optimizers/` so head-to-head comparisons share configuration.

## Canonical hyperparameter table

| name              | source                                                 | call                                                                                                | status        |
|-------------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------|
| adam              | `torch.optim`                                          | `Adam(lr=lr, betas=(0.9, 0.999), eps=1e-8)`                                                          | implemented   |
| adamw             | `torch.optim`                                          | `AdamW(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`                                      | implemented   |
| yogi              | vendored from `morpheus/training/optimizers/yogi.py`   | `Yogi(lr=lr, betas=(0.9, 0.999), eps=1e-3, initial_accumulator=1e-6, weight_decay=0.0)`             | implemented   |
| muon              | Keller Jordan reference                                | `Muon(lr=lr, momentum=0.95, nesterov=True, ns_steps=5)`                                              | not vendored  |
| lion              | `lion-pytorch` or vendored                             | `Lion(lr=lr, betas=(0.9, 0.99), weight_decay=0.0)`                                                   | not vendored  |
| sophia            | official `Liuhong99/Sophia`                            | `SophiaG(lr=lr, betas=(0.965, 0.99), rho=0.04, weight_decay=0.0, eps=1e-15)`                          | not vendored  |
| soap              | Vyas et al. reference                                  | `SOAP(lr=lr, betas=(0.95, 0.95), shampoo_beta=0.95, eps=1e-8, weight_decay=0.0, precondition_frequency=10)` | not vendored  |
| racaso_hutchinson | `RACASO/racaso.py`                                     | `RACASO(lr=lr, curvature='hutchinson')`                                                              | implemented*  |
| racaso_gnb        | `RACASO/racaso.py`                                     | `RACASO(lr=lr, curvature='gnb')`                                                                     | implemented*  |

`*` RACASO entries succeed when `racaso.RACASO` is importable from
`sys.path` and accepts the `curvature` keyword. The wrapper does not
import `racaso` at module import time; it imports lazily inside
`build_optimizer` so that bench infrastructure tests can run without
triggering Triton autotune via Morpheus pathways.

## Vendoring checklist for Phase 2

Before Phase 3 can sweep all 8 baselines:

- [ ] **Muon** — copy the Keller Jordan reference implementation into
      `bench/optimizers/muon.py` (Apache-2.0 / public). Wire it into
      `_build_muon` in `wrappers.py`.
- [ ] **Lion** — either `pip install lion-pytorch` (add to
      `requirements_bench.txt`) or vendor `bench/optimizers/lion.py` from
      `lucidrains/lion-pytorch` (MIT).
- [ ] **Sophia** — vendor `bench/optimizers/sophia.py` from the official
      `Liuhong99/Sophia` repo. Use `SophiaG` (Gauss-Newton-Bartlett);
      `SophiaH` is the Hutchinson variant we explicitly want to keep
      *separate* from `racaso_hutchinson` so the contrast is meaningful.
- [ ] **SOAP** — vendor `bench/optimizers/soap.py` from the Vyas et al.
      reference. Apache-2.0.

Each vendored file must include the upstream commit SHA and license
header. Hyperparameter defaults must match the table above exactly.

## Why a single wrapper module

Without this single source, drift creeps in (different default eps for
Adam between two benches → invisible bias in the comparison). The
wrapper pattern guarantees every benchmark in the repo calls
`build_optimizer("adam", ...)` and gets bit-identical optimizer state at
init.
