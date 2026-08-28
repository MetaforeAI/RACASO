# RACASO Benchmark Suite

Shared infrastructure for comparing RACASO (Hutchinson + GNB paths)
against 7 baseline optimizers on 5 controlled-conditions problems
designed to validate the RACASO paper's claims.

## Phase status

- **Phase 1 (this commit) — infrastructure only.** `BenchProblem` base
  class, `build_optimizer` dispatch, `run_bench.py` harness,
  `plot_bench.py` skeleton, `tests/` sanity suite. No problem modules.
  No runs. No figures.
- **Phase 2 — problem modules.** Five files land in `problems/`
  (`p1_off_axis_quad.py`, `p2_rosenbrock.py`, `p3_saddle.py`,
  `p4_row_spread.py`, `p5_div_backward.py`).
- **Phase 3 — execution.** The sweep is run on a clean host; results
  land in `bench_results.csv`.
- **Phase 4 — plotting.** `plot_bench.py` stubs become real figures.

## How to run (when Phase 2 lands)

Single config:

```bash
python -m bench.run_bench --problem p1_off_axis_quad \
    --optimizer adam --lr 1e-3 --seed 0
```

Full sweep (all problems × all optimizers × all LRs × all seeds):

```bash
python -m bench.run_bench --sweep --out bench_results.csv
```

In Phase 1 the sweep is a documented no-op (no problems registered) —
it writes a header-only CSV and exits cleanly. Single-config invocation
raises a clear error.

Full sweep results (`results.csv`) are large and git-ignored; regenerate via
`run_bench.py --sweep` or request the artifact.

## CSV schema

Columns of `bench_results.csv`, in order:

| column                   | type       | meaning |
|--------------------------|------------|---------|
| `problem`                | str        | problem short name |
| `optimizer`              | str        | optimizer short name |
| `lr`                     | float      | learning rate |
| `seed`                   | int        | random seed |
| `steps`                  | int        | total steps recorded in the trajectory |
| `convergence_step`       | int        | first step where `converged()` returned True; `-1` if never |
| `final_loss`             | float      | last recorded loss (may be `nan`) |
| `wall_clock_per_step_us` | float      | mean `optimizer.step()` time in microseconds |
| `nan_count`              | int        | count of non-finite losses observed |
| `l1_count`               | int        | RACASO L1 spread-cap activations |
| `l2_count`               | int        | RACASO L2 eigh safe-skip count |
| `l3_count`               | int        | RACASO L3 1-D Yogi fallback count |
| `l4_count`               | int        | RACASO L4 RAdam gate count |
| `l5_count`               | int        | RACASO L5 safe-skip count |
| `data_source`            | str        | provenance tag: `racaso_bench_sweep` for native rows or `liger_bench_sweep` for rows imported from the sibling Liger sweep (R1/R2/R3 only) |
| `loss_trajectory`        | str        | full per-step loss history, semicolon-separated floats |

**Data-source provenance.** R1/R2/R3 (CIFAR-10/ResNet-18, char-LM
Shakespeare, NanoGPT WikiText-2) rows are currently imported from the
sibling Liger sweep, where the optimizer was constructed as the
unwrapped `racaso` class. The bench code in this repo is a verbatim
vendored copy of the same bench infrastructure, so the harness
mechanics are identical; only the wrapper layer (i.e.,
`racaso_hutchinson` vs naked `racaso`) differs. A native re-run is
GPU-deferred; the current CSV is marked with `data_source` for
transparency. See paper §8.7-§8.9 footnotes.

The semicolon (not comma) separator inside `loss_trajectory` keeps the
CSV parseable by `csv.DictReader` / `pandas.read_csv` without
quoting tricks. `plot_bench.load_results` parses the column back to
`list[float]` automatically.

## Baselines

| optimizer          | status (Phase 1)                | source |
|--------------------|---------------------------------|--------|
| adam               | implemented                     | `torch.optim.Adam` |
| adamw              | implemented                     | `torch.optim.AdamW` |
| yogi               | implemented (vendored)          | `bench/optimizers/yogi.py` (Zaheer et al. 2018) |
| muon               | NotImplementedError             | Keller Jordan reference — vendor in Phase 2 |
| lion               | NotImplementedError             | `lion-pytorch` or vendored |
| sophia             | NotImplementedError             | official `Liuhong99/Sophia` |
| soap               | NotImplementedError             | Vyas et al. reference |
| racaso_hutchinson  | implemented (lazy import)       | `RACASO/racaso.py` |
| racaso_gnb         | implemented (lazy import)       | `RACASO/racaso.py` |

See `bench/optimizers/README.md` for canonical hyperparameter configs
and the Phase 2 vendoring checklist. Pinned package versions live in
`requirements_bench.txt`.

## Files

```
bench/
├── README.md                       # this file
├── requirements_bench.txt          # pinned versions
├── __init__.py
├── problems/
│   ├── __init__.py                 # empty marker — Phase 2 fills it
│   └── base.py                     # BenchProblem ABC
├── optimizers/
│   ├── README.md                   # canonical configs + vendoring checklist
│   ├── __init__.py
│   ├── wrappers.py                 # build_optimizer dispatch
│   └── yogi.py                     # vendored (Zaheer et al. 2018)
├── run_bench.py                    # harness — run_one + --sweep
├── plot_bench.py                   # load_results + plot stubs
└── tests/
    ├── __init__.py
    └── test_infrastructure.py      # Phase 1 sanity tests
```

## Constraints (from `CLAUDE.md`)

- This bench imports only `torch` (plus `pandas` for plotting). It does
  not import any heavyweight upstream, Triton, or anything that fires CUDA/Triton
  autotune at import time.
- AST-parse every file before committing:
  `python -c "import ast; ast.parse(open(p).read())"`.
- Phase 3 runs on a clean host. Phase 1 ships infrastructure only.
