"""One-shot: add a ``data_source`` column to bench/results.csv.

R1/R2/R3 rows currently use the optimizer name ``racaso`` (sibling
Liger-sweep convention) rather than ``racaso_hutchinson`` (RACASO's own
harness convention). This script:

  1. Reads bench/results.csv.
  2. Inserts ``data_source`` immediately after ``l5_count``.
     - Rows whose optimizer name is ``racaso`` (in any R1/R2/R3 row) AND
       whose problem name is in ``_REAL_TASK_PROBLEMS`` are marked
       ``liger_bench_sweep`` (borrowed from sibling repo's sweep).
     - All other rows are marked ``racaso_bench_sweep`` (native).
  3. Writes back.

Idempotent — if the column already exists, exits without rewriting.
"""
from __future__ import annotations

import csv
import sys
import os

# Long loss_trajectory cells exceed the default csv field limit; raise it.
csv.field_size_limit(sys.maxsize)


_REAL_TASK_PROBLEMS = ("r1_cifar10_resnet18", "r2_charlm_shakespeare", "r3_nanogpt_wikitext2")


def main() -> int:
    csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
    if not os.path.exists(csv_path):
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        print("error: empty CSV", file=sys.stderr)
        return 1
    header = rows[0]
    if "data_source" in header:
        print("[annotate] data_source already present; skipping")
        return 0
    # Insert data_source after l5_count.
    try:
        idx = header.index("l5_count")
    except ValueError:
        print("error: l5_count column not found in header", file=sys.stderr)
        return 1
    new_header = header[:idx + 1] + ["data_source"] + header[idx + 1:]
    new_rows = [new_header]
    n_borrowed = 0
    n_native = 0
    for row in rows[1:]:
        problem = row[0]
        optimizer = row[1]
        # Borrowed = R1/R2/R3 row with naked "racaso" optimizer name
        # (the sibling Liger sweep used the bare class name; RACASO's
        # own harness writes "racaso_hutchinson" or "racaso_gnb").
        if problem in _REAL_TASK_PROBLEMS and optimizer == "racaso":
            tag = "liger_bench_sweep"
            n_borrowed += 1
        else:
            tag = "racaso_bench_sweep"
            n_native += 1
        new_rows.append(row[:idx + 1] + [tag] + row[idx + 1:])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    print(f"[annotate] wrote {len(new_rows)-1} rows ({n_borrowed} borrowed, {n_native} native)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
