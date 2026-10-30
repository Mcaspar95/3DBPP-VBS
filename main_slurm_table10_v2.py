#!/usr/bin/env python3
"""Run and summarize the V2 MILP and beam search on Table 10's grid.

The solver and verification logic is shared with main_slurm_table10.py;
this module exposes only its V2 columns as a 150-task standalone array.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import main_slurm_table10 as table10


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_table10_v2"
ITEM_COUNTS = table10.ITEM_COUNTS
BIN_COUNTS = table10.BIN_COUNTS
INSTANCES_PER_CONFIG = table10.INSTANCES_PER_CONFIG
METHODS = table10.METHODS
COLUMNS = [f"v2-{method}" for method in METHODS]
METHOD_TIME_LIMITS = table10.METHOD_TIME_LIMITS


def build_grid():
    """Return 15 configurations x 5 instances x 2 V2 methods (150 tasks)."""
    return [(n, m, instance, f"v2-{method}")
            for m in BIN_COUNTS
            for n in ITEM_COUNTS
            for instance in range(1, INSTANCES_PER_CONFIG + 1)
            for method in METHODS]


def run_task(args, entry):
    """Run one task through the shared Table 10 V2 solver implementation."""
    return table10.run_task(args, entry)


def summarize(results_dir, csv_path=None):
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    records = {}
    for path in sorted(results_dir.glob("T10-*-v2-*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        if record.get("submodel") != "V2":
            continue
        records[(record["items"], record["bins"], record["instance"],
                 record["method"])] = record

    if not records:
        print(f"no V2 reports in {results_dir}", file=sys.stderr)
        return 1

    print("Table 10-style V2 results: means of objective (used bins) over "
          f"{INSTANCES_PER_CONFIG} instances")
    print(f"{'items':>5} {'bins':>4} | {'MILP objective (used bins)':>27} | "
          f"{'BS objective (used bins)':>25} | {'violations (MILP/BS)'}")
    print("-" * 94)

    rows = []
    missing = []
    for n_bins in BIN_COUNTS:
        for n_items in ITEM_COUNTS:
            values = {}
            violations = {}
            for method in METHODS:
                group = [records.get((n_items, n_bins, instance, method))
                         for instance in range(1, INSTANCES_PER_CONFIG + 1)]
                present = [record for record in group if record is not None]
                valid = [record for record in present
                         if record.get("objective") is not None]
                if len(present) < INSTANCES_PER_CONFIG:
                    missing.append(f"{n_items}x{n_bins}/{method} "
                                   f"({len(present)}/{INSTANCES_PER_CONFIG})")
                if valid:
                    objective = statistics.fmean(
                        record["objective"] for record in valid)
                    used_bins = statistics.fmean(
                        record["used_bins"] for record in valid)
                    values[method] = f"{objective:.4f}({used_bins:.1f})"
                else:
                    values[method] = "-"
                violations[method] = sum(
                    record.get("violations", 0) for record in present)

            rows.append({
                "items": n_items,
                "bins": n_bins,
                "milp": values["milp"],
                "bs": values["bs"],
                "milp_violations": violations["milp"],
                "bs_violations": violations["bs"],
            })
            print(f"{n_items:>5} {n_bins:>4} | {values['milp']:>27} | "
                  f"{values['bs']:>25} | "
                  f"{violations['milp']:>4}/{violations['bs']:<4}")

    if missing:
        print("\nMissing reports: " + ", ".join(missing))

    if csv_path:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {csv_path}")

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, help="zero-based Slurm array task ID")
    parser.add_argument("--all", action="store_true", help="run the full grid")
    parser.add_argument("--config", help="configuration, e.g. 70x2")
    parser.add_argument("--instance", type=int, help="instance number 1..5")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--time-limit", type=float,
                        help="override default limit (MILP 3600s, BS 900s)")
    parser.add_argument("--num_cpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=table10.SEED)
    parser.add_argument("--alpha", type=float, default=table10.ALPHA)
    parser.add_argument("--load-factor", type=float, default=table10.LOAD_FACTOR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--print-grid", action="store_true")
    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir, args.csv)

    grid = build_grid()
    if args.print_grid:
        print(f"{len(grid)} tasks: {len(ITEM_COUNTS)} item counts x "
              f"{len(BIN_COUNTS)} bin counts x {INSTANCES_PER_CONFIG} "
              f"instances x {len(METHODS)} V2 methods")
        for task_id, (n, m, instance, column) in enumerate(grid):
            method = column.split("-")[1]
            print(f"{task_id:3d} {n}x{m} i{instance:02d} "
                  f"V2-{method} limit="
                  f"{args.time_limit or METHOD_TIME_LIMITS[method]:g}s")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    selected = grid
    if args.config:
        try:
            n_text, m_text = args.config.lower().split("x")
            n, m = int(n_text), int(m_text)
        except ValueError:
            parser.error("--config must look like 70x2")
        selected = [entry for entry in selected
                    if entry[0] == n and entry[1] == m]
        if not selected:
            parser.error(f"unknown configuration {args.config!r}")
    if args.instance is not None:
        selected = [entry for entry in selected if entry[2] == args.instance]
        if not selected:
            parser.error("--instance must be between 1 and 5")
    if args.method:
        selected = [entry for entry in selected
                    if entry[3] == f"v2-{args.method}"]

    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id must be in 0..{len(grid) - 1}")
        entry = grid[args.id]
        if entry not in selected:
            print(f"skip: array task {args.id} excluded by filters")
            return 0
        return run_task(args, entry)
    if args.all:
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc
    if args.config and args.instance is not None and args.method:
        return run_task(args, selected[0])
    parser.error("use --id, --all, or --config ITEMSxBINS "
                 "--instance K --method METHOD")


if __name__ == "__main__":
    sys.exit(main())
