#!/usr/bin/env python3
"""Run a beam-search-only analog of Table 15 from Deplano et al. (2019).

Table 15 uses five equal-volume bins with different dimensions and item counts
160, 200, 400, and 800. The original catalogue/instances are not available in
this repository, so this runner creates one deterministic substitute instance
per item count. It runs V2 beam search ten times per instance (analogous to the
paper's repeatLimit=10), with a 300-second limit per run. This is an analog,
not a reproduction of Table 15: the paper compares two WFBF variants, whereas
this runner evaluates only V2 beam search.

One task is one (item count, repeat) pair. For SLURM arrays, use --print-grid
to inspect the grid and pass each array index with --id.

Examples:
    python3 main_slurm15.py --print-grid
    python3 main_slurm15.py --id 0
    python3 main_slurm15.py --all
    python3 main_slurm15.py --summarize
"""

import argparse
import importlib.util
import json
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_ROOT = ROOT / "MHKP" / "table15_analog"
RESULTS_DIR = ROOT / "results_table15_bs"
ITEM_COUNTS = [160, 200, 400, 800]
N_BINS = 5
REPEATS = 10
TIME_LIMIT = 300.0
INSTANCE_SEED = 20260917
DATA_SEED = 1
ALPHA = 0.8
LOAD_FACTOR = 3.0
STABILITY_RULE = "peraxis"

# Distinct integer dimensions with common volume 7,200,000. These are
# documented substitutes; the paper's five actual dimensions are unavailable.
BIN_SHAPES = [
    (300, 200, 120),
    (400, 180, 100),
    (600, 120, 100),
    (240, 150, 200),
    (360, 200, 100),
]

_MODULES = {}


def _load(name, filename):
    if name not in _MODULES:
        path = ROOT / filename
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def instance_path(n_items):
    return INSTANCE_ROOT / f"table15_n{n_items}.txt"


def ensure_instance(n_items):
    """Create and persist the deterministic substitute instance if absent."""
    path = instance_path(n_items)
    if path.is_file():
        return path

    import mhkp_instances as mi

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{INSTANCE_SEED}:table15-analog:{n_items}")
    bins = [(L, W, H, 1.0 / (L * W * H)) for L, W, H in BIN_SHAPES]
    items = mi.generate_items(n_items, BIN_SHAPES, rng)
    mi.write_instance(path, bins, items)
    return path


def build_grid():
    """Return all (item count, repeat) tasks in stable order."""
    return [(n_items, repeat)
            for n_items in ITEM_COUNTS
            for repeat in range(1, REPEATS + 1)]


def report_path(results_dir, n_items, repeat, time_limit):
    return results_dir / (
        f"T15-{n_items}items-i{repeat:02d}-v2-bs-t{time_limit:g}s.json")


def solve_one(n_items, repeat, time_limit, solver_seed=None):
    import mhkp_instances as mi

    BS2 = _load("mhkp_v2_bs_table15", "MHKP-V2-BS.py")
    path = ensure_instance(n_items)
    bins_raw, items_raw = mi.read_instance(path)
    inst = BS2._sa.load_instance(str(path))
    # Keep derived physical attributes fixed across repeats; vary only the
    # beam-search seed, as the repeat represents a randomized solver run.
    BS2.attach_v2_data(inst, bins_raw, items_raw, seed=DATA_SEED,
                       alpha=ALPHA, factor=LOAD_FACTOR)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    seed = repeat if solver_seed is None else solver_seed
    result = BS2.beam_search_v2(
        inst, time_limit=time_limit, seed=seed, rule=STABILITY_RULE,
        strict=True)
    wall_time = time.time() - t0
    placements = [dict(p) for p in result["placements"]]
    violations = BS2.verify_v2(inst, placements, rule=STABILITY_RULE)

    per_bin = []
    for bin_index, (L, W, H, _priority) in enumerate(bins_raw):
        bin_placements = [p for p in placements if p["bin"] == bin_index]
        filled_volume = sum(p["dx"] * p["dy"] * p["dz"]
                            for p in bin_placements)
        per_bin.append({
            "bin": bin_index,
            "dimensions": [L, W, H],
            "items_packed": len(bin_placements),
            "filling_rate": filled_volume / (L * W * H),
        })

    return {
        "table": 15,
        "experiment": "synthetic Table 15 analog; V2 beam search only",
        "instance_source": "generated substitute; not the paper's catalogue",
        "items": n_items,
        "bins": N_BINS,
        "instance_file": str(path.relative_to(ROOT)),
        "repeat": repeat,
        "solver": "v2-bs",
        "time_limit": time_limit,
        "seed": seed,
        "data_seed": DATA_SEED,
        "alpha": ALPHA,
        "load_factor": LOAD_FACTOR,
        "stability_rule": STABILITY_RULE,
        "bin_dimensions": [list(shape) for shape in BIN_SHAPES],
        "bin_volume": BIN_SHAPES[0][0] * BIN_SHAPES[0][1] * BIN_SHAPES[0][2],
        "objective": result["objective"],
        "wasted_space_objective": result["wasted"],
        "n_packed": result["n_packed"],
        "used_bins": len({p["bin"] for p in placements}),
        "runtime": result["runtime"],
        "wall_time": wall_time,
        "rollouts": result["rollouts"],
        "levels": result["levels"],
        "fell_back": False,
        "status": "DONE",
        "violations": len(violations),
        "violation_detail": violations[:10],
        "per_bin": per_bin,
        "placements": placements,
        "host": socket.gethostname(),
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_task(args, entry):
    n_items, repeat = entry
    out = report_path(args.results_dir, n_items, repeat, args.time_limit)
    if out.exists() and not args.force:
        print(f"skip (exists): {out.name}   [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = solve_one(n_items, repeat, args.time_limit,
                       solver_seed=args.seed)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)
    print(f"n={n_items:>3} repeat={repeat:02d} v2-bs "
          f"obj={record['objective']:.4f} used={record['used_bins']} "
          f"packed={record['n_packed']:>3} "
          f"{record['runtime']:>7.2f}s violations={record['violations']} "
          f"-> {out.name}")
    return 0


def summarize(results_dir):
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    records = []
    for path in sorted(results_dir.glob("T15-*-v2-bs-*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"unparsable report skipped: {path.name}", file=sys.stderr)
    if not records:
        print(f"no reports in {results_dir}", file=sys.stderr)
        return 1

    print("Table 15 analog — V2 beam search only; generated substitute instances")
    print(f"{'items':>5} {'runs':>4} {'obj mean':>10} {'obj best':>10} "
          f"{'used mean':>10} {'packed mean':>12} {'time mean':>10} "
          f"{'violations':>11}")
    for n_items in ITEM_COUNTS:
        group = [r for r in records
                 if r.get("items") == n_items and r.get("objective") is not None]
        if not group:
            continue
        objectives = [r["objective"] for r in group]
        print(f"{n_items:>5} {len(group):>4} "
              f"{sum(objectives) / len(group):>10.4f} "
              f"{min(objectives):>10.4f} "
              f"{sum(r['used_bins'] for r in group) / len(group):>10.2f} "
              f"{sum(r['n_packed'] for r in group) / len(group):>12.2f} "
              f"{sum(r['runtime'] for r in group) / len(group):>9.2f}s "
              f"{sum(r.get('violations', 0) for r in group):>11}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run a Table 15-inspired V2 beam-search experiment.")
    parser.add_argument("--print-grid", action="store_true",
                        help="print task IDs for a SLURM array")
    parser.add_argument("--id", type=int, help="run one zero-based task ID")
    parser.add_argument("--all", action="store_true",
                        help="run every item-count/repeat task")
    parser.add_argument("--items", type=int, choices=ITEM_COUNTS,
                        help="filter to one item count")
    parser.add_argument("--repeat", type=int, choices=range(1, REPEATS + 1),
                        help="filter to one repeat (1-10)")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help=f"seconds per run (default: {TIME_LIMIT:g})")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the beam-search seed for the selected run(s)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing reports")
    parser.add_argument("--summarize", action="store_true",
                        help="summarize JSON reports and exit")
    args = parser.parse_args()

    if args.summarize:
        return summarize(args.results_dir)

    grid = build_grid()
    if args.items is not None:
        grid = [entry for entry in grid if entry[0] == args.items]
    if args.repeat is not None:
        grid = [entry for entry in grid if entry[1] == args.repeat]

    if args.print_grid:
        for task_id, (n_items, repeat) in enumerate(grid):
            print(f"{task_id:>3}: {n_items} items, repeat {repeat:02d}")
        print(f"Total tasks: {len(grid)}")
        return 0

    if args.id is not None:
        if args.id < 0 or args.id >= len(grid):
            parser.error(f"--id must be between 0 and {len(grid) - 1}")
        entry = grid[args.id]
        return run_task(args, entry)

    if args.all:
        return sum(run_task(args, entry) for entry in grid)

    if args.items is not None or args.repeat is not None:
        return sum(run_task(args, entry) for entry in grid)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
