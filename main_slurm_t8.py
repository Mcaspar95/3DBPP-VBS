#!/usr/bin/env python3
"""SLURM array dispatcher for V2 on the single-bin n=18..200 sweep.

Runs WFBF, V2 beam search, and the V2 MILP on the existing t8_n* and
t12_n* instance groups. WFBF defaults to uncapped corner-point mode, BS
receives 60 seconds, and the MILP receives 3600 seconds. All results are
verified against the full V2 constraint set. One array task solves one
(size, instance, solver) triple.

Examples:
    python3 main_slurm_t8.py --print-grid
    python3 main_slurm_t8.py --id 0
    python3 main_slurm_t8.py --items 50 --instance 1 --solver bs
    python3 main_slurm_t8.py --all
    python3 main_slurm_t8.py --summarize

WFBF uses corner-point candidate positions by default. This is a deterministic
constructive pass and normally completes in milliseconds. Use
``--wfbf-mode grid`` only when explicitly requesting the literal grid variant.
"""

import argparse
import importlib.util
import json
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_ROOT = ROOT / "MHKP"
RESULTS_DIR = ROOT / "results_table8_V2"
ALPHA = 0.8  # pyramid base fraction, same as main_slurm.py (paper's suggested default)
LOAD_FACTOR = 3.0  # Lambda_i = 3 * Omega_i, the paper's equation (18)
STABILITY_RULE = "peraxis"

TABLE8_SIZES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
TABLE12_SIZES = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
ITEM_COUNTS = TABLE8_SIZES + TABLE12_SIZES
INSTANCES_PER_SIZE = 10
SOLVERS = ["wfbf", "bs", "milp"]
SOLVER_TIME_LIMITS = {"wfbf": None, "bs": 60.0, "milp": 3600.0}
SEED = 1
STABILITY_RULE = "peraxis"

_MODULES = {}


def _load(name, filename):
    """Import a project solver by path and cache it."""
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


def group_name(n_items):
    return f"t8_n{n_items}" if n_items in TABLE8_SIZES else f"t12_n{n_items}"


def instance_path(n_items, instance):
    group = group_name(n_items)
    return INSTANCE_ROOT / group / f"{group}_instance{instance:02d}.txt"


def build_grid():
    """Return all size/instance/solver entries, solver varying fastest."""
    return [(n, k, solver)
            for n in ITEM_COUNTS
            for k in range(1, INSTANCES_PER_SIZE + 1)
            for solver in SOLVERS]


def report_path(results_dir, n_items, instance, solver, time_limit,
                wfbf_mode="corner", grid_step=None):
    budget = "uncapped" if time_limit is None else f"t{time_limit:g}s"
    mode = f"-{wfbf_mode}" if solver == "wfbf" else ""
    step = f"-step{grid_step}" if solver == "wfbf" and grid_step is not None else ""
    return results_dir / (f"T8V2-n{n_items}-i{instance:02d}-{solver}"
                          f"{mode}{step}-{budget}.json")


def solve_one(n_items, instance, solver, time_limit, seed=SEED,
              wfbf_mode="corner", grid_step=None, threads=1, alpha=ALPHA,
              load_factor=LOAD_FACTOR):
    """Run one solver and return a JSON-serializable V2 experiment record."""
    import mhkp_instances as mi

    path = instance_path(n_items, instance)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance {path}; generate the t8/t12 instance groups first")

    bins, items = mi.read_instance(path)
    BS2 = _load("mhkp_v2_bs", "MHKP-V2-BS.py")
    V2 = _load("mhkp_v2_milp", "mhkp_v2_milp.py")
    sa = BS2._sa
    sa.STABILITY_RULE = STABILITY_RULE
    inst = sa.load_instance(str(path))
    BS2.attach_v2_data(inst, bins, items, seed=seed, alpha=alpha,
                       factor=load_factor)

    record = {
        "table": 8 if n_items in TABLE8_SIZES else 12,
        "submodel": "V2",
        "alpha": alpha,
        "load_factor": load_factor,
        "items": n_items,
        "bins": 1,
        "instance": instance,
        "instance_file": str(path.relative_to(ROOT)),
        "solver": solver,
        "wfbf_mode": wfbf_mode if solver == "wfbf" else None,
        "time_limit": time_limit,
        "seed": seed,
        "stability_rule": STABILITY_RULE,
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    started = time.time()
    placements = []

    if solver == "wfbf":
        import wfbf
        # WFBF corner mode is one deterministic constructive pass; it does
        # not have an anytime budget. Keep its actual wall-clock duration.
        result = wfbf.wfbf(bins, items, mode=wfbf_mode,
                           grid_step=grid_step, seed=seed)
        status = "DONE"
        placements = [dict(p) for p in result["placements"]]
        for placement in placements:
            placement.setdefault("reflected", 0)
        record.update({
            "objective": result["objective"],
            "n_packed": len(placements),
            "runtime": time.time() - started,
            "status": status,
            "grid_step": grid_step,
            "beta": wfbf._grid_beta(items),
        })

    elif solver == "bs":
        result = BS2.beam_search_v2_auto(inst, time_limit=time_limit,
                                         seed=seed, rule=STABILITY_RULE)
        placements = result["placements"]
        record.update({
            "objective": result["wasted"],
            "n_packed": result["n_packed"],
            "runtime": result["runtime"],
            "rollouts": result["rollouts"],
            "levels": result["levels"],
            "fell_back": bool(result.get("fell_back")),
            "status": "DONE",
        })

    elif solver == "milp":
        result = V2.solve(bins, items, time_limit=time_limit, threads=threads,
                          seed=seed, alpha=alpha, factor=load_factor)
        placements = V2.V0.solution_to_sa_frame(result["placements"])
        for placement, milp_placement in zip(placements, result["placements"]):
            placement["reflected"] = milp_placement.get("reflected", 0)
        record.update({
            "objective": result["objective"],
            "bound": result["bound"],
            "gap": result["gap"],
            "n_packed": result["n_packed"],
            "runtime": result["runtime"],
            "build_time": result["build_time"],
            "status": result["status_name"],
            "n_vars": result["n_vars"],
            "n_constrs": result["n_constrs"],
        })

    else:
        raise ValueError(f"unknown solver {solver!r}")

    violations = BS2.verify_v2(inst, placements, rule=STABILITY_RULE)
    record["violations"] = len(violations)
    record["violation_detail"] = violations[:10]
    record["feasible_V2"] = (None if record.get("objective") is None
                             else not violations)
    record["wall_time"] = time.time() - started
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return record


def run_task(args, entry):
    n_items, instance, solver = entry
    # Corner WFBF is a deterministic pass with no solver time-limit setting.
    # The global override applies to the timed BS/MILP solvers only.
    time_limit = (None if solver == "wfbf" else
                  args.time_limit if args.time_limit is not None else
                  SOLVER_TIME_LIMITS[solver])
    out = report_path(args.results_dir, n_items, instance, solver, time_limit,
                      args.wfbf_mode, args.grid_step)
    if out.exists() and not args.force:
        print(f"skip (exists): {out.name}   [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = solve_one(n_items, instance, solver, time_limit, seed=args.seed,
                       wfbf_mode=args.wfbf_mode, grid_step=args.grid_step,
                       threads=args.num_cpu, alpha=args.alpha,
                       load_factor=args.load_factor)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)

    objective = record["objective"]
    obj = "-" if objective is None else f"{objective:.4f}"
    print(f"n{n_items:<4} i{instance:02d} {solver:<5} obj {obj:>8}  "
          f"packed {record['n_packed']:>3}  {record['runtime']:>8.2f}s  "
          f"viol {record['violations']}  -> {out.name}")
    return 0


def summarize(results_dir, csv_path=None):
    """Print per-size averages from the task JSON files."""
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    reports = []
    for path in sorted(results_dir.glob("T8V2-*.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"unparsable report skipped: {path.name}", file=sys.stderr)
    if not reports:
        print(f"no reports in {results_dir}", file=sys.stderr)
        return 1

    by = {}
    for record in reports:
        key = (record["items"], record["solver"], record.get("wfbf_mode"))
        by.setdefault(key, []).append(record)

    columns = [("wfbf", "WFBF"), ("bs", "BS"), ("milp", "MILP")]
    rows = []
    print(f"V2 results: {len(reports)} reports from {results_dir.name}")
    print(f"{'n':>5} | {'method':<7} {'objective':>12} {'time (s)':>10} "
          f"{'gap %':>9} {'viol':>6} {'runs':>5}")
    print("-" * 68)
    for n_items in ITEM_COUNTS:
        for solver, label in columns:
            mode = None
            if solver == "wfbf":
                # Prefer corner-mode reports; grid is only included if no
                # corner result exists for this size.
                variants = [r for r in reports
                            if r["items"] == n_items and r["solver"] == solver]
                available_modes = {r.get("wfbf_mode") for r in variants}
                mode = "corner" if "corner" in available_modes else (
                    "grid" if "grid" in available_modes else None)
            records = by.get((n_items, solver, mode), [])
            values = [r for r in records if r.get("objective") is not None]
            avg = lambda field: (statistics.fmean(r[field] for r in values
                                                   if r.get(field) is not None)
                                 if any(r.get(field) is not None for r in values)
                                 else float("nan"))
            gap = avg("gap")
            row = {"items": n_items, "solver": solver,
                   "wfbf_mode": mode,
                   "objective": avg("objective"), "runtime": avg("runtime"),
                   "gap": gap,
                   "violations": sum(r.get("violations", 0) for r in records),
                   "runs": len(records)}
            rows.append(row)
            f = lambda x, w, p=4: (f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}")
            label += f"-{mode}" if solver == "wfbf" and mode else ""
            gap_text = "-" if gap != gap else f"{gap * 100:.2f}%"
            print(f"{n_items:>5} | {label:<7} {f(row['objective'], 12)} "
                  f"{f(row['runtime'], 10, 2)} {gap_text:>9} "
                  f"{row['violations']:>6} {row['runs']:>5}")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwritten: {csv_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for V2 on the n=18..200 single-bin sweep.")
    parser.add_argument("--id", type=int, default=None,
                        help="zero-based SLURM_ARRAY_TASK_ID")
    parser.add_argument("--items", type=int, default=None, metavar="N")
    parser.add_argument("--instance", type=int, default=None, metavar="K")
    parser.add_argument("--solver", choices=SOLVERS, default=None)
    parser.add_argument("--wfbf-mode", choices=("corner", "grid"), default="corner",
                        help="WFBF placement mode (default: corner)")
    parser.add_argument("--grid-step", type=int, default=None,
                        help="optional WFBF grid spacing; default is beta")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="override BS/MILP defaults (60s/3600s); corner WFBF is uncapped")
    parser.add_argument("--num_cpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"pyramid base fraction (default {ALPHA})")
    parser.add_argument("--load-factor", type=float, default=LOAD_FACTOR,
                        help=f"load-bearing multiplier (default {LOAD_FACTOR})")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="run all selected grid entries sequentially")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--csv", default=None, metavar="PATH")
    parser.add_argument("--print-grid", action="store_true")
    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir, args.csv)

    grid = build_grid()
    if args.print_grid:
        print(f"{len(grid)} tasks: {len(ITEM_COUNTS)} sizes x "
              f"{INSTANCES_PER_SIZE} instances x {len(SOLVERS)} methods")
        print("time limits: " + ", ".join(
            f"{solver}={'uncapped' if limit is None else f'{limit:g}s'}"
            for solver, limit in SOLVER_TIME_LIMITS.items()))
        print(f"WFBF mode: {args.wfbf_mode}")
        print(f"\\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    selected = grid
    if args.items is not None:
        selected = [entry for entry in selected if entry[0] == args.items]
        if not selected:
            parser.error(f"no such item count: {args.items}")
    if args.instance is not None:
        selected = [entry for entry in selected if entry[1] == args.instance]
        if not selected:
            parser.error(f"no instance {args.instance} in the selection")
    if args.solver:
        selected = [entry for entry in selected if entry[2] == args.solver]

    if args.all:
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc
    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id must be in 0..{len(grid) - 1}")
        entry = grid[args.id]
        if args.solver and entry[2] != args.solver:
            print(f"skip (solver): id {args.id} is {entry[2]}, not {args.solver}")
            return 0
        return run_task(args, entry)
    if args.items is not None and args.instance is not None and args.solver:
        return run_task(args, selected[0])
    parser.error("provide --id, --all, or --items with --instance and --solver")


if __name__ == "__main__":
    sys.exit(main())
