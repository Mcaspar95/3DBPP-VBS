#!/usr/bin/env python3
"""Slurm runner for V1 MILP and beam search on Table 10's multi-bin grid.

The grid is items {70, 100, 130, 160, 200} x bins {2, 3, 5}, with five
instances per configuration and both V1 methods (150 array tasks total).
Reports store objective, actual used-bin count, runtime, solver status and
independent V1 feasibility checks. The summary formats results as
objective(used bins), averaging objective and used-bin count over the five
instances for each configuration/method.

By default, instances are read from MHKP/t10_n{items}_b{bins}. The paper's
original catalogue is not included in this repository; locally generated
t10 instances are substitutes, not the paper's original data.
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
RESULTS_DIR = ROOT / "results_table10_v1"
ITEM_COUNTS = [70, 100, 130, 160, 200]
BIN_COUNTS = [2, 3, 5]
INSTANCES_PER_CONFIG = 5
METHODS = ["milp", "bs"]
TIME_LIMITS = {"milp": 3600.0, "bs": 900.0}
SEED = 1
ALPHA = 0.8
STABILITY_RULE = "peraxis"
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


def config_name(n_items, n_bins):
    return f"{n_items}x{n_bins}"


def instance_path(instance_root, n_items, n_bins, instance):
    group = f"t10_n{n_items}_b{n_bins}"
    return instance_root / group / f"{group}_instance{instance:02d}.txt"


def build_grid():
    """Return 15 configurations x 5 instances x 2 methods (150 tasks)."""
    return [(n, m, k, method)
            for m in BIN_COUNTS
            for n in ITEM_COUNTS
            for k in range(1, INSTANCES_PER_CONFIG + 1)
            for method in METHODS]


def report_path(results_dir, n_items, n_bins, instance, method, time_limit):
    return results_dir / (
        f"T10-V1-{config_name(n_items, n_bins)}-i{instance:02d}-"
        f"{method}-t{time_limit:g}s.json")


def solve_one(n_items, n_bins, instance, method, time_limit,
              instance_root=INSTANCE_ROOT, seed=SEED, alpha=ALPHA,
              threads=1):
    import mhkp_instances as mi

    path = instance_path(instance_root, n_items, n_bins, instance)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance: {path}\nGenerate the t10 instance groups "
            "with mhkp_instances.py or supply --instance-root.")

    bins, items = mi.read_instance(path)
    V1BS = _load("mhkp_v1_bs_table10_runner", "MHKP-V1-BS.py")
    V1BS._sa.STABILITY_RULE = STABILITY_RULE
    inst = V1BS._sa.load_instance(str(path))
    V1BS.attach_v1_data(inst, bins, items, seed=seed, alpha=alpha)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    if method == "milp":
        V1 = _load("mhkp_v1_milp_table10_runner", "mhkp_v1_milp.py")
        res = V1.solve(bins, items, time_limit=time_limit, threads=threads,
                       seed=seed, alpha=alpha)
        placements = res["placements"]
        objective = res["objective"]
        runtime = res["runtime"]
        status = res["status_name"]
        n_packed = res["n_packed"]
        violations = V1BS.verify_v1(inst, placements, rule=STABILITY_RULE)
        details = {
            "bound": res["bound"], "gap": res["gap"],
            "build_time": res["build_time"], "n_vars": res["n_vars"],
            "n_constrs": res["n_constrs"],
        }
    elif method == "bs":
        res = V1BS.beam_search_v1_auto(inst, time_limit=time_limit,
                                       seed=seed, rule=STABILITY_RULE)
        placements = res["placements"]
        objective = res["wasted"]
        runtime = res["runtime"]
        status = "DONE"
        n_packed = res["n_packed"]
        violations = V1BS.verify_v1(inst, placements, rule=STABILITY_RULE)
        details = {
            "rollouts": res["rollouts"], "levels": res["levels"],
            "fell_back": bool(res.get("fell_back")),
        }
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "table": 10,
        "submodel": "V1",
        "method": method,
        "config": config_name(n_items, n_bins),
        "items": n_items,
        "bins": n_bins,
        "instance": instance,
        "instance_file": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT) else str(path),
        "time_limit": time_limit,
        "seed": seed,
        "alpha": alpha,
        "stability_rule": STABILITY_RULE,
        "host": socket.gethostname(),
        "started": started,
        "objective": objective,
        "used_bins": len({p["bin"] for p in placements}) if placements else 0,
        "n_packed": n_packed,
        "runtime": runtime,
        "status": status,
        "violations": len(violations),
        "violation_detail": violations[:20],
        "feasible": bool(objective is not None and not violations),
        "wall_time": time.time() - t0,
        "solver_details": details,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_task(args, entry):
    n_items, n_bins, instance, method = entry
    limit = args.time_limit if args.time_limit is not None else TIME_LIMITS[method]
    out = report_path(args.results_dir, n_items, n_bins, instance, method, limit)
    if out.exists() and not args.force:
        print(f"skip (exists): {out.name} [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = solve_one(n_items, n_bins, instance, method, limit,
                       instance_root=args.instance_root, seed=args.seed,
                       alpha=args.alpha, threads=args.num_cpu)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)

    objective = record["objective"]
    objective_s = "-" if objective is None else f"{objective:.4f}"
    print(f"{record['config']:>7} i{instance:02d} V1-{method:<4} "
          f"{objective_s}({record['used_bins']}) "
          f"{record['runtime']:.1f}s {record['status']} "
          f"viol={record['violations']} -> {out.name}")
    return 0


def summarize(results_dir, csv_path=None):
    records = {}
    for path in sorted(results_dir.glob("T10-V1-*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        records[(record["items"], record["bins"], record["instance"],
                 record["method"])] = record

    if not records:
        print(f"no V1 reports in {results_dir}", file=sys.stderr)
        return 1

    print(f"Table 10-style V1 results: means of objective (used bins) over "
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
                group = [records.get((n_items, n_bins, k, method))
                         for k in range(1, INSTANCES_PER_CONFIG + 1)]
                present = [r for r in group if r is not None]
                valid_obj = [r for r in present if r.get("objective") is not None]
                if len(present) < INSTANCES_PER_CONFIG:
                    missing.append(f"{n_items}x{n_bins}/{method} "
                                   f"({len(present)}/{INSTANCES_PER_CONFIG})")
                if valid_obj:
                    obj_mean = statistics.fmean(r["objective"] for r in valid_obj)
                    used_mean = statistics.fmean(r["used_bins"] for r in valid_obj)
                    values[method] = f"{obj_mean:.4f}({used_mean:.1f})"
                else:
                    values[method] = "-"
                violations[method] = sum(r.get("violations", 0) for r in present)
            rows.append({"items": n_items, "bins": n_bins,
                         "milp": values["milp"], "bs": values["bs"],
                         "milp_violations": violations["milp"],
                         "bs_violations": violations["bs"]})
            print(f"{n_items:>5} {n_bins:>4} | {values['milp']:>27} | "
                  f"{values['bs']:>25} | {violations['milp']:>4}/{violations['bs']:<4}")

    if missing:
        print("\nMissing reports: " + ", ".join(missing))

    if csv_path:
        import csv
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {csv_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, help="zero-based Slurm array task ID")
    parser.add_argument("--all", action="store_true", help="run selected grid")
    parser.add_argument("--config", help="configuration, e.g. 70x2")
    parser.add_argument("--instance", type=int, help="instance index 1..5")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--time-limit", type=float,
                        help="override default limit (MILP 3600s, BS 900s)")
    parser.add_argument("--num_cpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--instance-root", type=Path, default=INSTANCE_ROOT)
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
              f"instances x {len(METHODS)} methods")
        for task_id, entry in enumerate(grid):
            n, m, k, method = entry
            print(f"{task_id:3d} {n}x{m} i{k:02d} V1-{method} "
                  f"limit={args.time_limit or TIME_LIMITS[method]:g}s")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    selected = grid
    if args.config:
        try:
            n_text, m_text = args.config.lower().split("x")
            n, m = int(n_text), int(m_text)
        except ValueError:
            parser.error("--config must look like 70x2")
        selected = [e for e in selected if e[0] == n and e[1] == m]
        if not selected:
            parser.error(f"unknown configuration {args.config!r}")
    if args.instance is not None:
        selected = [e for e in selected if e[2] == args.instance]
        if not selected:
            parser.error("--instance must be between 1 and 5")
    if args.method:
        selected = [e for e in selected if e[3] == args.method]

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
    parser.error("use --id, --all, or --config ITEMSxBINS --instance K --method METHOD")


if __name__ == "__main__":
    sys.exit(main())
