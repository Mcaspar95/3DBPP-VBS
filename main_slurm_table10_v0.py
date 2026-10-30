#!/usr/bin/env python3
"""Run the V0 MILP and beam search on Deplano et al. (2019), Table 10's grid.

The paper's grid is items {70, 100, 130, 160, 200} x bins {2, 3, 5}.
One array task runs one method on one configuration. Each JSON report stores
objective (1), the number of bins used by the returned packing, runtime,
solver status and independent V0 feasibility-check results. --summarize prints
MILP and BS as ``objective(used_bins)`` side by side.

The paper's original instance catalogue is not included in this repository.
By default this uses instance01 from the generated t10_n*_b* groups; results
are therefore experiments on substitute instances, not a numerical
reproduction of the paper's original rows. Point --instance-root at a
compatible tree if the original instances become available.
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
RESULTS_DIR = ROOT / "results_table10_v0"
ITEM_COUNTS = [70, 100, 130, 160, 200]
BIN_COUNTS = [2, 3, 5]
INSTANCE_NUMBER = 1
METHODS = ["milp", "bs"]
TIME_LIMITS = {"milp": 3600.0, "bs": 900.0}
STABILITY_RULE = "peraxis"  # same interpretation encoded by the V0 MILP
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


def instance_path(instance_root, n_items, n_bins):
    group = f"t10_n{n_items}_b{n_bins}"
    return instance_root / group / f"{group}_instance{INSTANCE_NUMBER:02d}.txt"


def build_grid():
    """Return the 15 configurations x 2 methods (30 array tasks)."""
    return [(n, m, method)
            for m in BIN_COUNTS
            for n in ITEM_COUNTS
            for method in METHODS]


def report_path(results_dir, n_items, n_bins, method, time_limit):
    budget = f"t{time_limit:g}s"
    return results_dir / (
        f"T10-V0-{config_name(n_items, n_bins)}-i{INSTANCE_NUMBER:02d}-"
        f"{method}-{budget}.json")


def solve_one(n_items, n_bins, method, time_limit, instance_root,
              seed=1, threads=1):
    import mhkp_instances as mi

    path = instance_path(instance_root, n_items, n_bins)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance: {path}\nGenerate the local substitute corpus "
            "with mhkp_instances.py or supply --instance-root.")

    bins, items = mi.read_instance(path)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    if method == "milp":
        milp = _load("mhkp_v0_milp_table10", "mhkp_v0_milp.py")
        res = milp.solve(bins, items, time_limit=time_limit, threads=threads)
        placements = res["placements"]
        objective = res["objective"]
        runtime = res["runtime"]
        status = res["status_name"]
        packed = res["n_packed"]
        # The MILP's axes differ from the heuristic verifier's axes.
        sa = _load("mhkp_v0_sa_table10", "MHKP-V0-SA.py")
        sa.STABILITY_RULE = STABILITY_RULE
        inst = sa.load_instance(str(path))
        checked_placements = milp.solution_to_sa_frame(placements)
        violations = sa.verify(inst, checked_placements, stability=True,
                               rule=STABILITY_RULE)
        details = {
            "bound": res["bound"], "gap": res["gap"],
            "build_time": res["build_time"], "n_vars": res["n_vars"],
            "n_constrs": res["n_constrs"],
        }
    elif method == "bs":
        bs = _load("mhkp_v0_bs_table10", "MHKP-V0-BS.py")
        bs._sa.STABILITY_RULE = STABILITY_RULE
        inst = bs.load_instance(str(path))
        res = bs.beam_search(inst, time_limit=time_limit, seed=seed,
                             rule=STABILITY_RULE)
        placements = res["placements"]
        objective = res["wasted"]
        runtime = res["runtime"]
        status = "DONE"
        packed = res["n_packed"]
        violations = bs.verify(inst, placements, stability=True,
                               rule=STABILITY_RULE)
        details = {"rollouts": res["rollouts"], "levels": res["levels"]}
    else:
        raise ValueError(f"unknown method {method!r}")

    used_bins = len({p["bin"] for p in placements}) if placements else 0
    return {
        "table": 10,
        "submodel": "V0",
        "method": method,
        "config": config_name(n_items, n_bins),
        "items": n_items,
        "bins_available": n_bins,
        "instance": INSTANCE_NUMBER,
        "instance_file": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT) else str(path),
        "time_limit": time_limit,
        "seed": seed,
        "stability_rule": STABILITY_RULE,
        "host": socket.gethostname(),
        "started": started,
        "objective": objective,
        "used_bins": used_bins,
        "n_packed": packed,
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
    n_items, n_bins, method = entry
    limit = args.time_limit if args.time_limit is not None else TIME_LIMITS[method]
    out = report_path(args.results_dir, n_items, n_bins, method, limit)
    if out.exists() and not args.force:
        print(f"skip (exists): {out.name} [--force to redo]")
        return 0
    args.results_dir.mkdir(parents=True, exist_ok=True)
    record = solve_one(n_items, n_bins, method, limit, args.instance_root,
                       seed=args.seed, threads=args.num_cpu)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)
    obj = "-" if record["objective"] is None else f"{record['objective']:.4f}"
    print(f"{record['config']:>7} V0-{method:<4} {obj}"
          f"({record['used_bins']}) {record['runtime']:.1f}s "
          f"{record['status']} viol={record['violations']} -> {out.name}")
    return 0


def summarize(results_dir):
    reports = {}
    for path in results_dir.glob("T10-V0-*.json"):
        try:
            r = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        reports[(r["items"], r["bins_available"], r["method"])] = r

    print("Table 10-style V0 results: objective (used bins)")
    print(f"{'items':>5} {'bins':>4} | {'MILP objective (used bins)':>27} | "
          f"{'BS objective (used bins)':>25} | {'status / violations'}")
    print("-" * 100)
    missing = []
    for m in BIN_COUNTS:
        for n in ITEM_COUNTS:
            row = [reports.get((n, m, method)) for method in METHODS]
            cells = []
            for method, r in zip(METHODS, row):
                if r is None or r.get("objective") is None:
                    cells.append("-")
                    missing.append(f"{n}x{m} {method}")
                else:
                    cells.append(f"{r['objective']:.4f}({r['used_bins']})")
            checks = ", ".join(
                f"{method}={r['status']}/v{r['violations']}"
                for method, r in zip(METHODS, row) if r is not None)
            print(f"{n:>5} {m:>4} | {cells[0]:>27} | {cells[1]:>25} | {checks}")
    if missing:
        print("\nMissing reports/results: " + ", ".join(missing))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, help="zero-based Slurm array task ID")
    parser.add_argument("--all", action="store_true", help="run the full grid")
    parser.add_argument("--config", help="one configuration, e.g. 70x2")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--time-limit", type=float,
                        help="override the method default (MILP 3600s, BS 900s)")
    parser.add_argument("--num_cpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instance-root", type=Path, default=INSTANCE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--print-grid", action="store_true")
    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir)

    grid = build_grid()
    if args.print_grid:
        print(f"{len(grid)} tasks: {len(ITEM_COUNTS)} item counts x "
              f"{len(BIN_COUNTS)} bin counts x {len(METHODS)} methods")
        for task_id, (n, m, method) in enumerate(grid):
            print(f"{task_id:3d} {n}x{m} V0-{method} "
                  f"limit={args.time_limit or TIME_LIMITS[method]:g}s")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    selected = grid
    if args.config:
        try:
            n_s, m_s = args.config.lower().split("x")
            n, m = int(n_s), int(m_s)
        except ValueError:
            parser.error("--config must look like 70x2")
        selected = [e for e in selected if e[0] == n and e[1] == m]
        if not selected:
            parser.error(f"unknown configuration: {args.config}")
    if args.method:
        selected = [e for e in selected if e[2] == args.method]

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
    if args.config and args.method:
        return run_task(args, selected[0])
    parser.error("use --id, --all, or --config ITEMSxBINS --method METHOD")


if __name__ == "__main__":
    sys.exit(main())
