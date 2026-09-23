#!/usr/bin/env python3
"""SLURM array-job dispatcher for the Tables 11+12 experiment: submodel V2 on
the single-bin size sweep of Deplano et al. (2019).

Tables 11 and 12 compare their heuristic (WFBF) against the V2 MILP on ONE
bin, sweeping the item count:

    Table 11:  n = 18, 19, ..., 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90
               (23 sizes - the same sweep as Table 8/t8_n*)
    Table 12:  n = 100, 110, 120, ..., 200  (11 sizes, step 10)

34 sizes in total. We generate ten instances per size rather than their one,
so a stochastic solver can be averaged, and run THREE columns over the same
instances, in the same layout as table_results_table13_v2.tex:

    milp        the V2 MILP (mhkp_v2_milp.py), Gurobi, 3600 s
    wfbf-grid   WFBF under the literal reading of Algorithm 1 line 15 -
                every position on the beta-grid - which is the reading
                whose RUNTIME is comparable to the paper's own WFBF
                column. Capped at 900 s.
    bs          the V2 beam search (MHKP-V2-BS.py), 900 s

wfbf-corner (candidates restricted to corner points) is deliberately NOT
run here, for the same reason it was dropped from the Table 13 table: its
millisecond runtime is not comparable to the paper's, and
table_results_wfbf_grid_vs_corner.tex already covers that comparison
elsewhere.

One array task solves one (n_items, instance, solver) triple, so the grid is
34 x 10 x 3 = 1020 jobs:

    #SBATCH --array=0-1019

Every task writes ONE json report into --results-dir, named so that nothing
collides across solvers, instances or time limits. --summarize then reads
those reports back and prints the Table 11/12 layout, split at n=90/100 to
match the paper's own table boundary. Nothing is aggregated in memory across
tasks, because no task ever sees more than its own triple.

wfbf.wfbf() has no budget of its own, so wfbf-grid is capped from outside
with SIGALRM, exactly as in main_slurm.py (the Table 13 dispatcher); a
timed-out task records status TIMEOUT and no objective.

Examples:
    python3 main_slurm_t11t12.py --print-grid
    python3 main_slurm_t11t12.py --id 0
    python3 main_slurm_t11t12.py --items 70 --instance 1 --solver bs
    python3 main_slurm_t11t12.py --all --solver wfbf-grid --grid-step 5
    python3 main_slurm_t11t12.py --summarize
    python3 main_slurm_t11t12.py --summarize --csv table11_12.csv
"""

import argparse
import contextlib
import importlib.util
import json
import signal
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INSTANCE_ROOT = ROOT / "MHKP"
RESULTS_DIR = ROOT / "results_table11_12_V2"

# The two sweeps, kept apart because they name different instance groups
# (t8_n* for Table 11, t12_n* for Table 12 - see mhkp_instances.py) and
# because the paper reports them as two tables split at n=90/100.
TABLE11_SIZES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
TABLE12_SIZES = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

ITEM_COUNTS = TABLE11_SIZES + TABLE12_SIZES

# Instances per size. The paper runs one; ten lets a stochastic solver be
# averaged without pretending to a precision their design does not claim.
INSTANCES_PER_SIZE = 10

SOLVERS = ["milp", "wfbf-grid", "bs"]

# Per-solver defaults, in seconds. The MILP needs the paper's own 3600 s to
# be comparable with their column. The beam search converges long before
# 900 s on these single-bin instances during development, so that budget is
# generous rather than binding. wfbf-grid enumerates the whole bin per item
# and needs an external cap (see time_limited); 900 s keeps a full run of
# 1020 tasks tractable, matching the cap used for Table 13.
SOLVER_TIME_LIMITS = {
    "milp": 3600.0,
    "wfbf-grid": 900.0,
    "bs": 900.0,
}

# Grid spacing for wfbf-grid. None means wfbf.py's own default, the
# instance's beta of Definition 4, i.e. the literal Algorithm 1. A coarser
# step is a purely computational concession and is recorded in the report,
# because it changes which positions the heuristic can even see.
GRID_STEP = None

# V2 data derivation (mhkp_v2_milp.derive_v2_data). Fixed here so that every
# solver and every task sees IDENTICAL weights, centres of mass and load
# limits for a given instance - without that the three columns would not be
# solving the same problem. Same defaults as main_slurm.py (Table 13), so
# the two experiments are directly comparable.
SEED = 1
ALPHA = 0.8            # pyramid base fraction, the paper's suggested default
LOAD_FACTOR = 3.0      # Lambda_i = 3 * Omega_i, their equation (18)
STABILITY_RULE = "peraxis"

_MODULES = {}


def _load(name, filename):
    """Import one of the solver modules by path and cache it.

    The solver filenames are not importable module names (they start with a
    digit and contain dashes), hence the explicit spec loading.
    """
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


class _Timeout(Exception):
    """Raised inside a solver that has run past its budget."""


@contextlib.contextmanager
def time_limited(seconds):
    """Interrupt the enclosed call after `seconds`, or don't cap it at all.

    wfbf.wfbf() takes no time limit of its own, so wfbf-grid - which can
    enumerate millions of positions per item - is capped from outside with a
    real-time alarm. A timed-out task records status TIMEOUT and no
    objective, which --summarize then shows as a gap rather than silently
    averaging over whichever instances happened to finish.

    SIGALRM is POSIX only. On Windows there is no equivalent that can
    interrupt a CPU-bound loop, so the call runs uncapped and the report
    says so via "time_limit_enforced".

    Yields True when the cap is actually armed, False otherwise.
    """
    if seconds is None or not hasattr(signal, "SIGALRM"):
        yield False
        return

    def fire(signum, frame):
        raise _Timeout(f"exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# =========================================================================
# Grid
# =========================================================================
def group_name(n_items):
    """t8_n* for Table 11's sweep, t12_n* for Table 12's - see
    mhkp_instances.py."""
    return f"t8_n{n_items}" if n_items in TABLE11_SIZES else f"t12_n{n_items}"


def table_of(n_items):
    return 11 if n_items in TABLE11_SIZES else 12


def instance_path(n_items, k):
    """Path of instance k (1-based) of one size."""
    group = group_name(n_items)
    width = 2 if n_items in TABLE11_SIZES else 2
    return INSTANCE_ROOT / group / f"{group}_instance{k:0{width}d}.txt"


def build_grid():
    """Every (n_items, instance, solver) triple, in a fixed order.

    Solver varies fastest, then instance, then size, so a partial array
    still covers whole sizes rather than a ragged edge.
    """
    grid = []
    for n_items in ITEM_COUNTS:
        for k in range(1, INSTANCES_PER_SIZE + 1):
            for solver in SOLVERS:
                grid.append((n_items, k, solver))
    return grid


def report_path(results_dir, n_items, k, solver, time_limit, grid_step=None):
    """Report filename, carrying everything that distinguishes a run."""
    budget = "uncapped" if time_limit is None else f"t{time_limit:g}s"
    step = f"-b{grid_step:g}" if grid_step is not None else ""
    return results_dir / (f"V2-n{n_items}-i{k:02d}-{solver}-{budget}"
                          f"{step}.json")


# =========================================================================
# Solving
# =========================================================================
def solve_one(n_items, k, solver, time_limit, seed=SEED, alpha=ALPHA,
              load_factor=LOAD_FACTOR, threads=1, grid_step=GRID_STEP):
    """Run one solver on one instance and return a result dict.

    Every solver is verified against the FULL V2 constraint set by the same
    checker, so no column can score on a packing the others would reject.
    WFBF-grid in particular enforces neither the centre-of-mass nor the
    load-bearing constraints, and its violations are reported rather than
    silently tolerated.
    """
    import mhkp_instances as mi

    BS2 = _load("mhkp_v2_bs", "MHKP-V2-BS.py")
    V2 = _load("mhkp_v2_milp", "mhkp_v2_milp.py")
    sa = BS2._sa

    sa.STABILITY_RULE = STABILITY_RULE

    path = instance_path(n_items, k)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance {path}\n"
            f"Run: python3 mhkp_instances.py --table8 --table12")

    bins_raw, items_raw = mi.read_instance(path)
    inst = sa.load_instance(str(path))
    BS2.attach_v2_data(inst, bins_raw, items_raw, seed=seed, alpha=alpha,
                       factor=load_factor)

    record = {
        "table": table_of(n_items),
        "submodel": "V2",
        "items": n_items,
        "bins": 1,
        "instance": k,
        "instance_file": str(path.relative_to(ROOT)),
        "solver": solver,
        "time_limit": time_limit,
        "seed": seed,
        "alpha": alpha,
        "load_factor": load_factor,
        "stability_rule": STABILITY_RULE,
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    t0 = time.time()

    if solver == "wfbf-grid":
        import wfbf

        beta = wfbf._grid_beta(items_raw)
        step = grid_step

        record.update({
            "mode": "grid",
            "beta": beta,
            "grid_step": step if step is not None else beta,
        })

        with time_limited(time_limit) as enforced:
            record["time_limit_enforced"] = bool(enforced)
            try:
                res = wfbf.wfbf(bins_raw, items_raw, mode="grid",
                                grid_step=step, seed=seed)
                status = "DONE"
            except _Timeout:
                res = None
                status = "TIMEOUT"

        wall = time.time() - t0

        if res is None:
            placements = []
            record.update({
                "objective": None,
                "n_packed": 0,
                "runtime": wall,
                "status": status,
            })
        else:
            placements = [dict(p) for p in res["placements"]]
            for p in placements:
                p.setdefault("reflected", 0)

            record.update({
                "objective": res["objective"],
                "n_packed": len(placements),
                "runtime": wall,
                "status": status,
                "repeats": res["repeats"],
            })

    elif solver == "milp":
        res = V2.solve(bins_raw, items_raw, time_limit=time_limit,
                       threads=threads, seed=seed, alpha=alpha,
                       factor=load_factor)
        wall = time.time() - t0

        placements = res["placements"]

        record.update({
            "objective": res["objective"],
            "bound": res["bound"],
            "gap": res["gap"],
            "n_packed": res["n_packed"],
            "runtime": res["runtime"],
            "build_time": res["build_time"],
            "wall_time": wall,
            "status": res["status_name"],
            "n_vars": res["n_vars"],
            "n_constrs": res["n_constrs"],
        })

    elif solver == "bs":
        res = BS2.beam_search_v2_auto(inst, time_limit=time_limit, seed=seed,
                                      rule=STABILITY_RULE)
        wall = time.time() - t0

        placements = res["placements"]

        record.update({
            "objective": res["wasted"],
            "n_packed": res["n_packed"],
            "runtime": res["runtime"],
            "wall_time": wall,
            "rollouts": res["rollouts"],
            "levels": res["levels"],
            "fell_back": bool(res.get("fell_back")),
            "status": "DONE",
        })

    else:
        raise ValueError(f"unknown solver {solver!r}")

    record["used_bins"] = len({p["bin"] for p in placements}) if placements else 0

    violations = BS2.verify_v2(inst, placements, rule=STABILITY_RULE)
    record["violations"] = len(violations)
    record["violation_detail"] = violations[:10]

    # An empty packing trivially violates nothing, so a run that produced no
    # solution must not be recorded as V2-feasible: there is nothing to have
    # been feasible.
    record["feasible_V2"] = (None if record.get("objective") is None
                             else not violations)

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return record


def run_task(args, entry):
    """Solve one grid entry and write its report."""
    n_items, k, solver = entry

    time_limit = (args.time_limit if args.time_limit is not None
                  else SOLVER_TIME_LIMITS[solver])

    out = report_path(args.results_dir, n_items, k, solver, time_limit,
                      args.grid_step)

    if out.exists() and not args.force:
        print(f"skip (exists): {out.name}   [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)

    record = solve_one(n_items, k, solver, time_limit, seed=args.seed,
                       alpha=args.alpha, load_factor=args.load_factor,
                       threads=args.num_cpu, grid_step=args.grid_step)

    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)

    obj = record["objective"]
    obj_s = "-" if obj is None else f"{obj:.4f}"

    print(f"n{n_items:<4} i{k:02d} {solver:<10} "
          f"obj {obj_s:>9}  used {record['used_bins']}  "
          f"packed {record['n_packed']:>3}  "
          f"{record['runtime']:>8.2f}s  "
          f"viol {record['violations']}  -> {out.name}")

    return 0


# =========================================================================
# Summary
# =========================================================================
def summarize(results_dir, csv_path=None):
    """Read the reports back and print them in the Table 11/12 layout."""
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    records = []
    for path in sorted(results_dir.glob("V2-*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"unparsable report skipped: {path.name}", file=sys.stderr)

    if not records:
        print(f"no reports in {results_dir}", file=sys.stderr)
        return 1

    by = {}
    for r in records:
        by.setdefault((r["items"], r["solver"]), []).append(r)

    header = (f"{'items':>5} | "
              f"{'MILP obj':>9} {'gap%':>7} {'t':>8} | "
              f"{'WFBFg obj':>9} {'t':>8} | "
              f"{'BS obj':>9} {'t':>8} | {'viol':>5}")

    print("=" * len(header))
    print(f"Tables 11+12 (submodel V2, 1 bin) - {len(records)} reports "
          f"from {results_dir.name}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = []
    missing = []

    for n_items in ITEM_COUNTS:

        if n_items == TABLE12_SIZES[0]:
            print("-" * len(header))
            print(f"{'(Table 12, n=100-200)':^{len(header)}}")
            print("-" * len(header))

        cells = {}
        for solver in SOLVERS:
            got = by.get((n_items, solver), [])
            have = [r for r in got if r.get("objective") is not None]
            cells[solver] = have
            if len(got) < INSTANCES_PER_SIZE:
                missing.append(f"n{n_items}/{solver} "
                               f"({len(got)}/{INSTANCES_PER_SIZE})")

        def agg(solver, field, default=float("nan")):
            vals = [r[field] for r in cells[solver] if r.get(field) is not None]
            return statistics.fmean(vals) if vals else default

        row = {
            "items": n_items,
            "table": table_of(n_items),
            "milp_obj": agg("milp", "objective"),
            "milp_gap": agg("milp", "gap"),
            "milp_t": agg("milp", "runtime"),
            "wfbf_grid_obj": agg("wfbf-grid", "objective"),
            "wfbf_grid_t": agg("wfbf-grid", "runtime"),
            "bs_obj": agg("bs", "objective"),
            "bs_t": agg("bs", "runtime"),
            "violations": sum(r.get("violations", 0)
                              for s in SOLVERS for r in cells[s]),
        }
        rows.append(row)

        def f(x, w, p=4):
            return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

        gap = row["milp_gap"]
        gap_s = "-" if gap != gap else f"{gap * 100:>6.2f}%"

        print(f"{n_items:>5} | "
              f"{f(row['milp_obj'], 9)} {gap_s:>7} {f(row['milp_t'], 8, 1)} | "
              f"{f(row['wfbf_grid_obj'], 9)} {f(row['wfbf_grid_t'], 8, 1)} | "
              f"{f(row['bs_obj'], 9)} {f(row['bs_t'], 8, 2)} | "
              f"{row['violations']:>5}")

    print("-" * len(header))

    def mean_of(field, table=None):
        vals = [r[field] for r in rows
                if r[field] == r[field] and (table is None or r["table"] == table)]
        return statistics.fmean(vals) if vals else float("nan")

    def f(x, w, p=4):
        return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

    for tbl in (11, 12):
        print(f"{'mean T' + str(tbl):>10} | "
              f"{f(mean_of('milp_obj', tbl), 9)} {'':>7} "
              f"{f(mean_of('milp_t', tbl), 8, 1)} | "
              f"{f(mean_of('wfbf_grid_obj', tbl), 9)} "
              f"{f(mean_of('wfbf_grid_t', tbl), 8, 1)} | "
              f"{f(mean_of('bs_obj', tbl), 9)} "
              f"{f(mean_of('bs_t', tbl), 8, 2)} | "
              f"{sum(r['violations'] for r in rows if r['table'] == tbl):>5}")

    print(f"{'mean all':>10} | "
          f"{f(mean_of('milp_obj'), 9)} {'':>7} {f(mean_of('milp_t'), 8, 1)} | "
          f"{f(mean_of('wfbf_grid_obj'), 9)} {f(mean_of('wfbf_grid_t'), 8, 1)} | "
          f"{f(mean_of('bs_obj'), 9)} {f(mean_of('bs_t'), 8, 2)} | "
          f"{sum(r['violations'] for r in rows):>5}")

    total_viol = sum(r["violations"] for r in rows)
    if total_viol:
        print()
        print(f"WARNING: {total_viol} V2 constraint violations across the "
              f"reports. WFBF-grid enforces neither the centre-of-mass nor "
              f"the load-bearing constraints, so its packings are expected "
              f"to fail; MILP and BS violations would be bugs.")
        for solver in SOLVERS:
            n = sum(r.get("violations", 0) for r in records
                    if r["solver"] == solver)
            print(f"    {solver:<10} {n}")

    if missing:
        print()
        print(f"incomplete ({len(missing)}):")
        for m in missing[:12]:
            print(f"    {m}")
        if len(missing) > 12:
            print(f"    ... (+{len(missing) - 12})")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwritten: {csv_path}")

    return 0


# =========================================================================
# main
# =========================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SLURM dispatcher for the Tables 11+12 experiment "
                    "(submodel V2, single bin).")

    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--items", type=int, default=None, metavar="N",
                        help="one item count, e.g. 70.")
    parser.add_argument("--instance", type=int, default=None, metavar="K",
                        help=f"instance number, 1..{INSTANCES_PER_SIZE}.")
    parser.add_argument("--solver", choices=SOLVERS, default=None,
                        help="restrict to one solver.")
    parser.add_argument("--grid-step", type=int, default=GRID_STEP,
                        metavar="STEP",
                        help="grid spacing for wfbf-grid (default: the "
                             "instance's own beta, i.e. the literal "
                             "Algorithm 1).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"where the json reports go (default: "
                             f"{RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=None,
                        help="override the solver's own budget "
                             "(milp: 3600 s, wfbf-grid: 900 s, bs: 900 s).")
    parser.add_argument("--num_cpu", type=int, default=1,
                        help="CPUs from SLURM; forwarded to Gurobi as its "
                             "thread count.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"pyramid base fraction (default {ALPHA})")
    parser.add_argument("--load-factor", type=float, default=LOAD_FACTOR,
                        help=f"Lambda_i as a multiple of the maximal weight "
                             f"(default {LOAD_FACTOR})")
    parser.add_argument("--force", action="store_true",
                        help="redo a task whose report already exists.")
    parser.add_argument("--all", action="store_true",
                        help="run every grid entry in this one process "
                             "(respecting --items / --solver filters).")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports and print "
                             "Tables 11+12.")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="with --summarize, also write the table as CSV.")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the SBATCH array range.")

    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir, args.csv)

    grid = build_grid()

    if args.print_grid:
        print(f"{len(grid)} tasks: {len(ITEM_COUNTS)} item counts "
              f"({len(TABLE11_SIZES)} from Table 11, "
              f"{len(TABLE12_SIZES)} from Table 12) x "
              f"{INSTANCES_PER_SIZE} instances x {len(SOLVERS)} solvers")
        print(f"time limits: " + ", ".join(
            f"{s}={t:g}s" for s, t in SOLVER_TIME_LIMITS.items()))
        print()
        for i, (ni, k, s) in enumerate(grid):
            print(f"  {i:4d}  n{ni:<4}  i{k:02d}  {s}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    selected = grid

    if args.items is not None:
        selected = [e for e in selected if e[0] == args.items]
        if not selected:
            parser.error(f"no such item count: {args.items}")

    if args.instance is not None:
        selected = [e for e in selected if e[1] == args.instance]
        if not selected:
            parser.error(f"no instance {args.instance} in the selection")

    if args.solver:
        selected = [e for e in selected if e[2] == args.solver]

    if args.all:
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc

    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id must be in 0..{len(grid) - 1}")
        entry = grid[args.id]
        # --solver, combined with --id, SKIPS this task rather than erroring
        # when the id's own solver does not match - this is what lets one
        # array script exclude a solver (e.g. a broken MILP) by adding
        # --solver to every task's invocation without renumbering the grid
        # or submitting a sparse, hand-built array range.
        if args.solver and entry[2] != args.solver:
            print(f"skip (solver): id {args.id} is "
                  f"n{entry[0]} i{entry[1]:02d} {entry[2]!r}, "
                  f"not {args.solver!r}")
            return 0
        return run_task(args, entry)

    if args.items is not None and args.instance is not None and args.solver:
        return run_task(args, selected[0])

    parser.error("give --id, or --all, or --items with --instance and "
                 "--solver; see --print-grid")


if __name__ == "__main__":
    sys.exit(main())
