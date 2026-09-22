#!/usr/bin/env python3
"""SLURM array-job dispatcher for the Table 13 experiment: submodel V2 on the
multi-bin grid of Deplano et al. (2019).

Table 13 compares their heuristic against the V2 MILP on

    items {70, 100, 130, 160, 200}  x  bins {2, 3, 5}

i.e. 15 configurations. We generate five instances per configuration rather
than their one, so a stochastic solver can be averaged, and run FOUR columns
over the same instances:

    wfbf-corner  the paper's constructive heuristic with candidate positions
                 restricted to corner points. Not time limited: a single
                 deterministic pass that finishes in milliseconds.
    wfbf-grid    the same heuristic reading Algorithm 1 line 15 literally -
                 every position on the beta-grid. This is the reading whose
                 RUNTIME is comparable to the paper's own WFBF column;
                 corner mode's milliseconds are not. Capped at 3600 s.
    milp         the V2 MILP (mhkp_v2_milp.py), Gurobi, 3600 s
    bs           the V2 beam search (MHKP-V2-BS.py), 900 s

Both WFBF modes are run because neither alone supports the comparison the
table makes: corner mode gives the better objective and is the only one that
scales, while grid mode is what the paper actually specifies and the only
one whose time can be quoted beside theirs. Runs before this split wrote
solver "wfbf" with the budget tag "uncapped"; they were corner-point runs,
and --summarize reads them as such (see _normalise).

One array task solves one (configuration, instance, solver) triple, so the
grid is 15 x 5 x 4 = 300 jobs:

    #SBATCH --array=0-299

Every task writes ONE json report into --results-dir, named so that nothing
collides across solvers, instances or time limits. --summarize then reads
those reports back and prints the Table 13 layout. Nothing is aggregated in
memory across tasks, because no task ever sees more than its own triple.

The time limits differ per solver BY DESIGN and are not a single knob:
the MILP needs the paper's own 3600 s to be comparable with their column,
while the beam search is given 900 s because it converges long before that
(it emptied the beam in well under 10 s on these instances during
development). wfbf-grid gets 3600 s for the same reason as the MILP, and
wfbf-corner needs none. --time-limit overrides whichever solver the task
runs, for sweeps; SOLVER_TIME_LIMITS holds the defaults.

wfbf.wfbf() has no budget of its own, so wfbf-grid is capped from outside
with SIGALRM and a timed-out task records status TIMEOUT with no objective.
If 3600 s turns out not to be enough at the larger sizes, --grid-step
coarsens the enumeration; that is a computational concession rather than
the paper's line 15, so it lands in both the report and the filename.

Examples:
    python3 main_slurm.py --print-grid
    python3 main_slurm.py --id 0
    python3 main_slurm.py --config 70x2 --instance 1 --solver bs
    python3 main_slurm.py --all --solver wfbf-corner      # one mode
    python3 main_slurm.py --all --solver wfbf             # both modes
    python3 main_slurm.py --all --solver wfbf-grid --grid-step 5
    python3 main_slurm.py --summarize
    python3 main_slurm.py --summarize --csv table13.csv
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
RESULTS_DIR = ROOT / "results_table13_V2"

# Table 13's grid.
ITEM_COUNTS = [70, 100, 130, 160, 200]
BIN_COUNTS = [2, 3, 5]

# Instances per configuration. The paper runs one; five lets a stochastic
# solver be averaged without pretending to a precision their design does not
# claim either.
INSTANCES_PER_CONFIG = 5

SOLVERS = ["wfbf-grid"] #, "wfbf-grid", "milp", "bs"

# The two WFBF columns are the SAME heuristic under the two readings of
# Algorithm 1 line 15, and they are run separately because they are not
# interchangeable:
#
#   wfbf-corner  positions restricted to the origin and the corners opened
#                by placed items. ~12 candidates per item on these
#                instances, hence the millisecond runtimes - which are NOT
#                comparable to the times the paper reports.
#   wfbf-grid    the literal enumeration: every position on the beta-grid,
#                beta being the gcd of the item sides (Definition 4), which
#                is 1 here and so means every integer position in the bin.
#                This is the reading whose TIME is comparable to the
#                paper's own WFBF column.
#
# wfbf.py offers both as mode="corner" / mode="grid"; see its module
# docstring, and table_results_wfbf_grid_vs_corner.tex for the two measured
# side by side at n = 18-22 (~28000x apart there).
WFBF_MODES = {"wfbf-corner": "corner", "wfbf-grid": "grid"}

# Per-solver defaults, in seconds. Corner-mode WFBF is a single
# deterministic pass over a handful of candidate positions and has no time
# limit at all - None records that rather than pretending to a budget it
# does not use. Grid mode enumerates the whole bin per item and does need a
# cap; it gets the MILP's 3600 s so that a timed-out run is still directly
# comparable to the paper's own hour.
SOLVER_TIME_LIMITS = {
    "wfbf-corner": 900,
    "wfbf-grid": 900.0,
    "milp": 3600.0,
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
# solving the same problem.
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

    wfbf.wfbf() takes no time limit of its own - it is a straight-line
    construction with no anytime behaviour to expose - so grid mode, which
    can enumerate millions of positions per item, is capped from outside
    with a real-time alarm. A timed-out task records status TIMEOUT and no
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
def config_name(n_items, n_bins):
    return f"{n_items}x{n_bins}"


def instance_path(n_items, n_bins, k):
    """Path of instance k (1-based) of one configuration."""
    group = f"t10_n{n_items}_b{n_bins}"
    return INSTANCE_ROOT / group / f"{group}_instance{k:02d}.txt"


def build_grid():
    """Every (n_items, n_bins, instance, solver) triple, in a fixed order.

    Solver varies fastest, then instance, then configuration, so a partial
    array still covers whole configurations rather than a ragged edge.
    """
    grid = []
    for n_bins in BIN_COUNTS:
        for n_items in ITEM_COUNTS:
            for k in range(1, INSTANCES_PER_CONFIG + 1):
                for solver in SOLVERS:
                    grid.append((n_items, n_bins, k, solver))
    return grid


def report_path(results_dir, n_items, n_bins, k, solver, time_limit,
                grid_step=None):
    """Report filename, carrying everything that distinguishes a run.

    The time limit is in the name so that a sweep over it keeps its runs
    apart instead of overwriting them, and the solver name now carries the
    WFBF mode - a corner-point run and a grid run of the same instance are
    different experiments, not two takes of one.

    A non-default --grid-step is appended for the same reason: coarsening
    the grid changes what the heuristic can see, so those runs must not
    land on the literal enumeration's filename.
    """
    budget = "uncapped" if time_limit is None else f"t{time_limit:g}s"
    step = f"-b{grid_step:g}" if grid_step is not None else ""
    return results_dir / (f"V2-{config_name(n_items, n_bins)}"
                          f"-i{k:02d}-{solver}-{budget}{step}.json")


# =========================================================================
# Solving
# =========================================================================
def solve_one(n_items, n_bins, k, solver, time_limit, seed=SEED,
              alpha=ALPHA, load_factor=LOAD_FACTOR, threads=1,
              grid_step=GRID_STEP):
    """Run one solver on one instance and return a result dict.

    Every solver is verified against the FULL V2 constraint set by the same
    checker, so no column can score on a packing the others would reject.
    WFBF in particular enforces neither the centre-of-mass nor the
    load-bearing constraints, and its violations are reported rather than
    silently tolerated.
    """
    import mhkp_instances as mi

    BS2 = _load("mhkp_v2_bs", "MHKP-V2-BS.py")
    V2 = _load("mhkp_v2_milp", "mhkp_v2_milp.py")
    sa = BS2._sa

    sa.STABILITY_RULE = STABILITY_RULE

    path = instance_path(n_items, n_bins, k)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance {path}\nRun the Table 10/13 generator first.")

    bins_raw, items_raw = mi.read_instance(path)
    inst = sa.load_instance(str(path))
    BS2.attach_v2_data(inst, bins_raw, items_raw, seed=seed, alpha=alpha,
                       factor=load_factor)

    record = {
        "table": 13,
        "submodel": "V2",
        "config": config_name(n_items, n_bins),
        "items": n_items,
        "bins": n_bins,
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

    if solver in WFBF_MODES:
        import wfbf

        mode = WFBF_MODES[solver]
        beta = wfbf._grid_beta(items_raw)
        step = grid_step if mode == "grid" else None

        record.update({
            "mode": mode,
            "beta": beta,
            "grid_step": step if step is not None else beta,
        })

        # Corner mode is a handful of candidates per item and finishes in
        # milliseconds; grid mode enumerates the bin and is capped from
        # outside, since wfbf.wfbf() has no budget of its own.
        with time_limited(time_limit) as enforced:
            record["time_limit_enforced"] = bool(enforced)
            try:
                res = wfbf.wfbf(bins_raw, items_raw, mode=mode,
                                grid_step=step, seed=seed)
                status = "DONE"
            except _Timeout:
                res = None
                status = "TIMEOUT"

        wall = time.time() - t0

        if res is None:
            # No partial packing to report: the construction was cut mid-bin
            # and its intermediate state is not a solution of anything.
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

    # Used bins, the quantity Table 13 prints in parentheses.
    record["used_bins"] = len({p["bin"] for p in placements}) if placements else 0

    # One verifier for all four columns, over the full V2 constraint set.
    violations = BS2.verify_v2(inst, placements, rule=STABILITY_RULE)
    record["violations"] = len(violations)
    record["violation_detail"] = violations[:10]

    # An empty packing trivially violates nothing, so a run that produced no
    # solution must not be recorded as V2-feasible: there is nothing to have
    # been feasible. None says "not answered" where True would claim a clean
    # verification the run never got to.
    record["feasible_V2"] = (None if record.get("objective") is None
                             else not violations)

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return record


def run_task(args, entry):
    """Solve one grid entry and write its report."""
    n_items, n_bins, k, solver = entry

    time_limit = (args.time_limit if args.time_limit is not None
                  else SOLVER_TIME_LIMITS[solver])

    out = report_path(args.results_dir, n_items, n_bins, k, solver,
                      time_limit, args.grid_step)

    if out.exists() and not args.force:
        print(f"skip (exists): {out.name}   [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)

    record = solve_one(n_items, n_bins, k, solver, time_limit,
                       seed=args.seed, alpha=args.alpha,
                       load_factor=args.load_factor, threads=args.num_cpu,
                       grid_step=args.grid_step)

    # Written atomically: a preempted task must not leave a half-written
    # report that --summarize would then fail to parse.
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)

    obj = record["objective"]
    obj_s = "-" if obj is None else f"{obj:.4f}"

    print(f"{record['config']:>7} i{k:02d} {solver:<11} "
          f"obj {obj_s:>9}  used {record['used_bins']}  "
          f"packed {record['n_packed']:>3}  "
          f"{record['runtime']:>8.2f}s  "
          f"viol {record['violations']}  -> {out.name}")

    return 0


# =========================================================================
# Summary
# =========================================================================
def _normalise(record):
    """Bring a report written before the WFBF split onto today's names.

    Those runs wrote solver "wfbf" with mode "corner" alongside the budget
    tag "uncapped", which named the time limit rather than the thing that
    actually distinguishes them from a grid run. They are corner-point runs
    and are read as such, so the 75 existing reports keep counting instead
    of being silently dropped from the corner column.
    """
    if record.get("solver") == "wfbf":
        record["solver"] = f"wfbf-{record.get('mode', 'corner')}"
    return record


def summarize(results_dir, csv_path=None):
    """Read the reports back and print them in Table 13's layout."""
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    records = []
    for path in sorted(results_dir.glob("V2-*.json")):
        try:
            records.append(_normalise(json.loads(path.read_text())))
        except json.JSONDecodeError:
            print(f"unparsable report skipped: {path.name}", file=sys.stderr)

    if not records:
        print(f"no reports in {results_dir}", file=sys.stderr)
        return 1

    # (config, solver) -> list of records
    by = {}
    for r in records:
        by.setdefault((r["items"], r["bins"], r["solver"]), []).append(r)

    header = (f"{'items':>5} {'bins':>4} | "
              f"{'WFBFc obj':>9} {'used':>4} {'t':>8} | "
              f"{'WFBFg obj':>9} {'used':>4} {'t':>8} | "
              f"{'MILP obj':>9} {'used':>4} {'gap%':>7} {'t':>8} | "
              f"{'BS obj':>9} {'used':>4} {'t':>8} | {'viol':>5}")

    print("=" * len(header))
    print(f"Table 13 (submodel V2, multiple bins) - {len(records)} reports "
          f"from {results_dir.name}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = []
    missing = []

    for n_bins in BIN_COUNTS:
        for n_items in ITEM_COUNTS:

            cells = {}
            for solver in SOLVERS:
                got = by.get((n_items, n_bins, solver), [])
                have = [r for r in got if r.get("objective") is not None]
                cells[solver] = have
                if len(got) < INSTANCES_PER_CONFIG:
                    missing.append(
                        f"{config_name(n_items, n_bins)}/{solver}"
                        f" ({len(got)}/{INSTANCES_PER_CONFIG})")

            def agg(solver, field, default=float("nan")):
                vals = [r[field] for r in cells[solver]
                        if r.get(field) is not None]
                return statistics.fmean(vals) if vals else default

            row = {
                "items": n_items, "bins": n_bins,
                "wfbf_corner_obj": agg("wfbf-corner", "objective"),
                "wfbf_corner_used": agg("wfbf-corner", "used_bins"),
                "wfbf_corner_t": agg("wfbf-corner", "runtime"),
                "wfbf_grid_obj": agg("wfbf-grid", "objective"),
                "wfbf_grid_used": agg("wfbf-grid", "used_bins"),
                "wfbf_grid_t": agg("wfbf-grid", "runtime"),
                "milp_obj": agg("milp", "objective"),
                "milp_used": agg("milp", "used_bins"),
                "milp_gap": agg("milp", "gap"),
                "milp_t": agg("milp", "runtime"),
                "bs_obj": agg("bs", "objective"),
                "bs_used": agg("bs", "used_bins"),
                "bs_t": agg("bs", "runtime"),
                "violations": sum(r.get("violations", 0)
                                  for s in SOLVERS for r in cells[s]),
            }
            rows.append(row)

            def f(x, w, p=4):
                return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

            gap = row["milp_gap"]
            gap_s = "-" if gap != gap else f"{gap * 100:>6.2f}%"

            print(f"{n_items:>5} {n_bins:>4} | "
                  f"{f(row['wfbf_corner_obj'], 9)} "
                  f"{f(row['wfbf_corner_used'], 4, 1)} "
                  f"{f(row['wfbf_corner_t'], 8, 4)} | "
                  f"{f(row['wfbf_grid_obj'], 9)} "
                  f"{f(row['wfbf_grid_used'], 4, 1)} "
                  f"{f(row['wfbf_grid_t'], 8, 2)} | "
                  f"{f(row['milp_obj'], 9)} {f(row['milp_used'], 4, 1)} "
                  f"{gap_s:>7} {f(row['milp_t'], 8, 1)} | "
                  f"{f(row['bs_obj'], 9)} {f(row['bs_used'], 4, 1)} "
                  f"{f(row['bs_t'], 8, 2)} | {row['violations']:>5}")

    print("-" * len(header))

    def mean_of(field):
        vals = [r[field] for r in rows if r[field] == r[field]]
        return statistics.fmean(vals) if vals else float("nan")

    def f(x, w, p=4):
        return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

    print(f"{'mean':>10} | {f(mean_of('wfbf_corner_obj'), 9)} {'':>4} "
          f"{f(mean_of('wfbf_corner_t'), 8, 4)} | "
          f"{f(mean_of('wfbf_grid_obj'), 9)} {'':>4} "
          f"{f(mean_of('wfbf_grid_t'), 8, 2)} | "
          f"{f(mean_of('milp_obj'), 9)} {'':>4} {'':>7} "
          f"{f(mean_of('milp_t'), 8, 1)} | "
          f"{f(mean_of('bs_obj'), 9)} {'':>4} {f(mean_of('bs_t'), 8, 2)} | "
          f"{sum(r['violations'] for r in rows):>5}")

    total_viol = sum(r["violations"] for r in rows)
    if total_viol:
        print()
        print(f"WARNING: {total_viol} V2 constraint violations across the "
              f"reports. WFBF enforces neither the centre-of-mass nor the "
              f"load-bearing constraints, so its packings are expected to "
              f"fail; MILP and BS violations would be bugs.")
        for solver in SOLVERS:
            n = sum(r.get("violations", 0) for r in records
                    if r["solver"] == solver)
            print(f"    {solver:<5} {n}")

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
        description="SLURM dispatcher for the Table 13 experiment "
                    "(submodel V2, multiple bins).")

    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--config", default=None, metavar="ITEMSxBINS",
                        help="one configuration, e.g. 70x2.")
    parser.add_argument("--instance", type=int, default=None, metavar="K",
                        help=f"instance number within the configuration, "
                             f"1..{INSTANCES_PER_CONFIG}.")
    parser.add_argument("--solver", choices=SOLVERS + ["wfbf"], default=None,
                        help="restrict to one solver; the bare 'wfbf' means "
                             "both WFBF modes.")
    parser.add_argument("--grid-step", type=int, default=GRID_STEP,
                        metavar="STEP",
                        help="grid spacing for wfbf-grid (default: the "
                             "instance's own beta, i.e. the literal "
                             "Algorithm 1). A coarser step makes the "
                             "enumeration tractable but is no longer the "
                             "paper's line 15; it is recorded in the report "
                             "and in the filename.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"where the json reports go (default: "
                             f"{RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=None,
                        help="override the solver's own budget "
                             "(wfbf: none, milp: 3600 s, bs: 900 s).")
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
                             "(respecting --config / --solver filters).")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports and print "
                             "Table 13.")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="with --summarize, also write the table as CSV.")
    parser.add_argument("--print-grid", action="store_true",
                        help="print the grid and the SBATCH array range.")

    args = parser.parse_args(argv)

    if args.summarize:
        return summarize(args.results_dir, args.csv)

    grid = build_grid()

    if args.print_grid:
        print(f"{len(grid)} tasks: {len(ITEM_COUNTS)} item counts x "
              f"{len(BIN_COUNTS)} bin counts x {INSTANCES_PER_CONFIG} "
              f"instances x {len(SOLVERS)} solvers")
        print(f"time limits: " + ", ".join(
            f"{s}={'uncapped' if t is None else f'{t:g}s'}"
            for s, t in SOLVER_TIME_LIMITS.items()))
        print()
        for i, (ni, nb, k, s) in enumerate(grid):
            print(f"  {i:3d}  {config_name(ni, nb):>7}  i{k:02d}  {s}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

    # Which entries to run.
    selected = grid

    if args.config:
        try:
            ci, cb = args.config.lower().split("x")
            ci, cb = int(ci), int(cb)
        except ValueError:
            parser.error(f"--config must look like 70x2, got {args.config!r}")
        selected = [e for e in selected if e[0] == ci and e[1] == cb]
        if not selected:
            parser.error(f"no such configuration: {args.config}")

    if args.instance is not None:
        selected = [e for e in selected if e[2] == args.instance]
        if not selected:
            parser.error(f"no instance {args.instance} in the selection")

    if args.solver == "wfbf":
        selected = [e for e in selected if e[3] in WFBF_MODES]
    elif args.solver:
        selected = [e for e in selected if e[3] == args.solver]

    if args.all:
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc

    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id must be in 0..{len(grid) - 1}")
        return run_task(args, grid[args.id])

    if args.config and args.instance is not None and args.solver:
        # "--solver wfbf" selects both modes, so run whatever the filters
        # left rather than silently taking the first of them.
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc

    parser.error("give --id, or --all, or --config with --instance and "
                 "--solver; see --print-grid")


if __name__ == "__main__":
    sys.exit(main())
