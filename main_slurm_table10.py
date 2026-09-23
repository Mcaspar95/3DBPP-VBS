#!/usr/bin/env python3
"""SLURM array-job dispatcher for the Table 10 experiment: our MILP and beam
search under submodels V0, V1 and V2 on the multi-bin grid of Deplano et al.
(2019).

Table 10 of the paper compares THEIR OWN V0/V1/V2 MILP results on

    items {70, 100, 130, 160, 200}  x  bins {2, 3, 5}

i.e. 15 configurations, at a 3600 s limit, with no heuristic column at all
(that comparison is Table 13, and main_slurm.py already covers it for V2).
This file rebuilds the SAME grid and reports our own two solvers - the MILP
of mhkp_v{0,1,2}_milp.py and the beam search of MHKP-V{0,1,2}-BS.py - each
at the paper's own budget for that method: 3600 s for the MILP (matching
Table 10's own limit) and 900 s for the beam search (matching the budget
used throughout this project's other multi-bin runs, e.g. Table 13).

We generate five instances per configuration rather than their one, so a
stochastic solver can be averaged, and run SIX columns over the same
instances - one MILP and one BS run per submodel:

    v0-milp   V0 (stability only), Gurobi, 3600 s
    v0-bs     V0 beam search, 900 s
    v1-milp   V1 (+ centre of mass), Gurobi, 3600 s
    v1-bs     V1 beam search, 900 s
    v2-milp   V2 (+ load bearing), Gurobi, 3600 s
    v2-bs     V2 beam search, 900 s

Every submodel and method sees IDENTICAL derived data (weights, centres of
mass, load limits) for a given instance, seed, alpha and load factor, so
that V0/V1/V2 differ only in which constraints are active, not in what data
they are solving over - the same discipline main_slurm.py already applies
to V2's three columns.

One array task solves one (n_items, n_bins, instance, column) sextuple, so
the grid is 15 configurations x 5 instances x 6 columns = 450 jobs:

    #SBATCH --array=0-449

Every task writes ONE json report into --results-dir, named so that nothing
collides across submodels, methods, instances or time limits. --summarize
reads those reports back and prints the Table 10 layout: objective (1) with
the number of opened bins in parentheses, split by submodel, mirroring the
paper's own table. Nothing is aggregated in memory across tasks, because no
task ever sees more than its own sextuple.

KNOWN ISSUE (see table_results_table11_v2.tex, table_results_table13_v2.tex):
the V1/V2 MILP has a confirmed defect where it returns the empty packing as
status OPTIMAL with a zero gap once instances are large enough (n >= 27 on
the single-bin sweep; also seen on this multi-bin grid). It is run here
anyway, because the defect and its extent on THIS grid, across all three
submodels, is itself part of what this experiment is meant to show; the
column is reported as measured, exactly as main_slurm.py already does for
V2's MILP column in Table 13.

Examples:
    python3 main_slurm_table10.py --print-grid
    python3 main_slurm_table10.py --id 0
    python3 main_slurm_table10.py --config 70x2 --instance 1 --column v1-bs
    python3 main_slurm_table10.py --all --column v0-milp
    python3 main_slurm_table10.py --summarize
    python3 main_slurm_table10.py --summarize --csv table10.csv
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
RESULTS_DIR = ROOT / "results_table10"

# Table 10's grid - identical to Table 13's (main_slurm.py), since both
# compare methods on the same multi-bin instances.
ITEM_COUNTS = [70, 100, 130, 160, 200]
BIN_COUNTS = [2, 3, 5]

# Instances per configuration. The paper runs one; five lets a stochastic
# solver be averaged without pretending to a precision their design does
# not claim either. Reuses the t10_n{items}_b{bins} groups already
# generated for Table 13.
INSTANCES_PER_CONFIG = 5

SUBMODELS = ["v0", "v1", "v2"]
METHODS = ["milp", "bs"]

# One column per (submodel, method) pair, in the order the summary prints
# them - submodel outermost, to match the paper's V0 | V1 | V2 layout.
COLUMNS = [f"{sm}-{me}" for sm in SUBMODELS for me in METHODS]

# Per-method defaults, in seconds. The MILP gets the paper's own Table 10
# budget; the beam search gets the 900 s used throughout this project's
# other multi-bin experiments (Table 13). Both are independent of
# submodel: V1 and V2 do not get a longer budget for having more
# constraints, exactly as in the paper's own table.
METHOD_TIME_LIMITS = {"milp": 3600.0, "bs": 900.0}

# V1/V2 data derivation (mhkp_v2_milp.derive_v2_data covers V1's fields
# too). Fixed here so that every submodel and method sees IDENTICAL
# weights, centres of mass and load limits for a given instance - without
# that, V0 vs V1 vs V2 would not be comparisons of the SAME instance under
# different constraints, but of different instances entirely.
SEED = 1
ALPHA = 0.8            # pyramid base fraction, the paper's suggested default
LOAD_FACTOR = 3.0      # Lambda_i = 3 * Omega_i, their equation (18)
STABILITY_RULE = "peraxis"

_MODULES = {}


def _load(name, filename):
    """Import one of the solver modules by path and cache it.

    The solver filenames are not importable module names (they start with a
    digit and contain dashes, or are dashed themselves), hence the explicit
    spec loading.
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


# =========================================================================
# Grid
# =========================================================================
def config_name(n_items, n_bins):
    return f"{n_items}x{n_bins}"


def instance_path(n_items, n_bins, k):
    """Path of instance k (1-based) of one configuration.

    Reuses the t10_n{items}_b{bins} groups generated for Table 13
    (main_slurm.py), so this experiment runs on exactly the same
    instances as that one - V0/V1/V2 here are directly comparable to the
    V2 column already reported there.
    """
    group = f"t10_n{n_items}_b{n_bins}"
    return INSTANCE_ROOT / group / f"{group}_instance{k:02d}.txt"


def build_grid():
    """Every (n_items, n_bins, instance, column) quadruple, in a fixed
    order.

    Column varies fastest, then instance, then configuration, so a partial
    array still covers whole configurations rather than a ragged edge.
    """
    grid = []
    for n_bins in BIN_COUNTS:
        for n_items in ITEM_COUNTS:
            for k in range(1, INSTANCES_PER_CONFIG + 1):
                for column in COLUMNS:
                    grid.append((n_items, n_bins, k, column))
    return grid


def report_path(results_dir, n_items, n_bins, k, column, time_limit):
    """Report filename, carrying everything that distinguishes a run."""
    budget = "uncapped" if time_limit is None else f"t{time_limit:g}s"
    return results_dir / (f"T10-{config_name(n_items, n_bins)}"
                          f"-i{k:02d}-{column}-{budget}.json")


# =========================================================================
# Solving
# =========================================================================
def solve_one(n_items, n_bins, k, column, time_limit, seed=SEED,
              alpha=ALPHA, load_factor=LOAD_FACTOR, threads=1):
    """Run one (submodel, method) column on one instance and return a
    result dict.

    Every submodel's solutions are verified against THAT SAME SUBMODEL's
    full constraint set: a V0 run is checked for stability only, a V1 run
    additionally for the centre-of-mass pyramid, a V2 run additionally for
    load bearing. Comparing a V0 packing against V2's constraints would be
    comparing it to requirements it was never asked to meet.
    """
    import mhkp_instances as mi

    submodel, method = column.split("-")

    path = instance_path(n_items, n_bins, k)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing instance {path}\nRun the Table 10/13 generator "
            f"first (mhkp_instances.py / custom_instances.py, t10_n* "
            f"groups).")

    bins_raw, items_raw = mi.read_instance(path)

    record = {
        "table": 10,
        "submodel": submodel.upper(),
        "method": method,
        "column": column,
        "config": config_name(n_items, n_bins),
        "items": n_items,
        "bins": n_bins,
        "instance": k,
        "instance_file": str(path.relative_to(ROOT)),
        "time_limit": time_limit,
        "seed": seed,
        "alpha": alpha,
        "load_factor": load_factor,
        "stability_rule": STABILITY_RULE,
        "host": socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    t0 = time.time()

    if submodel == "v0":

        sa = _load("mhkp_v0_sa", "MHKP-V0-SA.py")
        sa.STABILITY_RULE = STABILITY_RULE
        inst = sa.load_instance(str(path))

        if method == "milp":
            V0 = _load("mhkp_v0_milp", "mhkp_v0_milp.py")
            res = V0.solve(bins_raw, items_raw, time_limit=time_limit,
                           threads=threads)
            placements = res["placements"]

            record.update({
                "objective": res["objective"], "bound": res["bound"],
                "gap": res["gap"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "build_time": res["build_time"],
                "status": res["status_name"], "n_vars": res["n_vars"],
                "n_constrs": res["n_constrs"],
            })
        else:
            BS0 = _load("mhkp_v0_bs", "MHKP-V0-BS.py")
            res = BS0.beam_search(inst, time_limit=time_limit, seed=seed,
                                  rule=STABILITY_RULE)
            placements = res["placements"]

            record.update({
                "objective": res["wasted"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "rollouts": res["rollouts"],
                "levels": res["levels"], "status": "DONE",
            })

        violations = sa.verify(inst, placements, stability=True,
                               rule=STABILITY_RULE)

    elif submodel == "v1":

        BS1 = _load("mhkp_v1_bs", "MHKP-V1-BS.py")
        sa = BS1._sa
        sa.STABILITY_RULE = STABILITY_RULE
        inst = sa.load_instance(str(path))
        BS1.attach_v1_data(inst, bins_raw, items_raw, seed=seed, alpha=alpha)

        if method == "milp":
            V1 = _load("mhkp_v1_milp", "mhkp_v1_milp.py")
            res = V1.solve(bins_raw, items_raw, time_limit=time_limit,
                           threads=threads, seed=seed, alpha=alpha)
            placements = res["placements"]

            record.update({
                "objective": res["objective"], "bound": res["bound"],
                "gap": res["gap"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "build_time": res["build_time"],
                "status": res["status_name"], "n_vars": res["n_vars"],
                "n_constrs": res["n_constrs"],
            })
        else:
            res = BS1.beam_search_v1_auto(inst, time_limit=time_limit,
                                          seed=seed, rule=STABILITY_RULE)
            placements = res["placements"]

            record.update({
                "objective": res["wasted"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "rollouts": res["rollouts"],
                "levels": res["levels"],
                "fell_back": bool(res.get("fell_back")), "status": "DONE",
            })

        violations = BS1.verify_v1(inst, placements, rule=STABILITY_RULE)

    elif submodel == "v2":

        BS2 = _load("mhkp_v2_bs", "MHKP-V2-BS.py")
        sa = BS2._sa
        sa.STABILITY_RULE = STABILITY_RULE
        inst = sa.load_instance(str(path))
        BS2.attach_v2_data(inst, bins_raw, items_raw, seed=seed, alpha=alpha,
                           factor=load_factor)

        if method == "milp":
            V2 = _load("mhkp_v2_milp", "mhkp_v2_milp.py")
            res = V2.solve(bins_raw, items_raw, time_limit=time_limit,
                           threads=threads, seed=seed, alpha=alpha,
                           factor=load_factor)
            placements = res["placements"]

            record.update({
                "objective": res["objective"], "bound": res["bound"],
                "gap": res["gap"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "build_time": res["build_time"],
                "status": res["status_name"], "n_vars": res["n_vars"],
                "n_constrs": res["n_constrs"],
            })
        else:
            res = BS2.beam_search_v2_auto(inst, time_limit=time_limit,
                                          seed=seed, rule=STABILITY_RULE)
            placements = res["placements"]

            record.update({
                "objective": res["wasted"], "n_packed": res["n_packed"],
                "runtime": res["runtime"], "rollouts": res["rollouts"],
                "levels": res["levels"],
                "fell_back": bool(res.get("fell_back")), "status": "DONE",
            })

        violations = BS2.verify_v2(inst, placements, rule=STABILITY_RULE)

    else:
        raise ValueError(f"unknown submodel {submodel!r}")

    record["wall_time"] = time.time() - t0

    # Used bins, the quantity Table 10 prints in parentheses.
    record["used_bins"] = len({p["bin"] for p in placements}) if placements else 0

    record["violations"] = len(violations)
    record["violation_detail"] = violations[:10]

    # An empty packing trivially violates nothing, so a run that produced
    # no solution must not be recorded as feasible: there is nothing to
    # have been feasible.
    record["feasible"] = (None if record.get("objective") is None
                          else not violations)

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return record


def run_task(args, entry):
    """Solve one grid entry and write its report."""
    n_items, n_bins, k, column = entry

    method = column.split("-")[1]
    time_limit = (args.time_limit if args.time_limit is not None
                  else METHOD_TIME_LIMITS[method])

    out = report_path(args.results_dir, n_items, n_bins, k, column,
                      time_limit)

    if out.exists() and not args.force:
        print(f"skip (exists): {out.name}   [--force to redo]")
        return 0

    args.results_dir.mkdir(parents=True, exist_ok=True)

    record = solve_one(n_items, n_bins, k, column, time_limit,
                       seed=args.seed, alpha=args.alpha,
                       load_factor=args.load_factor, threads=args.num_cpu)

    # Written atomically: a preempted task must not leave a half-written
    # report that --summarize would then fail to parse.
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)

    obj = record["objective"]
    obj_s = "-" if obj is None else f"{obj:.4f}"

    print(f"{record['config']:>7} i{k:02d} {column:<8} "
          f"obj {obj_s:>9}  used {record['used_bins']}  "
          f"packed {record['n_packed']:>3}  "
          f"{record['runtime']:>8.2f}s  "
          f"viol {record['violations']}  -> {out.name}")

    return 0


# =========================================================================
# Summary
# =========================================================================
def summarize(results_dir, csv_path=None):
    """Read the reports back and print them in Table 10's layout: V0, V1,
    V2 side by side, MILP and BS as a double column per submodel."""
    if not results_dir.is_dir():
        print(f"no results directory {results_dir}", file=sys.stderr)
        return 1

    records = []
    for path in sorted(results_dir.glob("T10-*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"unparsable report skipped: {path.name}", file=sys.stderr)

    if not records:
        print(f"no reports in {results_dir}", file=sys.stderr)
        return 1

    by = {}
    for r in records:
        by.setdefault((r["items"], r["bins"], r["column"]), []).append(r)

    col_width = len("MILP obj") + 4 + 7 + len("BS obj") + 4 + 7  # one submodel block
    submodel_row = (f"{'items':>5} {'bins':>4} | "
                    + " | ".join(f"{sm.upper():^{col_width}}"
                                 for sm in SUBMODELS)
                    + f" | {'viol':>5}")
    header = (f"{'items':>5} {'bins':>4} | " + " | ".join(
        f"{'MILP obj':>9}{'used':>4}{'t':>7}{'BS obj':>9}{'used':>4}{'t':>7}"
        for _ in SUBMODELS) + f" | {'viol':>5}")

    print("=" * len(header))
    print(f"Table 10 (V0/V1/V2, MILP vs BS, multiple bins) - "
          f"{len(records)} reports from {results_dir.name}")
    print("=" * len(header))
    print(submodel_row)
    print(header)
    print("-" * len(header))

    rows = []
    missing = []

    for n_bins in BIN_COUNTS:
        for n_items in ITEM_COUNTS:

            cells = {}
            for column in COLUMNS:
                got = by.get((n_items, n_bins, column), [])
                have = [r for r in got if r.get("objective") is not None]
                cells[column] = have
                if len(got) < INSTANCES_PER_CONFIG:
                    missing.append(f"{config_name(n_items, n_bins)}/{column}"
                                  f" ({len(got)}/{INSTANCES_PER_CONFIG})")

            def agg(column, field, default=float("nan")):
                vals = [r[field] for r in cells[column]
                        if r.get(field) is not None]
                return statistics.fmean(vals) if vals else default

            row = {"items": n_items, "bins": n_bins}
            for sm in SUBMODELS:
                row[f"{sm}_milp_obj"] = agg(f"{sm}-milp", "objective")
                row[f"{sm}_milp_used"] = agg(f"{sm}-milp", "used_bins")
                row[f"{sm}_milp_t"] = agg(f"{sm}-milp", "runtime")
                row[f"{sm}_bs_obj"] = agg(f"{sm}-bs", "objective")
                row[f"{sm}_bs_used"] = agg(f"{sm}-bs", "used_bins")
                row[f"{sm}_bs_t"] = agg(f"{sm}-bs", "runtime")

            row["violations"] = sum(r.get("violations", 0)
                                    for c in COLUMNS for r in cells[c])
            rows.append(row)

            def f(x, w, p=4):
                return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

            cells_str = []
            for sm in SUBMODELS:
                cells_str.append(
                    f"{f(row[f'{sm}_milp_obj'], 9)}"
                    f"{f(row[f'{sm}_milp_used'], 4, 1)}"
                    f"{f(row[f'{sm}_milp_t'], 7, 1)}"
                    f"{f(row[f'{sm}_bs_obj'], 9)}"
                    f"{f(row[f'{sm}_bs_used'], 4, 1)}"
                    f"{f(row[f'{sm}_bs_t'], 7, 1)}")

            print(f"{n_items:>5} {n_bins:>4} | " + " | ".join(cells_str)
                  + f" | {row['violations']:>5}")

    print("-" * 100)

    def mean_of(field):
        vals = [r[field] for r in rows if r[field] == r[field]]
        return statistics.fmean(vals) if vals else float("nan")

    def f(x, w, p=4):
        return f"{'-':>{w}}" if x != x else f"{x:>{w}.{p}f}"

    cells_str = []
    for sm in SUBMODELS:
        cells_str.append(
            f"{f(mean_of(f'{sm}_milp_obj'), 9)}{'':>4}"
            f"{f(mean_of(f'{sm}_milp_t'), 7, 1)}"
            f"{f(mean_of(f'{sm}_bs_obj'), 9)}{'':>4}"
            f"{f(mean_of(f'{sm}_bs_t'), 7, 1)}")

    print(f"{'mean':>10} | " + " | ".join(cells_str)
          + f" | {sum(r['violations'] for r in rows):>5}")

    total_viol = sum(r["violations"] for r in rows)
    if total_viol:
        print()
        print(f"WARNING: {total_viol} constraint violations across the "
              f"reports (each checked against ITS OWN submodel's full "
              f"constraint set).")
        for column in COLUMNS:
            n = sum(r.get("violations", 0) for r in records
                    if r["column"] == column)
            print(f"    {column:<8} {n}")

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
        description="SLURM dispatcher for the Table 10 experiment "
                    "(our MILP and beam search, V0/V1/V2, multiple bins).")

    parser.add_argument("--id", type=int, default=None,
                        help="SLURM_ARRAY_TASK_ID, zero-based.")
    parser.add_argument("--config", default=None, metavar="ITEMSxBINS",
                        help="one configuration, e.g. 70x2.")
    parser.add_argument("--instance", type=int, default=None, metavar="K",
                        help=f"instance number within the configuration, "
                             f"1..{INSTANCES_PER_CONFIG}.")
    parser.add_argument("--column", choices=COLUMNS, default=None,
                        help="restrict to one (submodel, method) column, "
                             "e.g. v1-bs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help=f"where the json reports go (default: "
                             f"{RESULTS_DIR.name})")
    parser.add_argument("--time-limit", "--timelimit", dest="time_limit",
                        type=float, default=None,
                        help="override the method's own budget "
                             "(milp: 3600 s, bs: 900 s).")
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
                             "(respecting --config / --column filters).")
    parser.add_argument("--summarize", action="store_true",
                        help="do not solve: read the reports and print "
                             "Table 10.")
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
              f"instances x {len(COLUMNS)} columns")
        print(f"columns: {', '.join(COLUMNS)}")
        print(f"time limits: " + ", ".join(
            f"{m}={t:g}s" for m, t in METHOD_TIME_LIMITS.items()))
        print()
        for i, (ni, nb, k, c) in enumerate(grid):
            print(f"  {i:3d}  {config_name(ni, nb):>7}  i{k:02d}  {c}")
        print(f"\n#SBATCH --array=0-{len(grid) - 1}")
        return 0

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

    if args.column:
        selected = [e for e in selected if e[3] == args.column]

    if args.all:
        rc = 0
        for entry in selected:
            rc |= run_task(args, entry)
        return rc

    if args.id is not None:
        if not 0 <= args.id < len(grid):
            parser.error(f"--id must be in 0..{len(grid) - 1}")
        entry = grid[args.id]
        # --column, combined with --id, SKIPS this task rather than
        # erroring when the id's own column does not match - this is what
        # lets one array script exclude a column (e.g. a known-broken MILP)
        # by adding --column to every task's invocation without
        # renumbering the grid or submitting a sparse array range.
        if args.column and entry[3] != args.column:
            print(f"skip (column): id {args.id} is "
                  f"{config_name(entry[0], entry[1])} i{entry[2]:02d} "
                  f"{entry[3]!r}, not {args.column!r}")
            return 0
        return run_task(args, entry)

    if args.config and args.instance is not None and args.column:
        return run_task(args, selected[0])

    parser.error("give --id, or --all, or --config with --instance and "
                 "--column; see --print-grid")


if __name__ == "__main__":
    sys.exit(main())
