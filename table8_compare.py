"""New Table 8: our SA against the paper's WFBF, on identical instances.

Layout follows Deplano et al.'s Table 8 - one bin, sizes 18 to 90 - but every
column here is measured on OUR instances, so the columns are comparable with
each other, which the paper's own values are not (its instances are
unpublished, and several are volume-limited where ours are over-subscribed).

Columns
-------
  WFBF              the paper's Algorithm 1/2, corner-point enumeration, which
                    is WFBF at its best under V0 (the literal grid enumeration
                    is reported separately - it is much worse here because the
                    rank function centres items for a V1/V2 CoM constraint that
                    V0 does not impose).
  SA greedy         our decoder's initial construction, zero SA iterations.
                    This is the like-for-like comparison against WFBF: one
                    constructive pass against another.
  SA 0.1s / 3s      the annealing, at two budgets.
  time              seconds actually consumed.

Objective is formula (1); with one bin and p_j = 1/V_j it is the wasted
fraction, so lower is better and 1 - obj is the fill rate.
"""
import importlib.util, sys, glob, statistics as st, json, time, argparse
from pathlib import Path

ROOT = Path("/Users/marvincaspar/3DBPP-VBS")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v0", ROOT / "MHKP-V0-SA.py")
v0 = importlib.util.module_from_spec(spec); sys.modules["v0"] = v0
spec.loader.exec_module(v0)
import wfbf
import mhkp_instances as mi

SIZES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         35, 40, 45, 50, 55, 60, 65, 70, 80, 90]


def run_size(n, grid_step, budgets):
    files = sorted(glob.glob(str(ROOT / f"MHKP/t8_n{n}/*.txt")))

    acc = {k: [] for k in ("wfbf", "wfbf_t", "wfbf_pk", "grid", "grid_pk",
                           "greedy", "greedy_t", "greedy_pk")}
    for b in budgets:
        acc[f"sa{b}"] = []; acc[f"sa{b}_pk"] = []
    viol = 0

    for f in files:
        bins, items = mi.read_instance(f)
        inst = v0.load_instance(f)

        t = time.time()
        w = wfbf.wfbf(bins, items, mode="corner", seed=1)
        acc["wfbf_t"].append(time.time() - t)
        acc["wfbf"].append(w["objective"]); acc["wfbf_pk"].append(w["n_packed"])
        viol += len(wfbf.verify(bins, items, w["placements"]))

        g = wfbf.wfbf(bins, items, mode="grid", grid_step=grid_step, seed=1)
        acc["grid"].append(g["objective"]); acc["grid_pk"].append(g["n_packed"])
        viol += len(wfbf.verify(bins, items, g["placements"]))

        t = time.time()
        sol = v0.build_initial_solution(inst, stability=True)
        acc["greedy_t"].append(time.time() - t)
        acc["greedy"].append(v0.wasted_space_objective(inst, sol["packed_by_bin"]))
        acc["greedy_pk"].append(len(sol["packed"]))
        viol += len(v0.verify(inst, sol["placements"], stability=True))

        for b in budgets:
            r = v0.simulated_annealing(inst, time_limit=b, seed=1, stability=True)
            acc[f"sa{b}"].append(r["wasted"]); acc[f"sa{b}_pk"].append(r["n_packed"])
            viol += len(v0.verify(inst, r["placements"], stability=True))

    out = {"items": n, "violations": viol, "n_inst": len(files)}
    for k, v in acc.items():
        out[k] = st.fmean(v) if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-step", type=int, default=4)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 3.0])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--out", default="table8_compare.json")
    a = ap.parse_args()

    b1, b2 = a.budgets
    rows = []
    t0 = time.time()

    W = 112
    print("=" * W)
    print("TABLE 8 (rebuilt) - our SA vs the paper's WFBF, identical instances, "
          "submodel V0")
    print("objective (1) = wasted fraction, LOWER IS BETTER | 10 instances per "
          "size | all packings verified")
    print("=" * W)
    print(f"{'':>5} | {'WFBF (paper)':^23} | {'SA greedy':^17} | "
          f"{f'SA {b1}s':^17} | {f'SA {b2}s':^17} |")
    print(f"{'n':>5} | {'obj':>8}{'packed':>8}{'s':>6} | "
          f"{'obj':>8}{'packed':>8} | {'obj':>8}{'packed':>8} | "
          f"{'obj':>8}{'packed':>8} |")
    print("-" * W)

    for n in a.sizes:
        r = run_size(n, a.grid_step, a.budgets)
        rows.append(r)
        print(f"{n:>5} | {r['wfbf']:>8.4f}{r['wfbf_pk']:>6.1f}/{n:<2}"
              f"{r['wfbf_t']:>6.3f} | "
              f"{r['greedy']:>8.4f}{r['greedy_pk']:>6.1f}/{n:<2} | "
              f"{r[f'sa{b1}']:>8.4f}{r[f'sa{b1}_pk']:>6.1f}/{n:<2} | "
              f"{r[f'sa{b2}']:>8.4f}{r[f'sa{b2}_pk']:>6.1f}/{n:<2} |")

    print("-" * W)

    mw = st.fmean(r["wfbf"] for r in rows)
    mg = st.fmean(r["greedy"] for r in rows)
    m1 = st.fmean(r[f"sa{b1}"] for r in rows)
    m2 = st.fmean(r[f"sa{b2}"] for r in rows)
    print(f"{'mean':>5} | {mw:>8.4f}{'':>14} | {mg:>8.4f}{'':>8} | "
          f"{m1:>8.4f}{'':>8} | {m2:>8.4f}{'':>8} |")

    tv = sum(r["violations"] for r in rows)
    print(f"\nverification: {'ALL FEASIBLE' if tv == 0 else f'{tv} VIOLATIONS'}"
          f"    wall clock {time.time()-t0:.0f}s")

    gw = sum(1 for r in rows if r["greedy"] < r["wfbf"] - 1e-12)
    s1 = sum(1 for r in rows if r[f"sa{b1}"] < r["wfbf"] - 1e-12)
    s2 = sum(1 for r in rows if r[f"sa{b2}"] < r["wfbf"] - 1e-12)
    k = len(rows)
    print(f"\nvs WFBF, per size:")
    print(f"  SA greedy (same effort class) better on {gw}/{k}")
    print(f"  SA {b1}s  better on {s1}/{k}")
    print(f"  SA {b2}s  better on {s2}/{k}")

    mgrid = st.fmean(r["grid"] for r in rows)
    print(f"\nWFBF literal grid enumeration (step {a.grid_step}): "
          f"mean obj {mgrid:.4f} vs corner {mw:.4f}")
    print("  the rank function centres items for the V1/V2 CoM constraint,")
    print("  which fragments free space under V0 - see wfbf.py.")

    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
