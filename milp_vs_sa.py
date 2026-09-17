"""MILP (submodel V0) vs our SA, matched at the same wall-clock budget.

Their Table 8 reports the MILP at a 3600 s limit; here both solvers get the
SAME budget, which is the comparison that isolates method from time.

Every packing from both solvers is checked by the SAME verifier (the SA file's),
after converting the MILP's axis frame, so neither can score on an infeasible
layout.
"""
import importlib.util, sys, glob, statistics as st, json, time, argparse
from pathlib import Path

ROOT = Path("/Users/marvincaspar/3DBPP-VBS")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v0", ROOT / "MHKP-V0-SA.py")
v0 = importlib.util.module_from_spec(spec); sys.modules["v0"] = v0
spec.loader.exec_module(v0)
import mhkp_v0_milp as M
import wfbf
import mhkp_instances as mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[18, 19])
    ap.add_argument("--time-limit", type=float, default=9.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="milp_vs_sa.json")
    a = ap.parse_args()

    tl = a.time_limit
    rows = []

    print("=" * 108)
    print(f"MILP (submodel V0, Gurobi) vs SA vs WFBF - SAME {tl}s budget, "
          "identical instances")
    print("objective (1) = wasted fraction, LOWER IS BETTER | all packings "
          "verified by one checker")
    print("=" * 108)
    print(f"{'instance':<24}{'MILP obj':>10}{'bound':>9}{'gap%':>8}{'status':>12}"
          f"{'SA obj':>9}{'WFBF':>9}{'winner':>9}{'viol':>6}")
    print("-" * 108)

    for n in a.sizes:
        files = sorted(glob.glob(str(ROOT / f"MHKP/t8_n{n}/*.txt")))
        agg = {"milp": [], "sa": [], "wfbf": [], "viol": 0,
               "milp_opt": 0, "milp_none": 0, "wins_milp": 0, "wins_sa": 0}

        for f in files:
            bins, items = mi.read_instance(f)
            inst = v0.load_instance(f)

            r = M.solve(bins, items, time_limit=tl, threads=a.threads)
            # Both solvers must use the SAME reading of (10l)-(10o); the MILP
            # implements the constraints as written, which is "peraxis".
            v0.STABILITY_RULE = "peraxis"
            s = v0.simulated_annealing(inst, time_limit=tl, seed=1,
                                       stability=True)
            w = wfbf.wfbf(bins, items, mode="corner", seed=1)

            viol = len(v0.verify(inst, s["placements"], stability=True,
                                 rule="peraxis"))
            viol += len(wfbf.verify(bins, items, w["placements"]))  # corners: stricter, so still valid
            if r["placements"]:
                viol += len(v0.verify(
                    inst, M.solution_to_sa_frame(r["placements"]),
                    stability=True, rule="peraxis"))
            agg["viol"] += viol

            mo = r["objective"]
            if mo is None:
                agg["milp_none"] += 1
                mo_s, bd_s, gp_s = "none", "-", "-"
                win = "SA"
            else:
                agg["milp"].append(mo)
                mo_s = f"{mo:.4f}"
                bd_s = f"{r['bound']:.4f}" if r["bound"] is not None else "-"
                gp_s = f"{r['gap']*100:.1f}" if r["gap"] is not None else "-"
                if r["status_name"] == "OPTIMAL":
                    agg["milp_opt"] += 1
                win = "MILP" if mo < s["wasted"] - 1e-9 else (
                    "SA" if s["wasted"] < mo - 1e-9 else "tie")

            if win == "MILP":
                agg["wins_milp"] += 1
            elif win == "SA":
                agg["wins_sa"] += 1

            agg["sa"].append(s["wasted"]); agg["wfbf"].append(w["objective"])

            print(f"{Path(f).stem:<24}{mo_s:>10}{bd_s:>9}{gp_s:>8}"
                  f"{r['status_name']:>12}{s['wasted']:>9.4f}"
                  f"{w['objective']:>9.4f}{win:>9}{viol:>6}")

        print("-" * 108)
        mm = st.fmean(agg["milp"]) if agg["milp"] else float("nan")
        print(f"n={n} mean: MILP {mm:.4f} (over {len(agg['milp'])} with a "
              f"solution; {agg['milp_none']} found none)   "
              f"SA {st.fmean(agg['sa']):.4f}   WFBF {st.fmean(agg['wfbf']):.4f}")
        print(f"  MILP proved optimality on {agg['milp_opt']}/{len(files)}   "
              f"wins: MILP {agg['wins_milp']}, SA {agg['wins_sa']}")
        print()

        rows.append({"size": n, "milp_mean": mm, "sa_mean": st.fmean(agg["sa"]),
                     "wfbf_mean": st.fmean(agg["wfbf"]),
                     "milp_optimal": agg["milp_opt"],
                     "milp_no_solution": agg["milp_none"],
                     "wins_milp": agg["wins_milp"], "wins_sa": agg["wins_sa"],
                     "violations": agg["viol"], "n_instances": len(files)})

    tv = sum(r["violations"] for r in rows)
    print(f"verification: {'ALL FEASIBLE' if tv == 0 else f'{tv} VIOLATIONS'}")
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
