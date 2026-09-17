"""Table 8, all 23 sizes: beam search vs simulated annealing vs hill climbing.

All three solvers share the decoder, the stability rule and the verifier, so
differences are attributable to the metaheuristic alone. Objective is formula
(1), the priority-weighted wasted space; with one bin and p_j = 1/V_j it is the
wasted fraction, so LOWER IS BETTER.

WFBF (the paper's own heuristic) is included as the constructive baseline.
"""
import importlib.util, sys, glob, statistics as st, json, time, argparse
from pathlib import Path

ROOT = Path("/Users/marvincaspar/3DBPP-VBS")
sys.path.insert(0, str(ROOT))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, ROOT / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


bs = load("bs", "MHKP-V0-BS.py")
hc = load("hc", "MHKP-V0-HC.py")
v0 = bs._sa
import wfbf
import mhkp_instances as mi

SIZES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         35, 40, 45, 50, 55, 60, 65, 70, 80, 90]


def run_size(n, tl, seed, width, actions):
    files = sorted(glob.glob(str(ROOT / f"MHKP/t8_n{n}/*.txt")))
    acc = {k: [] for k in ("bs", "sa", "hc", "wf", "bs_pk", "sa_pk", "hc_pk",
                           "bs_roll", "sa_it", "hc_it")}
    viol = 0
    wins = {"BS": 0, "SA": 0, "HC": 0}

    for f in files:
        inst = v0.load_instance(f)
        bins, items = mi.read_instance(f)

        r = bs.beam_search(inst, width=width, max_actions=actions,
                           time_limit=tl, seed=seed)
        s = v0.simulated_annealing(inst, time_limit=tl, seed=seed,
                                   stability=True)
        h = hc.hill_climb(inst, time_limit=tl, seed=seed, stability=True)
        w = wfbf.wfbf(bins, items, mode="corner", seed=seed)

        for tag, res in (("bs", r), ("sa", s), ("hc", h)):
            viol += len(v0.verify(inst, res["placements"], stability=True))
        viol += len(wfbf.verify(bins, items, w["placements"]))

        acc["bs"].append(r["wasted"]); acc["bs_pk"].append(r["n_packed"])
        acc["sa"].append(s["wasted"]); acc["sa_pk"].append(s["n_packed"])
        acc["hc"].append(h["wasted"]); acc["hc_pk"].append(h["n_packed"])
        acc["wf"].append(w["objective"])
        acc["bs_roll"].append(r["rollouts"]); acc["sa_it"].append(s["iterations"])
        acc["hc_it"].append(h["iterations"])

        trio = [("BS", r["wasted"]), ("SA", s["wasted"]), ("HC", h["wasted"])]
        wins[min(trio, key=lambda t: t[1])[0]] += 1

    out = {"items": n, "violations": viol, "wins": wins,
           "n_inst": len(files)}
    for k, v in acc.items():
        out[k] = st.fmean(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--actions", type=int, default=60)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--out", default="table8_three.json")
    a = ap.parse_args()

    W = 104
    print("=" * W)
    print("TABLE 8, ALL 23 SIZES - beam search vs SA vs hill climbing "
          "(submodel V0)")
    print(f"objective (1), LOWER IS BETTER | {a.time_limit}s per solver per "
          f"instance | 10 instances per size")
    print(f"beam w={a.width}, m={a.actions} | WFBF = the paper's constructive "
          "heuristic, for reference")
    print("=" * W)
    print(f"{'n':>4} | {'WFBF':>8} | {'BS':>8}{'SA':>9}{'HC':>9} | "
          f"{'best':>5} | {'BS roll':>8}{'SA iter':>9}{'HC iter':>9} | {'viol':>5}")
    print("-" * W)

    rows = []
    t0 = time.time()

    for n in a.sizes:
        r = run_size(n, a.time_limit, a.seed, a.width, a.actions)
        rows.append(r)
        best = min([("BS", r["bs"]), ("SA", r["sa"]), ("HC", r["hc"])],
                   key=lambda t: t[1])[0]
        print(f"{n:>4} | {r['wf']:>8.4f} | {r['bs']:>8.4f}{r['sa']:>9.4f}"
              f"{r['hc']:>9.4f} | {best:>5} | {r['bs_roll']:>8.0f}"
              f"{r['sa_it']:>9.0f}{r['hc_it']:>9.0f} | {r['violations']:>5}")

    print("-" * W)
    mb = st.fmean(r["bs"] for r in rows)
    ms = st.fmean(r["sa"] for r in rows)
    mh = st.fmean(r["hc"] for r in rows)
    mw = st.fmean(r["wf"] for r in rows)
    print(f"{'mean':>4} | {mw:>8.4f} | {mb:>8.4f}{ms:>9.4f}{mh:>9.4f} |")

    tw = {"BS": 0, "SA": 0, "HC": 0}
    for r in rows:
        for k in tw:
            tw[k] += r["wins"][k]
    total = sum(tw.values())
    print(f"\nper-instance wins over all {total} runs: "
          f"BS {tw['BS']}, SA {tw['SA']}, HC {tw['HC']}")

    sizes_best = {"BS": 0, "SA": 0, "HC": 0}
    for r in rows:
        b = min([("BS", r["bs"]), ("SA", r["sa"]), ("HC", r["hc"])],
                key=lambda t: t[1])[0]
        sizes_best[b] += 1
    print(f"sizes where each is best on the mean: "
          f"BS {sizes_best['BS']}, SA {sizes_best['SA']}, HC {sizes_best['HC']}")

    small = [r for r in rows if r["items"] <= 30]
    large = [r for r in rows if r["items"] >= 45]
    for lab, g in (("n<=30", small), ("n>=45", large)):
        if not g:
            continue
        print(f"  {lab}: BS {st.fmean(r['bs'] for r in g):.4f}  "
              f"SA {st.fmean(r['sa'] for r in g):.4f}  "
              f"HC {st.fmean(r['hc'] for r in g):.4f}")

    tv = sum(r["violations"] for r in rows)
    print(f"\nverification: {'ALL FEASIBLE' if tv == 0 else f'{tv} VIOLATIONS'}"
          f"   wall clock {time.time()-t0:.0f}s")
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
