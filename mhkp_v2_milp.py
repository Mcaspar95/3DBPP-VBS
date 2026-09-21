#!/usr/bin/env python3
"""
Submodel V2 of Deplano et al. (2019) as a MILP, solved with Gurobi
==================================================================

V2 = V1 plus load bearing, i.e. constraints (1)-(13g) of

    Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
    "A mixed-integer linear model for the multiple heterogeneous knapsack
     problem with realistic container loading constraints and bins' priority"
    Intl. Trans. in Op. Res. 28(6), 3244-3275.

The three submodels of the paper's Section 3.2:

    V0  (1)-(10r)   geometry, priority objective, static stability
    V1  (1)-(12n)   V0 + centre-of-mass distribution
    V2  (1)-(13g)   V1 + load bearing

This file BUILDS ON mhkp_v1_milp.build_model, which in turn builds on
mhkp_v0_milp.build_model, so each group of constraints exists in exactly
one place and the three submodels cannot drift apart.


Load bearing, constraints (13a)-(13g)
-------------------------------------
Every item i resists a maximum load Lambda_i. The load it carries is "the
sum of the items' weights inside the cuboid space over it" (Section 3).
The indicator is

    lambda[i,k] = 1  iff item k lies OVER item i

which the paper defines by four conditions, all of which must hold:

    (13a)-(13b)  k sits at or above i's top face: z'_i <= z_k, written as
                 a big-M pair so that lambda is forced to 0 when k is below
                 (13b uses the +1 that makes "strictly below" exclusive)
    (13c)-(13d)  k's CoM lies within i's x-extent
    (13e)-(13f)  k's CoM lies within i's y-extent

    (13g)        sum_k lambda[i,k] * omega_k <= Lambda_i

Note what this does NOT say: it is not "k rests directly on i". An item
three layers up still counts against i as long as its CoM falls inside i's
footprint, which is the physically right reading - the load accumulates
down a column. That is why (13a) compares against i's TOP face rather than
requiring contact.

The CoM terms sum over bins, sum_j tau^x_k,j, because tau is zero for every
bin that does not hold k (constraints (12a), (12d), (12g)); the sum is
therefore k's CoM if it is packed and zero otherwise. This is exactly why
V2 needs V1: without the tau variables there is nothing to write (13c)-(13f)
against, which is also why the paper orders the submodels this way.

One consequence worth stating: (13c)-(13f) only constrain lambda when i and
k are in the SAME bin, because tau is bin-summed. Two items in different
bins can satisfy all four conditions vacuously, so (13g) is additionally
restricted here to same-bin pairs via S[i,k], V0's own "same bin"
indicator. Without that, an item could be charged for the weight of an item
it does not carry, which would be wrong in the safe direction but wrong
nonetheless.


Lambda_i
--------
Section 5, equation (18): the load bearing is three times the item's
MAXIMAL weight, Lambda_i = 3 * Omega_i, where Omega_i is the maximum of
equation (17) and not the realised weight omega_i. `derive_v2_data` follows
that, reusing mhkp_v1_milp.derive_v1_data for the weights and CoMs so that
V1 and V2 see identical data for a given instance and seed.


Usage
-----
    python3 mhkp_v2_milp.py MHKP/t8_n18/t8_n18_instance01.txt
    python3 mhkp_v2_milp.py --time-limit 60 MHKP/t8_n18/*.txt
    python3 mhkp_v2_milp.py --compare MHKP/t8_n18/t8_n18_instance01.txt
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import mhkp_instances as mi


def _load(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load("mhkp_v1_milp", "mhkp_v1_milp.py")
V0 = V1.V0

DEFAULT_ALPHA = V1.DEFAULT_ALPHA

# Equation (18): "we opt for simplicity and set the load bearing to be three
# times the maximal weight of the item".
LOAD_BEARING_FACTOR = 3.0


def derive_v2_data(bins, items, seed=1, alpha=DEFAULT_ALPHA,
                   factor=LOAD_BEARING_FACTOR):
    """
    Derive everything V2 needs: V1's weights, CoMs and pyramid, plus the
    per-item load-bearing limits Lambda.

    Returns (omega, kappa, varrho, xi, Lambda).

    Lambda_i = factor * Omega_i with Omega_i the MAXIMAL weight of equation
    (17), not the realised weight omega_i. The distinction matters: omega_i
    is drawn uniformly in [0.3, 1.0] * Omega_i, so using omega_i would make
    the limit 3x a random draw rather than 3x the item's capacity, and would
    couple an item's strength to how heavy it happens to be.
    """

    omega, kappa, varrho, xi = V1.derive_v1_data(bins, items, seed=seed,
                                                 alpha=alpha)

    max_item_volume = max(l * w * h for (l, w, h) in items)

    Lambda = []

    for (l, w, h) in items:
        volume = l * w * h

        # Equation (17), the same expression derive_v1_data uses for the
        # maximum before applying its 30%-100% draw.
        omega_max = ((volume / max_item_volume)
                     * V1.FT20_WEIGHT_LIMIT
                     * (max_item_volume / V1.FT20_VOLUME))

        Lambda.append(factor * omega_max)

    return omega, kappa, varrho, xi, Lambda


def build_model(bins, items, beta=1, time_limit=9.0, threads=None,
                verbose=False, mip_gap=None, omega=None, kappa=None,
                varrho=None, xi=None, Lambda=None, seed=1,
                alpha=DEFAULT_ALPHA, factor=LOAD_BEARING_FACTOR):
    """
    Build submodel V2: V1's constraints (1)-(12n) plus (13a)-(13g).

    Any of the data arrays may be supplied directly; those omitted are
    derived by `derive_v2_data`.
    """

    if (omega is None or kappa is None or varrho is None or xi is None
            or Lambda is None):
        d = derive_v2_data(bins, items, seed=seed, alpha=alpha, factor=factor)

        omega = d[0] if omega is None else omega
        kappa = d[1] if kappa is None else kappa
        varrho = d[2] if varrho is None else varrho
        xi = d[3] if xi is None else xi
        Lambda = d[4] if Lambda is None else Lambda

    # ---- V1: constraints (1)-(12n) ----
    model, v = V1.build_model(bins, items, beta=beta, time_limit=time_limit,
                              threads=threads, verbose=verbose,
                              mip_gap=mip_gap, omega=omega, kappa=kappa,
                              varrho=varrho, xi=xi, seed=seed, alpha=alpha)

    n = len(items)
    m = len(bins)

    I = range(n)
    J = range(m)

    x, y, z = v["x"], v["y"], v["z"]
    xp, yp, zp = v["xp"], v["yp"], v["zp"]
    tau_x, tau_y = v["tau_x"], v["tau_y"]
    S = v["S"]

    Lmax = max(b[0] for b in bins)
    Wmax = max(b[1] for b in bins)
    Hmax = max(b[2] for b in bins)

    # ---- V2 variable ----
    # lam[i, k] = 1 iff item k lies over item i and therefore loads it.
    lam = model.addVars(n, n, vtype=GRB.BINARY, name="lambda")

    for i in I:
        # (13): an item does not load itself.
        model.addConstr(lam[i, i] == 0, name=f"c13_self_{i}")

        for k in I:

            if i == k:
                continue

            # ---- (13a)-(13b): k is above i's top face ----
            # (13a) releases the bound when lam = 0; (13b) forces lam = 1
            # whenever k starts at or above i's top, the +1 making the
            # "below" case strict.
            #
            # (13b) is written by the paper for PACKED items. Taken
            # literally it also fires for unpacked ones: an unpacked item
            # has z_i = z'_i = 0, so the constraint reads
            # lam[i,k] * H >= z_k + 1, forcing lam[i,k] = 1 even though
            # neither item is in a bin - which then contradicts the
            # same-bin restriction below and makes the whole model
            # INFEASIBLE, including the empty packing. Releasing it by
            # S[i,k] restores the intended reading: the implication only
            # binds for two items that actually share a bin.
            model.addConstr(zp[i] <= z[k] + (1 - lam[i, k]) * Hmax,
                            name=f"c13a_{i}_{k}")
            model.addConstr(
                lam[i, k] * Hmax + zp[i] >= z[k] + 1 - Hmax * (1 - S[i, k]),
                name=f"c13b_{i}_{k}")

            # ---- (13c)-(13d): k's CoM within i's x-extent ----
            com_x_k = gp.quicksum(tau_x[k, j] for j in J)

            model.addConstr(
                (lam[i, k] - 1) * Wmax + beta * x[i] <= com_x_k,
                name=f"c13c_{i}_{k}")
            model.addConstr(
                com_x_k <= xp[i] + (1 - lam[i, k]) * Wmax,
                name=f"c13d_{i}_{k}")

            # ---- (13e)-(13f): k's CoM within i's y-extent ----
            com_y_k = gp.quicksum(tau_y[k, j] for j in J)

            model.addConstr(
                (lam[i, k] - 1) * Lmax + beta * y[i] <= com_y_k,
                name=f"c13e_{i}_{k}")
            model.addConstr(
                com_y_k <= yp[i] + (1 - lam[i, k]) * Lmax,
                name=f"c13f_{i}_{k}")

            # Only items sharing a bin can load one another. The paper's
            # (13c)-(13f) leave this implicit, because tau is summed over
            # bins and is zero for an unpacked item; making it explicit
            # stops an item in one bin being charged for weight in another.
            model.addConstr(lam[i, k] <= S[i, k], name=f"c13_samebin_{i}_{k}")

    # ---- (13g): the load on i may not exceed its capacity ----
    for i in I:
        model.addConstr(
            gp.quicksum(lam[i, k] * omega[k] for k in I if k != i)
            <= Lambda[i], name=f"c13g_{i}")

    v.update({"lam": lam, "Lambda": Lambda})

    return model, v


def solve(bins, items, beta=None, time_limit=9.0, threads=None,
          verbose=False, mip_gap=None, omega=None, kappa=None, varrho=None,
          xi=None, Lambda=None, seed=1, alpha=DEFAULT_ALPHA,
          factor=LOAD_BEARING_FACTOR):
    """Build and solve V2, returning a result dict shaped like V1's."""

    if beta is None:
        beta = mi.grid_beta(items)

    t0 = time.time()
    model, v = build_model(bins, items, beta=beta, time_limit=time_limit,
                           threads=threads, verbose=verbose, mip_gap=mip_gap,
                           omega=omega, kappa=kappa, varrho=varrho, xi=xi,
                           Lambda=Lambda, seed=seed, alpha=alpha,
                           factor=factor)
    build_time = time.time() - t0

    model.optimize()

    runtime = model.Runtime
    status = model.Status

    placements = []
    objective = None
    bound = None
    gap = None

    if model.SolCount > 0:
        objective = model.ObjVal

        try:
            bound = model.ObjBound
            gap = model.MIPGap
        except AttributeError:
            pass

        C, phi, R = v["C"], v["phi"], v["R"]

        for i in range(len(items)):
            for j in range(len(bins)):
                if C[i, j].X > 0.5:
                    l, w, h = items[i]
                    r = 0 if phi[0, i].X > 0.5 else 1
                    dx = l if r == 0 else w
                    dy = w if r == 0 else l
                    placements.append({
                        "item": i, "bin": j, "rotation": r,
                        "reflected": int(R[i].X > 0.5),
                        "x": int(round(beta * v["x"][i].X)),
                        "y": int(round(beta * v["y"][i].X)),
                        "z": int(round(v["z"][i].X)),
                        "dx": dx, "dy": dy, "dz": h,
                    })
                    break

    return {
        "objective": objective,
        "bound": bound,
        "gap": gap,
        "status": status,
        "status_name": {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                        GRB.INFEASIBLE: "INFEASIBLE",
                        GRB.INTERRUPTED: "INTERRUPTED"}.get(status,
                                                            str(status)),
        "runtime": runtime,
        "build_time": build_time,
        "n_packed": len(placements),
        "placements": placements,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
        "beta": beta,
        "omega": v["omega"],
        "kappa": v["kappa"],
        "varrho": v["varrho"],
        "xi": v["xi"],
        "Lambda": v["Lambda"],
    }


def check_load_bearing(bins, items, result, tol=1e-6):
    """
    Independently verify (13g) from the returned placements.

    Recomputes which items lie over which - by the paper's own definition,
    CoM inside the footprint and sitting at or above the top face - rather
    than reading the model's lambda variables, so a modelling error cannot
    verify itself.
    """

    omega = result["omega"]
    kappa = result["kappa"]
    Lambda = result["Lambda"]

    violations = []

    by_bin = {}
    for pl in result["placements"]:
        by_bin.setdefault(pl["bin"], []).append(pl)

    for j, placed in by_bin.items():

        for lower in placed:

            top = lower["z"] + lower["dz"]
            load = 0.0

            for upper in placed:

                if upper is lower:
                    continue

                # k must sit at or above i's top face.
                if upper["z"] < top - tol:
                    continue

                # k's CoM must fall within i's footprint.
                kx, ky, kz = kappa[upper["item"]]

                ox = kx if upper["rotation"] == 0 else ky
                oy = ky if upper["rotation"] == 0 else kx

                if upper.get("reflected"):
                    ox = upper["dx"] - ox
                    oy = upper["dy"] - oy

                cx = upper["x"] + ox
                cy = upper["y"] + oy

                if not (lower["x"] - tol <= cx <= lower["x"] + lower["dx"] + tol):
                    continue
                if not (lower["y"] - tol <= cy <= lower["y"] + lower["dy"] + tol):
                    continue

                load += omega[upper["item"]]

            cap = Lambda[lower["item"]]

            if load > cap + tol:
                violations.append(
                    f"bin {j}, item {lower['item']}: bears {load:.1f} "
                    f"> capacity {cap:.1f}")

    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Submodel V2 (stability + centre of mass + load "
                    "bearing) as a MILP.")

    parser.add_argument("instances", nargs="+")
    parser.add_argument("--time-limit", type=float, default=9.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--factor", type=float, default=LOAD_BEARING_FACTOR,
                        help=f"load bearing as a multiple of the maximal "
                             f"weight (default {LOAD_BEARING_FACTOR})")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compare", action="store_true",
                        help="also solve V0 and V1, to show what each "
                             "constraint group costs")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    header = (f"{'instance':<26}{'V2 obj':>9}{'gap%':>8}{'status':>12}"
              f"{'packed':>8}{'time':>7}{'CoM':>5}{'load':>6}")

    if args.compare:
        header += f"{'V1 obj':>9}{'V0 obj':>9}"

    print("=" * len(header))
    print(f"Submodel V2 (constraints (1)-(13g)) | {args.time_limit}s | "
          f"alpha={args.alpha}, Lambda={args.factor}x, seed={args.seed}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for spec in args.instances:

        bins, items = mi.read_instance(spec)

        res = solve(bins, items, time_limit=args.time_limit,
                    threads=args.threads, verbose=args.verbose,
                    mip_gap=args.mip_gap, seed=args.seed, alpha=args.alpha,
                    factor=args.factor)

        obj = "-" if res["objective"] is None else f"{res['objective']:.4f}"
        gap = "-" if res["gap"] is None else f"{res['gap'] * 100:.2f}"

        com_v = V1.check_com(bins, items, res)
        load_v = check_load_bearing(bins, items, res)

        row = (f"{Path(spec).stem:<26}{obj:>9}{gap:>8}"
               f"{res['status_name']:>12}{res['n_packed']:>8}"
               f"{res['runtime']:>7.2f}"
               f"{('ok' if not com_v else f'{len(com_v)}!'):>5}"
               f"{('ok' if not load_v else f'{len(load_v)}!'):>6}")

        if args.compare:
            r1 = V1.solve(bins, items, time_limit=args.time_limit,
                          seed=args.seed, alpha=args.alpha)
            r0 = V0.solve(bins, items, time_limit=args.time_limit)

            o1 = "-" if r1["objective"] is None else f"{r1['objective']:.4f}"
            o0 = "-" if r0["objective"] is None else f"{r0['objective']:.4f}"
            row += f"{o1:>9}{o0:>9}"

        print(row)

        for x in (com_v + load_v)[:3]:
            print(f"    {x}")


if __name__ == "__main__":
    main()
