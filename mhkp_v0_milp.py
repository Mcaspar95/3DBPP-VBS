#!/usr/bin/env python3
"""
Submodel V0 of Deplano et al. (2019) as a MILP, solved with Gurobi
==================================================================

Exact counterpart to MHKP-V0-SA.py, implementing submodel V0 of

    Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
    "A mixed-integer linear model for the multiple heterogeneous knapsack
     problem with realistic container loading constraints and bins' priority"
    Intl. Trans. in Op. Res. 28(6), 3244-3275.

V0 is constraints (1)-(10r): the priority objective, the geometry, and static
stability. It excludes the centre-of-mass distribution (V1, constraints
(11a)-(12n)) and load bearing (V2, (13a)-(13g)), which need weight and CoM data
that V0 never reads.

This is the column the paper's Table 8 actually reports. It exists here so that
our SA can be measured against the exact method on identical instances, the way
the paper measures its own heuristic.

Axis convention
---------------
The paper's model uses z as the VERTICAL axis: constraint (6c) reads
z'_i - z_i = h_i, and the stability constraints (10a)-(10c) put item i on top of
k when z_i = z'_k. Its x/y are the floor plane, with x bounded by W and y by L
(constraints (7a), (7b)).

That is the opposite of the convention in MHKP-V0-SA.py, where y is vertical.
This file follows THE PAPER, so that each constraint here can be read directly
against the paper's numbering. `solution_to_sa_frame` converts a solved packing
into the SA file's frame so that both can be checked by the same verifier.

Model size
----------
The paper gives V0 as 9n^2 + nm + m + 9n variables and
n^2m^2 - nm^2 - n^2m + 2nm + 25n^2 + 3m - 15n constraints, for n items and
m bins. The n^2m^2 term comes from constraint (8d), which is posted over every
pair of items AND every pair of distinct bins; with one bin it vanishes.
"""

import argparse
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import mhkp_instances as mi


def build_model(bins, items, beta=1, time_limit=9.0, threads=None,
                verbose=False, mip_gap=None):
    """
    Build submodel V0 for the given instance.

    `bins` are (L, W, H, priority) and `items` are (l, w, h), i.e. exactly what
    mhkp_instances.read_instance returns.

    Returns (model, variables) with the variables needed to read a solution.
    """

    n = len(items)
    m = len(bins)

    I = range(n)
    J = range(m)
    O = (0, 1)              # Table 2: two rotations about the vertical axis

    # Big-M values: the largest bin along each axis (the paper precomputes
    # L, W, H as the maxima over the bin set, just above constraint (7a)).
    Lmax = max(b[0] for b in bins)
    Wmax = max(b[1] for b in bins)
    Hmax = max(b[2] for b in bins)

    model = gp.Model("MHKP_V0")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("TimeLimit", time_limit)
    if threads is not None:
        model.setParam("Threads", threads)
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    # ---- Variables (Table 5) ----
    C = model.addVars(n, m, vtype=GRB.BINARY, name="C")      # item i in bin j
    Z = model.addVars(m, vtype=GRB.BINARY, name="Z")         # bin j opened
    phi = model.addVars(len(O), n, vtype=GRB.BINARY, name="phi")

    # Grid coordinates: beta*x and beta*y are the actual positions (Definition
    # 4). z is free, being the vertical axis determined by what is below.
    x = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=Wmax // beta, name="x")
    y = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=Lmax // beta, name="y")
    z = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Hmax, name="z")

    xp = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Wmax, name="xprime")
    yp = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Lmax, name="yprime")
    zp = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Hmax, name="zprime")

    S = model.addVars(n, n, vtype=GRB.BINARY, name="S")      # same bin
    xpos = model.addVars(n, n, vtype=GRB.BINARY, name="xpos")
    ypos = model.addVars(n, n, vtype=GRB.BINARY, name="ypos")
    zpos = model.addVars(n, n, vtype=GRB.BINARY, name="zpos")

    # Stability: theta[i, k] = item i rests on item k; theta_ground[i] = on the
    # bin floor. pi[i, k, c] = corner c of item i is on top of item k.
    theta = model.addVars(n, n, vtype=GRB.BINARY, name="theta")
    theta_g = model.addVars(n, vtype=GRB.BINARY, name="theta_ground")
    pi = model.addVars(n, n, 4, vtype=GRB.BINARY, name="pi")

    # ---- Objective (1): minimise priority-weighted wasted space ----
    model.setObjective(
        gp.quicksum(
            bins[j][3] * (bins[j][0] * bins[j][1] * bins[j][2]
                          - gp.quicksum(items[i][0] * items[i][1] * items[i][2]
                                        * C[i, j] for i in I))
            for j in J),
        GRB.MINIMIZE)

    # ---- (3a)-(3d): assignment and rotation ----
    for i in I:
        for j in J:
            model.addConstr(C[i, j] <= Z[j], name=f"c3a_{i}_{j}")

    for i in I:
        model.addConstr(gp.quicksum(C[i, j] for j in J) <= 1, name=f"c3b_{i}")

    for j in J:
        model.addConstr(gp.quicksum(C[i, j] for i in I) >= Z[j], name=f"c3c_{j}")

    for i in I:
        model.addConstr(gp.quicksum(phi[o, i] for o in O)
                        == gp.quicksum(C[i, j] for j in J), name=f"c3d_{i}")

    # ---- (5): volume capacity. (4) is the weight limit, which V0 has no data
    # for and which the paper's V0 experiments therefore cannot bind.
    for j in J:
        model.addConstr(
            gp.quicksum(items[i][0] * items[i][1] * items[i][2] * C[i, j]
                        for i in I)
            <= bins[j][0] * bins[j][1] * bins[j][2], name=f"c5_{j}")

    # ---- (6a)-(6c): item extents under the chosen rotation ----
    # Rotation 0 puts length along x and width along y; rotation 1 swaps them.
    # The height is invariant, so (6c) has no rotation term.
    for i in I:
        l, w, h = items[i]
        model.addConstr(xp[i] - beta * x[i] == l * phi[0, i] + w * phi[1, i],
                        name=f"c6a_{i}")
        model.addConstr(yp[i] - beta * y[i] == l * phi[1, i] + w * phi[0, i],
                        name=f"c6b_{i}")
        model.addConstr(zp[i] - z[i] == h * gp.quicksum(C[i, j] for j in J),
                        name=f"c6c_{i}")

    # ---- (7a)-(7c): containment within the assigned bin ----
    # Written as one constraint per (item, bin) with a big-M released when the
    # item is not in that bin, which is tighter than the paper's form and
    # equivalent: an unassigned item is unconstrained either way.
    for i in I:
        for j in J:
            model.addConstr(xp[i] <= bins[j][1] + Wmax * (1 - C[i, j]),
                            name=f"c7a_{i}_{j}")
            model.addConstr(yp[i] <= bins[j][0] + Lmax * (1 - C[i, j]),
                            name=f"c7b_{i}_{j}")
            model.addConstr(zp[i] <= bins[j][2] + Hmax * (1 - C[i, j]),
                            name=f"c7c_{i}_{j}")

    # ---- (8a)-(8d): S[i,k] = 1 iff i and k share a bin ----
    for i in I:
        for k in I:
            if i == k:
                continue
            model.addConstr(S[i, k] <= gp.quicksum(C[i, j] for j in J),
                            name=f"c8a_{i}_{k}")
            model.addConstr(S[i, k] <= gp.quicksum(C[k, j] for j in J),
                            name=f"c8b_{i}_{k}")
            for j in J:
                model.addConstr(S[i, k] >= C[k, j] + C[i, j] - 1,
                                name=f"c8c_{i}_{k}_{j}")
            # (8d): i and k cannot be "together" while sitting in two
            # different bins. Only needed when there is more than one bin.
            for j in J:
                for lbin in J:
                    if lbin == j:
                        continue
                    model.addConstr(S[i, k] + C[i, j] + C[k, lbin] <= 2,
                                    name=f"c8d_{i}_{k}_{j}_{lbin}")

    # ---- (9a)-(9e): non-overlap ----
    # Posted once per unordered pair: the disjunction (9a) already carries both
    # directions through xpos[i,k] and xpos[k,i], so posting it for k < i as
    # well would only duplicate it.
    for i in I:
        for k in I:
            if k <= i:
                continue
            model.addConstr(
                xpos[i, k] + ypos[i, k] + zpos[i, k]
                + xpos[k, i] + ypos[k, i] + zpos[k, i] >= S[i, k],
                name=f"c9a_{i}_{k}")

    for i in I:
        for k in I:
            if i == k:
                continue
            # (9b): item k is before item i along x.
            model.addConstr(xp[k] <= beta * x[i] + (1 - xpos[i, k]) * Wmax,
                            name=f"c9b_{i}_{k}")
            model.addConstr(yp[k] <= beta * y[i] + (1 - ypos[i, k]) * Lmax,
                            name=f"c9c_{i}_{k}")
            model.addConstr(zp[k] <= z[i] + (1 - zpos[i, k]) * Hmax,
                            name=f"c9d_{i}_{k}")

    for i in I:
        model.addConstr(xpos[i, i] == 0, name=f"c9e_x_{i}")
        model.addConstr(ypos[i, i] == 0, name=f"c9e_y_{i}")
        model.addConstr(zpos[i, i] == 0, name=f"c9e_z_{i}")

    # ---- (10a)-(10r): static stability ----
    # (10a): an item on the ground has z = 0.
    for i in I:
        model.addConstr(z[i] <= Hmax * (1 - theta_g[i]), name=f"c10a_{i}")

    # (10b)-(10c): resting on k means exact surface contact, z_i = z'_k.
    for i in I:
        for k in I:
            if i == k:
                continue
            model.addConstr(z[i] <= Hmax * (1 - theta[i, k]) + zp[k],
                            name=f"c10b_{i}_{k}")
            model.addConstr(z[i] >= Hmax * (theta[i, k] - 1) + zp[k],
                            name=f"c10c_{i}_{k}")

    # (10d)-(10k): which bottom corners of i lie over item k.
    # pi[i,k,0..1] are the two x-side corners, pi[i,k,2..3] the two y-side.
    for i in I:
        for k in I:
            if i == k:
                continue
            model.addConstr(
                beta * x[i] + Wmax * (theta[i, k] - 1) <= beta * x[k]
                + (1 - pi[i, k, 0]) * Wmax, name=f"c10d_{i}_{k}")
            model.addConstr(
                beta * x[k] + Wmax * (pi[i, k, 0] - 1) <= xp[i]
                + (1 - theta[i, k]) * Wmax, name=f"c10e_{i}_{k}")
            model.addConstr(
                beta * x[i] + Wmax * (theta[i, k] - 1) <= xp[k]
                + (1 - pi[i, k, 1]) * Wmax, name=f"c10f_{i}_{k}")
            model.addConstr(
                xp[k] + Wmax * (pi[i, k, 1] - 1) <= xp[i]
                + (1 - theta[i, k]) * Wmax, name=f"c10g_{i}_{k}")
            model.addConstr(
                beta * y[i] + Lmax * (theta[i, k] - 1) <= beta * y[k]
                + (1 - pi[i, k, 2]) * Lmax, name=f"c10h_{i}_{k}")
            model.addConstr(
                beta * y[k] + Lmax * (pi[i, k, 2] - 1) <= yp[i]
                + (1 - theta[i, k]) * Lmax, name=f"c10i_{i}_{k}")
            model.addConstr(
                beta * y[i] + Lmax * (theta[i, k] - 1) <= yp[k]
                + (1 - pi[i, k, 3]) * Lmax, name=f"c10j_{i}_{k}")
            model.addConstr(
                yp[k] + Lmax * (pi[i, k, 3] - 1) <= yp[i]
                + (1 - theta[i, k]) * Lmax, name=f"c10k_{i}_{k}")

            # Minimum contact. Constraints (10d)-(10k) are non-strict, so on
            # integer coordinates they are satisfied by boxes that merely TOUCH
            # edge to edge with zero overlapping area - which supports nothing.
            # The paper has no minimum-contact requirement and so admits such
            # "support"; observed in practice (an item resting on a neighbour
            # with y-overlap exactly 0). These two constraints require at least
            # one unit of overlap on each axis when theta is set, which is the
            # evident intent of "on the top surface of".
            model.addConstr(
                beta * x[i] + 1 <= xp[k] + Wmax * (1 - theta[i, k]),
                name=f"c10contact_x1_{i}_{k}")
            model.addConstr(
                beta * x[k] + 1 <= xp[i] + Wmax * (1 - theta[i, k]),
                name=f"c10contact_x2_{i}_{k}")
            model.addConstr(
                beta * y[i] + 1 <= yp[k] + Lmax * (1 - theta[i, k]),
                name=f"c10contact_y1_{i}_{k}")
            model.addConstr(
                beta * y[k] + 1 <= yp[i] + Lmax * (1 - theta[i, k]),
                name=f"c10contact_y2_{i}_{k}")

            # (10l)-(10o): at least one corner from each axis pair, and no
            # corner indicator set unless i really rests on k.
            model.addConstr(pi[i, k, 0] + pi[i, k, 1] >= theta[i, k],
                            name=f"c10l_{i}_{k}")
            model.addConstr(pi[i, k, 2] + pi[i, k, 3] >= theta[i, k],
                            name=f"c10m_{i}_{k}")
            model.addConstr(pi[i, k, 0] + pi[i, k, 1] <= 2 * theta[i, k],
                            name=f"c10n_{i}_{k}")
            model.addConstr(pi[i, k, 2] + pi[i, k, 3] <= 2 * theta[i, k],
                            name=f"c10o_{i}_{k}")

            # (10p): only items in the same bin can support one another.
            model.addConstr(theta[i, k] <= S[i, k], name=f"c10p_{i}_{k}")

    # (10q): every packed item is supported, by the ground or by some item.
    for i in I:
        model.addConstr(
            theta_g[i] + gp.quicksum(theta[i, k] for k in I if k != i)
            >= gp.quicksum(C[i, j] for j in J), name=f"c10q_{i}")

    # (10r): no self-support.
    for i in I:
        model.addConstr(theta[i, i] == 0, name=f"c10r_t_{i}")
        for c in range(4):
            model.addConstr(pi[i, i, c] == 0, name=f"c10r_p_{i}_{c}")

    # S is exposed because V2's load bearing needs it: (13g) must charge an
    # item only for weight in its OWN bin, and S[i,k] is already the
    # "same bin" indicator built for (10p).
    return model, {"C": C, "Z": Z, "phi": phi, "x": x, "y": y, "z": z,
                   "xp": xp, "yp": yp, "zp": zp, "S": S, "beta": beta}


def solve(bins, items, beta=None, time_limit=9.0, threads=None,
          verbose=False, mip_gap=None):
    """Build and solve V0, returning a result dict."""

    if beta is None:
        beta = mi.grid_beta(items)

    t0 = time.time()
    model, v = build_model(bins, items, beta=beta, time_limit=time_limit,
                           threads=threads, verbose=verbose, mip_gap=mip_gap)
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

        C, phi = v["C"], v["phi"]
        for i in range(len(items)):
            for j in range(len(bins)):
                if C[i, j].X > 0.5:
                    l, w, h = items[i]
                    r = 0 if phi[0, i].X > 0.5 else 1
                    dx = l if r == 0 else w        # along x (bounded by W)
                    dy = w if r == 0 else l        # along y (bounded by L)
                    placements.append({
                        "item": i, "bin": j, "rotation": r,
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
                        GRB.INTERRUPTED: "INTERRUPTED"}.get(status, str(status)),
        "runtime": runtime,
        "build_time": build_time,
        "n_packed": len(placements),
        "placements": placements,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
        "beta": beta,
    }


def solution_to_sa_frame(placements):
    """
    Convert a solved packing into MHKP-V0-SA.py's frame.

    The model uses z as the vertical axis and (x, y) as the floor; the SA file
    uses y as the vertical axis and (x, z) as the floor. Mapping the model's
    (x, y, z) to the SA file's (x, z, y) therefore lets the SA verifier check
    this packing, which is the point: one independent checker for both solvers.
    """

    out = []

    for p in placements:
        out.append({
            "item": p["item"], "bin": p["bin"], "rotation": p["rotation"],
            "x": p["x"], "y": p["z"], "z": p["y"],
            "dx": p["dx"], "dy": p["dz"], "dz": p["dy"],
        })

    return out


def main():

    parser = argparse.ArgumentParser(
        description="Submodel V0 of Deplano et al. (2019), solved with Gurobi.")

    parser.add_argument("instances", nargs="+")
    parser.add_argument("--time-limit", type=float, default=9.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    for spec in args.instances:

        bins, items = mi.read_instance(spec)

        res = solve(bins, items, time_limit=args.time_limit,
                    threads=args.threads, verbose=args.verbose,
                    mip_gap=args.mip_gap)

        obj = "none" if res["objective"] is None else f"{res['objective']:.4f}"
        gap = "-" if res["gap"] is None else f"{res['gap']*100:.2f}%"

        print(f"{Path(spec).stem:<26} obj {obj:>8}  gap {gap:>8}  "
              f"packed {res['n_packed']}/{len(items)}  "
              f"{res['status_name']:<11} {res['runtime']:.2f}s  "
              f"({res['n_vars']} vars, {res['n_constrs']} constrs)")


if __name__ == "__main__":
    main()
