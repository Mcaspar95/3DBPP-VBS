#!/usr/bin/env python3
"""
Submodel V1 of Deplano et al. (2019) as a MILP, solved with Gurobi
==================================================================

V1 = V0 plus the centre-of-mass (CoM) distribution constraints, i.e.
constraints (1)-(12n) of

    Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
    "A mixed-integer linear model for the multiple heterogeneous knapsack
     problem with realistic container loading constraints and bins' priority"
    Intl. Trans. in Op. Res. 28(6), 3244-3275.

V0 (mhkp_v0_milp.py) is constraints (1)-(10r): geometry, the priority
objective and static stability. V1 adds (11a)-(12n):

    (11a)-(11h)  the CoM of an item after rotation AND reflection
    (12a)-(12i)  that CoM expressed in the coordinate frame of its bin
    (12j)-(12n)  the global CoM of each bin confined to a pyramidal region

V2 would further add load bearing, (13a)-(13g); it is not implemented here.

This file BUILDS ON mhkp_v0_milp.build_model rather than restating the
shared constraints, so (1)-(10r) exist in exactly one place and V0 and V1
cannot drift apart.


What V1 needs that V0 does not
------------------------------
V0 reads only geometry. V1 additionally needs, per item i:

    omega_i    the item's weight                  (constraints (12j)-(12n))
    kappa_i    the CoM offset (kx, ky, kz) from
               the item's front-bottom-left corner  (constraints (11a)-(12i))

and, per bin j, the pyramid that confines the loaded CoM:

    varrho_j   the pyramid vertex; the paper suggests the bin centre in x
               and y, and HALF THE HEIGHT in z (Section 3, above (12a))
    xi_j       the base radius, precomputed as alpha * min(L_j, W_j) with
               alpha in (0, 1]; the paper suggests alpha = 0.8

The instance files under MHKP/ carry geometry and bin priority only, so
these are DERIVED from the instance, deterministically from a seed, by
`derive_v1_data` below, following the paper's own generation rules
(Section 5, equations (16)-(18)). The derivation is reproducible: the same
instance and seed always give the same weights and CoMs, so a V1 result can
be regenerated exactly. `derive_v1_data` is separate from the model so that
real weight/CoM data, when available, can be passed in directly instead.


The pyramid, and why (12j)-(12m) are linear
-------------------------------------------
The admissible region for a bin's global CoM is a pyramid: wide at the
floor, narrowing to the vertex varrho_j. Writing M_j = sum_p omega_p C_p,j
for the loaded weight of bin j, the paper's (12j) reads

    sum_i tau^x_i,j omega_i  <=  varrho^x_j M_j
                                 + (varrho^z_j M_j - sum_i tau^z_i,j omega_i)
                                   / varrho^z_j * xi_j

Every term is linear in the decision variables: M_j is a linear expression
in C, the tau are variables, and varrho^z_j and xi_j are CONSTANTS, so the
division is by a constant, not by a variable. The bracketed term is the
weighted height headroom below the vertex, so the permitted horizontal
deviation shrinks to zero as the CoM rises to varrho^z_j, which is what
makes the region a pyramid rather than a box. (12n) caps the CoM height at
the vertex, which is what keeps that headroom non-negative.

An empty bin has M_j = 0 and all tau_i,j = 0 by (12a), (12d), (12g), so
(12j)-(12n) reduce to 0 <= 0 and are vacuous. No special case is needed.


Axis convention
---------------
As in mhkp_v0_milp.py, this file follows THE PAPER: z is vertical, x and y
are the floor plane, with x bounded by W and y by L. Note that the paper's
(12a) bounds tau^x by W_j and (12d) bounds tau^y by L_j, consistent with
(7a)-(7b); this file keeps that pairing.


Usage
-----
    python3 mhkp_v1_milp.py MHKP/t8_n18/t8_n18_instance01.txt
    python3 mhkp_v1_milp.py --time-limit 60 --alpha 0.8 MHKP/t8_n18/*.txt
    python3 mhkp_v1_milp.py --compare-v0 MHKP/t8_n18/t8_n18_instance01.txt
"""

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import mhkp_instances as mi


def _load_v0():
    """Load mhkp_v0_milp.py, whose shared constraints V1 builds on."""

    path = Path(__file__).resolve().parent / "mhkp_v0_milp.py"

    spec = importlib.util.spec_from_file_location("mhkp_v0_milp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mhkp_v0_milp"] = module
    spec.loader.exec_module(module)

    return module


V0 = _load_v0()


# Default pyramid base as a fraction of the smaller floor dimension
# (Section 3, above (12a): "By default, we suggest alpha = 0.8").
DEFAULT_ALPHA = 0.8

# Reference values of equations (16)-(17): a 40' ISO container and a 20'
# weight limit. Only their RATIO to the instance's own volumes matters, so
# these scale the derived weights without affecting feasibility.
FT40_VOLUME = 12.192 * 2.438 * 2.591      # m^3, 40' ISO internal
FT40_WEIGHT_LIMIT = 26_700.0              # kg, 40' payload
FT20_VOLUME = 6.058 * 2.438 * 2.591       # m^3, 20' ISO internal
FT20_WEIGHT_LIMIT = 28_200.0              # kg, 20' payload


def derive_v1_data(bins, items, seed=1, alpha=DEFAULT_ALPHA):
    """
    Derive the weights and CoM offsets that V1 needs but the instance files
    do not carry, following the paper's Section 5.

    Equation (17): an item's MAXIMAL weight is proportional to its volume.
    The paper then draws the actual weight uniformly between 30% and 100%
    of that maximum ("The item weight has been generated randomly with a
    uniform distribution between 30% and 100% of the items' maximal
    weight").

    CoM: "generated in a random point under 3/5 of the height and between
    20% and 80% of the other dimensions", measured from the item's
    front-bottom-left corner in its UNROTATED frame, which is what
    constraints (11a)-(11h) then rotate and reflect.

    Returns (omega, kappa, varrho, xi):
        omega[i]   weight of item i
        kappa[i]   (kx, ky, kz) CoM offset of item i, unrotated
        varrho[j]  (rx, ry, rz) pyramid vertex of bin j
        xi[j]      pyramid base radius of bin j

    Deterministic in `seed`: the same instance always yields the same data.
    """

    rng = random.Random(seed)

    max_item_volume = max(l * w * h for (l, w, h) in items)

    omega = []
    kappa = []

    for (l, w, h) in items:

        # (17): maximal weight proportional to volume.
        volume = l * w * h

        omega_max = ((volume / max_item_volume)
                     * FT20_WEIGHT_LIMIT
                     * (max_item_volume / FT20_VOLUME))

        # Section 5: actual weight is 30%-100% of the maximum.
        omega.append(rng.uniform(0.3, 1.0) * omega_max)

        # Section 5: CoM under 3/5 of the height, 20%-80% of the others.
        kappa.append((
            rng.uniform(0.2, 0.8) * l,
            rng.uniform(0.2, 0.8) * w,
            rng.uniform(0.0, 0.6) * h,
        ))

    varrho = []
    xi = []

    for (L, W, H, _priority) in bins:

        # Section 3: the suggested vertex is the centre in x and y and half
        # the height in z. x is bounded by W and y by L, per (7a)-(7b).
        varrho.append((0.5 * W, 0.5 * L, 0.5 * H))

        # xi_j = alpha * L_j if W_j > L_j else alpha * W_j, i.e. alpha
        # times the SMALLER floor dimension.
        xi.append(alpha * (L if W > L else W))

    return omega, kappa, varrho, xi


def build_model(bins, items, beta=1, time_limit=9.0, threads=None,
                verbose=False, mip_gap=None, omega=None, kappa=None,
                varrho=None, xi=None, seed=1, alpha=DEFAULT_ALPHA):
    """
    Build submodel V1: V0's constraints (1)-(10r) plus (11a)-(12n).

    `omega`, `kappa`, `varrho` and `xi` may be supplied directly; any that
    are omitted are derived by `derive_v1_data`.

    Returns (model, variables) with V0's variables plus V1's.
    """

    # ---- V0: constraints (1)-(10r) ----
    model, v = V0.build_model(bins, items, beta=beta, time_limit=time_limit,
                              threads=threads, verbose=verbose,
                              mip_gap=mip_gap)

    if omega is None or kappa is None or varrho is None or xi is None:
        d_omega, d_kappa, d_varrho, d_xi = derive_v1_data(
            bins, items, seed=seed, alpha=alpha)

        omega = d_omega if omega is None else omega
        kappa = d_kappa if kappa is None else kappa
        varrho = d_varrho if varrho is None else varrho
        xi = d_xi if xi is None else xi

    n = len(items)
    m = len(bins)

    I = range(n)
    J = range(m)

    C, phi = v["C"], v["phi"]
    x, y, z = v["x"], v["y"], v["z"]
    xp, yp = v["xp"], v["yp"]

    Lmax = max(b[0] for b in bins)
    Wmax = max(b[1] for b in bins)
    Hmax = max(b[2] for b in bins)

    # ---- V1 variables ----

    # R[i] = 1 if item i is reflected. V0 has no reflection variable: none
    # of (1)-(10r) can observe a reflection, because reflecting a cuboid
    # leaves its occupied region unchanged. It becomes meaningful only here,
    # where the CoM sits asymmetrically inside that region.
    R = model.addVars(n, vtype=GRB.BINARY, name="R")

    # upsilon: the item's CoM offset after rotation and reflection, still
    # relative to the item's own corner. (11a)-(11h).
    ups_x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Wmax,
                          name="upsilon_x")
    ups_y = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=Lmax,
                          name="upsilon_y")

    # tau: that CoM in the frame of bin j, and zero when the item is not in
    # bin j. (12a)-(12i).
    tau_x = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0, ub=Wmax,
                          name="tau_x")
    tau_y = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0, ub=Lmax,
                          name="tau_y")
    tau_z = model.addVars(n, m, vtype=GRB.CONTINUOUS, lb=0, ub=Hmax,
                          name="tau_z")

    # ---- (11a)-(11d): CoM along x, under rotation and reflection ----
    # Rotation 0 keeps kx on the x-axis; rotation 1 swaps in ky. Reflection
    # mirrors the offset within the item's own extent, x' - offset. Each
    # pair is released by a big-M on the branch that does not apply.
    for i in I:
        kx, ky, _kz = kappa[i]

        rotated_x = kx * phi[0, i] + ky * phi[1, i]

        model.addConstr(ups_x[i] <= rotated_x + R[i] * Wmax, name=f"c11a_{i}")
        model.addConstr(ups_x[i] >= rotated_x - R[i] * Wmax, name=f"c11b_{i}")

        model.addConstr(
            ups_x[i] <= xp[i] - beta * x[i] - rotated_x + (1 - R[i]) * Wmax,
            name=f"c11c_{i}")
        model.addConstr(
            ups_x[i] >= xp[i] - beta * x[i] - rotated_x - (1 - R[i]) * Wmax,
            name=f"c11d_{i}")

    # ---- (11e)-(11h): CoM along y ----
    for i in I:
        kx, ky, _kz = kappa[i]

        rotated_y = ky * phi[0, i] + kx * phi[1, i]

        model.addConstr(ups_y[i] <= rotated_y + R[i] * Lmax, name=f"c11e_{i}")
        model.addConstr(ups_y[i] >= rotated_y - R[i] * Lmax, name=f"c11f_{i}")

        model.addConstr(
            ups_y[i] <= yp[i] - beta * y[i] - rotated_y + (1 - R[i]) * Lmax,
            name=f"c11g_{i}")
        model.addConstr(
            ups_y[i] >= yp[i] - beta * y[i] - rotated_y - (1 - R[i]) * Lmax,
            name=f"c11h_{i}")

    # An unpacked item has no CoM to place; pinning R[i] removes a free
    # binary that would otherwise leave the model with symmetric solutions.
    for i in I:
        model.addConstr(R[i] <= gp.quicksum(C[i, j] for j in J),
                        name=f"c11_unpacked_{i}")

    # ---- (12a)-(12i): the CoM in the bin's frame ----
    # Each triple forces tau = (position + offset) when the item is in bin
    # j, and tau = 0 otherwise, so the sums in (12j)-(12n) range only over
    # the items actually loaded into that bin.
    for i in I:
        _kx, _ky, kz = kappa[i]

        for j in J:
            L, W, H, _p = bins[j]

            model.addConstr(tau_x[i, j] <= W * C[i, j], name=f"c12a_{i}_{j}")
            model.addConstr(tau_x[i, j] <= ups_x[i] + beta * x[i],
                            name=f"c12b_{i}_{j}")
            model.addConstr(
                tau_x[i, j] >= ups_x[i] + beta * x[i] - W * (1 - C[i, j]),
                name=f"c12c_{i}_{j}")

            model.addConstr(tau_y[i, j] <= L * C[i, j], name=f"c12d_{i}_{j}")
            model.addConstr(tau_y[i, j] <= ups_y[i] + beta * y[i],
                            name=f"c12e_{i}_{j}")
            model.addConstr(
                tau_y[i, j] >= ups_y[i] + beta * y[i] - L * (1 - C[i, j]),
                name=f"c12f_{i}_{j}")

            # (12g)-(12i): z needs no rotation term, there being no rotation
            # about the vertical axis.
            model.addConstr(tau_z[i, j] <= H * C[i, j], name=f"c12g_{i}_{j}")
            model.addConstr(tau_z[i, j] <= kz + z[i], name=f"c12h_{i}_{j}")
            model.addConstr(tau_z[i, j] >= kz + z[i] - H * (1 - C[i, j]),
                            name=f"c12i_{i}_{j}")

    # ---- (12j)-(12n): the global CoM of each bin inside its pyramid ----
    for j in J:
        rx, ry, rz = varrho[j]

        # A zero-height vertex would divide by zero and, being degenerate,
        # admits no CoM above the floor at all.
        if rz <= 0:
            raise ValueError(
                f"bin {j}: pyramid vertex height varrho^z must be positive")

        # Loaded weight of the bin, linear in C.
        mass = gp.quicksum(omega[p] * C[p, j] for p in I)

        sum_tau_x = gp.quicksum(tau_x[i, j] * omega[i] for i in I)
        sum_tau_y = gp.quicksum(tau_y[i, j] * omega[i] for i in I)
        sum_tau_z = gp.quicksum(tau_z[i, j] * omega[i] for i in I)

        # The weighted headroom below the vertex, scaled to the base radius.
        # Linear: rz and xi[j] are constants.
        slack = (rz * mass - sum_tau_z) * (xi[j] / rz)

        model.addConstr(sum_tau_x <= rx * mass + slack, name=f"c12j_{j}")
        model.addConstr(sum_tau_x >= rx * mass - slack, name=f"c12k_{j}")
        model.addConstr(sum_tau_y <= ry * mass + slack, name=f"c12l_{j}")
        model.addConstr(sum_tau_y >= ry * mass - slack, name=f"c12m_{j}")

        # (12n): the CoM may not rise above the vertex. This is also what
        # keeps `slack` non-negative, so the pyramid never inverts.
        model.addConstr(sum_tau_z <= rz * mass, name=f"c12n_{j}")

    v.update({"R": R, "ups_x": ups_x, "ups_y": ups_y,
              "tau_x": tau_x, "tau_y": tau_y, "tau_z": tau_z,
              "omega": omega, "kappa": kappa, "varrho": varrho, "xi": xi})

    return model, v


def solve(bins, items, beta=None, time_limit=9.0, threads=None,
          verbose=False, mip_gap=None, omega=None, kappa=None,
          varrho=None, xi=None, seed=1, alpha=DEFAULT_ALPHA):
    """Build and solve V1, returning a result dict shaped like V0's."""

    if beta is None:
        beta = mi.grid_beta(items)

    t0 = time.time()
    model, v = build_model(bins, items, beta=beta, time_limit=time_limit,
                           threads=threads, verbose=verbose, mip_gap=mip_gap,
                           omega=omega, kappa=kappa, varrho=varrho, xi=xi,
                           seed=seed, alpha=alpha)
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
    }


def check_com(bins, items, result, tol=1e-6):
    """
    Independently verify that each loaded bin's CoM lies inside its pyramid.

    Recomputes the CoM from the returned placements rather than from the
    model's own tau variables, so that a modelling error in (11)-(12) cannot
    verify itself.

    Returns a list of violations; empty means every bin is within its region.
    """

    omega = result["omega"]
    kappa = result["kappa"]
    varrho = result["varrho"]
    xi = result["xi"]

    violations = []

    by_bin = {}
    for pl in result["placements"]:
        by_bin.setdefault(pl["bin"], []).append(pl)

    for j, placed in by_bin.items():

        rx, ry, rz = varrho[j]

        mass = 0.0
        cx = cy = cz = 0.0

        for pl in placed:
            i = pl["item"]
            kx, ky, kz = kappa[i]
            w = omega[i]

            # The offset within the item, after rotation and reflection.
            ox = kx if pl["rotation"] == 0 else ky
            oy = ky if pl["rotation"] == 0 else kx

            if pl["reflected"]:
                ox = pl["dx"] - ox
                oy = pl["dy"] - oy

            mass += w
            cx += w * (pl["x"] + ox)
            cy += w * (pl["y"] + oy)
            cz += w * (pl["z"] + kz)

        if mass <= 0:
            continue

        cx /= mass
        cy /= mass
        cz /= mass

        if cz > rz + tol:
            violations.append(
                f"bin {j}: CoM height {cz:.3f} exceeds vertex {rz:.3f}")
            continue

        # Permitted horizontal deviation at this height.
        allowed = (rz - cz) / rz * xi[j]

        if abs(cx - rx) > allowed + tol:
            violations.append(
                f"bin {j}: CoM x deviation {abs(cx - rx):.3f} "
                f"exceeds {allowed:.3f} at height {cz:.3f}")

        if abs(cy - ry) > allowed + tol:
            violations.append(
                f"bin {j}: CoM y deviation {abs(cy - ry):.3f} "
                f"exceeds {allowed:.3f} at height {cz:.3f}")

    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Submodel V1 (stability + centre of mass) as a MILP.")

    parser.add_argument("instances", nargs="+",
                        help="instance files to solve")
    parser.add_argument("--time-limit", type=float, default=9.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"pyramid base fraction (default "
                             f"{DEFAULT_ALPHA})")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for the derived weights and CoMs")
    parser.add_argument("--compare-v0", action="store_true",
                        help="also solve V0, to show the cost of the CoM "
                             "constraints")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    header = (f"{'instance':<28}{'V1 obj':>9}{'gap%':>8}{'status':>12}"
              f"{'packed':>8}{'time':>8}{'CoM':>6}")

    if args.compare_v0:
        header += f"{'V0 obj':>9}{'V0 pk':>7}"

    print("=" * len(header))
    print(f"Submodel V1 (constraints (1)-(12n)) | {args.time_limit}s limit "
          f"| alpha={args.alpha}, seed={args.seed}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for spec in args.instances:

        bins, items = mi.read_instance(spec)

        res = solve(bins, items, time_limit=args.time_limit,
                    threads=args.threads, verbose=args.verbose,
                    mip_gap=args.mip_gap, seed=args.seed, alpha=args.alpha)

        obj = "-" if res["objective"] is None else f"{res['objective']:.4f}"
        gap = "-" if res["gap"] is None else f"{res['gap'] * 100:.2f}"

        viol = check_com(bins, items, res)
        com = "ok" if not viol else f"{len(viol)}!"

        row = (f"{Path(spec).stem:<28}{obj:>9}{gap:>8}"
               f"{res['status_name']:>12}{res['n_packed']:>8}"
               f"{res['runtime']:>8.2f}{com:>6}")

        if args.compare_v0:
            r0 = V0.solve(bins, items, time_limit=args.time_limit,
                          threads=args.threads, mip_gap=args.mip_gap)
            o0 = "-" if r0["objective"] is None else f"{r0['objective']:.4f}"
            row += f"{o0:>9}{r0['n_packed']:>7}"

        print(row)

        for v in viol[:3]:
            print(f"    CoM violation: {v}")


if __name__ == "__main__":
    main()
