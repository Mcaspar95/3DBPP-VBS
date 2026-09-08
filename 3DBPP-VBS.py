#!/usr/bin/env python3
"""
3DBPP-VBS: Three-dimensional Bin Packing Problem with Variable Box Sizes
=======================================================================

MILP model of Section 3.2 in:

    Xu, X., Wu, B., Ma, Z., Yu, Y. (2026)
    "The three-dimensional bin packing problem with variable box size"
    Transportation Research Part E 214, 105038.

Single bin of size L (length, Z-axis), W (width, X-axis), H (height, Y-axis).
Each box may be compressed vertically; its length and width then expand so
that the original volume and the length-to-width aspect ratio are preserved.


Box size variation (Section 3.1, formula (1))
---------------------------------------------
Three variants k = 1, 2, 3 with compression ratios s^k = 1, 0.85, 0.8:

    h_i^k = s^k * h_i
    l_i^k = sqrt(1 / s^k) * l_i
    w_i^k = sqrt(1 / s^k) * w_i

Sizes are rounded to integers. The paper's Table 2 (500 x 250 x 200) lists
variant 2 as 425/271/216 and variant 3 as 400/279/223, which corresponds to
truncation of l and w, so truncation is used here.

Together with the 6 orthogonal orientations this yields |C_i| = 18 feasible
configurations per box.


MILP model (Section 3.2)
------------------------
Decision variables:

    t_{i,c} = 1 if box i is packed in configuration c            (binary)
    p_i     = 1 if box i is packed into the bin                  (binary)
    b^1..b^6_{ij} relative-position indicators for each pair i<j (binary)
    x_i, y_i, z_i  coordinates of the left-back-bottom corner    (continuous)

    max  sum_i sum_c w_{i,c} h_{i,c} l_{i,c} t_{i,c} / (L W H)         (2)

    s.t. sum_c t_{i,c} = p_i                                  for all i (3)

         x_i + sum_c w_{i,c} t_{i,c} <= W                     for all i (4)
         y_i + sum_c h_{i,c} t_{i,c} <= H                     for all i (5)
         z_i + sum_c l_{i,c} t_{i,c} <= L                     for all i (6)

         x_i + sum_c w_{i,c} t_{i,c} <= x_j + W (1 - b^1_{ij})  i != j  (7)
         x_j + sum_c w_{j,c} t_{j,c} <= x_i + W (1 - b^2_{ij})  i != j  (8)
         y_i + sum_c h_{i,c} t_{i,c} <= y_j + H (1 - b^3_{ij})  i != j  (9)
         y_j + sum_c h_{j,c} t_{j,c} <= y_i + H (1 - b^4_{ij})  i != j (10)
         z_i + sum_c l_{i,c} t_{i,c} <= z_j + L (1 - b^5_{ij})  i != j (11)
         z_j + sum_c l_{j,c} t_{j,c} <= z_i + L (1 - b^6_{ij})  i != j (12)

         b^1 + b^2 + b^3 + b^4 + b^5 + b^6 >= p_i + p_j - 1    i != j (13)

         x_i, y_i, z_i >= 0                                            (14)
         all t, p, b binary                                            (15)

Note on constraint (13): the paper prints "b^5_{ij} + b^5_{ij}" as the last
two terms, which is a typo -- the disjunction needs all six indicators, so
b^6_{ij} is used here (otherwise the "in front of" relation could never
separate a pair).

Constraints (7)-(13) are stated for all ordered pairs i != j in the paper.
Every unordered pair {i, j} therefore appears twice, generating the same six
inequalities with the roles of i and j exchanged. This implementation posts
them once per unordered pair i < j, which is equivalent and halves the model
size.


Instances
---------
BR instances (Bischoff & Ratcliff) from the "BR" folder, in the simple text
format

    n W H L
    w_1 h_1 l_1
    ...

with one line per physical box. The original JSON files (0BR_orig/) are read
too when passed directly; there the item types carry a "Demand" count that is
expanded into individual boxes, and the "Stock", "Cost", "Value", "DemandMax"
and "C1_*" fields are not part of the model and are ignored.


Environment
-----------
The project's dependencies (gurobipy, matplotlib, numpy) live in the .venv
directory next to this file; see requirements.txt. The script re-executes
itself with .venv/bin/python when started with another interpreter, so all
of these work without activating the environment first:

    ./3DBPP-VBS.py
    python3 3DBPP-VBS.py
    .venv/bin/python 3DBPP-VBS.py

To recreate the environment from scratch:

    python3.11 -m venv .venv
    .venv/bin/pip install -r requirements.txt


Usage
-----
    python3 3DBPP-VBS.py               # Martello class 1, instance 1
    python3 3DBPP-VBS.py --classic     # fixed sizes (plain 3DBPP)
    python3 3DBPP-VBS.py --both        # 3DBPP vs 3DBPP-VBS, with Diff
    python3 3DBPP-VBS.py --instance Martello/class_5/class5_n15_instance1.txt

    # BR instances are much larger: cap the box count to stay tractable
    python3 3DBPP-VBS.py --instance BR/BR1/1.txt --max-boxes 20
    python3 3DBPP-VBS.py --instance BR/BR1/1.txt --time-limit 600 --plot

    # a batch over several instances, summarized and written to CSV
    python3 3DBPP-VBS.py --batch Martello/class_*/[!.]*instance1.txt \
            --both --time-limit 60 --csv results.csv

The MILP has O(n^2) binaries and becomes intractable quickly; the paper itself
does not run Gurobi on full BR instances ("Gurobi is not employed due to the
computational limits imposed by the large instance sizes", Section 5.4.2).
Use --max-boxes to take a prefix of the box list for tractable runs.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


# Run under the project's .venv even when started with a different
# interpreter, so that gurobipy and matplotlib are always available.
# Re-executes this script once with .venv/bin/python and then continues
# there; the guard variable prevents an endless re-exec loop.
def _activate_venv():

    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"

    if not venv_python.exists():
        return

    if Path(sys.executable).resolve() == venv_python.resolve():
        return

    if os.environ.get("_3DBPP_VBS_VENV") == "1":
        return

    os.environ["_3DBPP_VBS_VENV"] = "1"

    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_activate_venv()


import gurobipy as gp
from gurobipy import GRB


# Compression ratios s^k for the three variants (Section 3.1).
COMPRESSION_RATIOS = [1.0, 0.85, 0.8]


# ----------------------------------------------------------------------
# Instance loading
# ----------------------------------------------------------------------

def load_br_instance(path):
    """
    Read one BR JSON instance.

    Returns (L, W, H, boxes) where boxes is a list of (l, w, h) triples,
    one entry per physical box (item types are expanded by their Demand).
    The "Stock", "Cost", "Demand", "Value", "DemandMax" and "C1_*" fields
    are not part of the 3DBPP-VBS model and are ignored.
    """

    with open(path) as f:
        data = json.load(f)

    obj = data["Objects"][0]
    L = int(obj["Length"])
    W = int(obj["Depth"])
    H = int(obj["Height"])

    boxes = []

    for item in data["Items"]:
        l = int(item["Length"])
        w = int(item["Depth"])
        h = int(item["Height"])

        for _ in range(int(item["Demand"])):
            boxes.append((l, w, h))

    return L, W, H, boxes


def load_martello_instance(path):
    """
    Read one instance in the simple martello.py text format:

        n W H D
        w_1 h_1 d_1
        ...

    Returns (L, W, H, boxes) with boxes as (l, w, h) triples.
    """

    with open(path) as f:
        tokens = f.read().split()

    n = int(tokens[0])
    W, H, L = int(tokens[1]), int(tokens[2]), int(tokens[3])

    boxes = []

    for i in range(n):
        w, h, l = (int(v) for v in tokens[4 + 3 * i: 7 + 3 * i])
        boxes.append((l, w, h))

    return L, W, H, boxes


def load_instance(path):
    """
    Load an instance, picking the reader from the file extension:
    ".json" for the original BR files, anything else for the simple
    text format used by the converted BR/ and Martello/ instances.
    """

    path = Path(path)

    if path.suffix.lower() == ".json":
        return load_br_instance(path)

    return load_martello_instance(path)


# ----------------------------------------------------------------------
# Configurations: 3 variants x 6 orientations = 18 per box
# ----------------------------------------------------------------------

def box_variants(l, w, h, classic=False):
    """
    The three size variants of a box, formula (1).

    Returns a list of (variant_index, l_k, w_k, h_k). With classic=True only
    the original size is returned, which reduces the model to the standard
    3DBPP used as a benchmark in the paper.
    """

    ratios = [1.0] if classic else COMPRESSION_RATIOS

    variants = []

    for k, s in enumerate(ratios, start=1):
        factor = math.sqrt(1.0 / s)

        h_k = int(s * h)
        l_k = int(factor * l)
        w_k = int(factor * w)

        variants.append((k, l_k, w_k, h_k))

    return variants


def box_configurations(l, w, h, classic=False):
    """
    All feasible configurations C_i of a box: every size variant combined
    with the 6 orthogonal orientations (Fig. 1).

    A configuration is a dict with the edge lengths along the axes:
        wc -> X-axis (bounded by W)
        hc -> Y-axis (bounded by H)
        lc -> Z-axis (bounded by L)

    Duplicate configurations (which arise when a box has equal edges) are
    removed, since they only add symmetric branches to the search tree.
    """

    configurations = []
    seen = set()

    for k, l_k, w_k, h_k in box_variants(l, w, h, classic=classic):

        # The 6 orthogonal orientations: which original edge goes to which
        # axis, as (Z-axis, X-axis, Y-axis) = (length, width, height).
        orientations = [
            (l_k, w_k, h_k),
            (l_k, h_k, w_k),
            (w_k, l_k, h_k),
            (w_k, h_k, l_k),
            (h_k, l_k, w_k),
            (h_k, w_k, l_k),
        ]

        for o, (lc, wc, hc) in enumerate(orientations, start=1):

            key = (lc, wc, hc)
            if key in seen:
                continue
            seen.add(key)

            configurations.append({
                "variant": k,
                "orientation": o,
                "lc": lc,   # Z
                "wc": wc,   # X
                "hc": hc,   # Y
            })

    return configurations


# ----------------------------------------------------------------------
# MILP model (Section 3.2)
# ----------------------------------------------------------------------

def solve_3dbpp_vbs(
    L,
    W,
    H,
    boxes,
    classic=False,
    time_limit=3600.0,
    mip_gap=None,
    threads=None,
    verbose=True,
):
    """
    Build and solve the MILP of Section 3.2.

    Returns a result dict with the utilization, the packed boxes and their
    coordinates/configurations, and solver statistics.
    """

    n = len(boxes)
    B = range(n)

    # Feasible configurations per box. A configuration that cannot fit into
    # the empty bin is dropped right away -- it could never satisfy (4)-(6).
    configs = []

    for (l, w, h) in boxes:
        feasible = [
            c for c in box_configurations(l, w, h, classic=classic)
            if c["wc"] <= W and c["hc"] <= H and c["lc"] <= L
        ]
        configs.append(feasible)

    if verbose:
        env = gp.Env()
    else:
        # Also suppresses the license banner printed when the environment
        # is started, which would otherwise interleave with batch output.
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

    model = gp.Model("3DBPP-VBS", env=env)

    if not verbose:
        model.setParam("OutputFlag", 0)

    model.setParam("TimeLimit", time_limit)

    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)

    if threads is not None:
        model.setParam("Threads", threads)

    # --- decision variables -------------------------------------------

    # t[i, c] = 1 if box i is packed in configuration c
    t = {}
    for i in B:
        for c in range(len(configs[i])):
            t[i, c] = model.addVar(vtype=GRB.BINARY, name=f"t_{i}_{c}")

    # p[i] = 1 if box i is packed into the bin
    p = model.addVars(n, vtype=GRB.BINARY, name="p")

    # x, y, z: left-back-bottom corner of box i, constraint (14)
    x = model.addVars(n, lb=0.0, ub=W, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(n, lb=0.0, ub=H, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(n, lb=0.0, ub=L, vtype=GRB.CONTINUOUS, name="z")

    # Relative-position indicators, posted once per unordered pair i < j.
    pairs = [(i, j) for i in B for j in B if i < j]

    b1 = model.addVars(pairs, vtype=GRB.BINARY, name="b1")  # i left of j
    b2 = model.addVars(pairs, vtype=GRB.BINARY, name="b2")  # i right of j
    b3 = model.addVars(pairs, vtype=GRB.BINARY, name="b3")  # i below j
    b4 = model.addVars(pairs, vtype=GRB.BINARY, name="b4")  # i above j
    b5 = model.addVars(pairs, vtype=GRB.BINARY, name="b5")  # i behind j
    b6 = model.addVars(pairs, vtype=GRB.BINARY, name="b6")  # i in front of j

    # Edge length of box i along each axis, as a linear expression in t.
    wx = {i: gp.quicksum(configs[i][c]["wc"] * t[i, c]
                         for c in range(len(configs[i]))) for i in B}
    hy = {i: gp.quicksum(configs[i][c]["hc"] * t[i, c]
                         for c in range(len(configs[i]))) for i in B}
    lz = {i: gp.quicksum(configs[i][c]["lc"] * t[i, c]
                         for c in range(len(configs[i]))) for i in B}

    # --- objective (2): maximize space utilization ---------------------

    bin_volume = L * W * H

    model.setObjective(
        gp.quicksum(
            configs[i][c]["wc"] * configs[i][c]["hc"] * configs[i][c]["lc"]
            * t[i, c]
            for i in B for c in range(len(configs[i]))
        ) / bin_volume,
        GRB.MAXIMIZE,
    )

    # --- constraint (3): at most one configuration per box -------------

    for i in B:
        model.addConstr(
            gp.quicksum(t[i, c] for c in range(len(configs[i]))) == p[i],
            name=f"assign_{i}",
        )

    # --- constraints (4)-(6): stay inside the bin ----------------------

    for i in B:
        model.addConstr(x[i] + wx[i] <= W, name=f"bin_x_{i}")
        model.addConstr(y[i] + hy[i] <= H, name=f"bin_y_{i}")
        model.addConstr(z[i] + lz[i] <= L, name=f"bin_z_{i}")

    # --- constraints (7)-(13): pairwise non-overlap --------------------

    for (i, j) in pairs:
        model.addConstr(x[i] + wx[i] <= x[j] + W * (1 - b1[i, j]),
                        name=f"sep_x1_{i}_{j}")
        model.addConstr(x[j] + wx[j] <= x[i] + W * (1 - b2[i, j]),
                        name=f"sep_x2_{i}_{j}")
        model.addConstr(y[i] + hy[i] <= y[j] + H * (1 - b3[i, j]),
                        name=f"sep_y1_{i}_{j}")
        model.addConstr(y[j] + hy[j] <= y[i] + H * (1 - b4[i, j]),
                        name=f"sep_y2_{i}_{j}")
        model.addConstr(z[i] + lz[i] <= z[j] + L * (1 - b5[i, j]),
                        name=f"sep_z1_{i}_{j}")
        model.addConstr(z[j] + lz[j] <= z[i] + L * (1 - b6[i, j]),
                        name=f"sep_z2_{i}_{j}")

        # (13): if both boxes are packed, at least one separation applies.
        model.addConstr(
            b1[i, j] + b2[i, j] + b3[i, j]
            + b4[i, j] + b5[i, j] + b6[i, j] >= p[i] + p[j] - 1,
            name=f"nonoverlap_{i}_{j}",
        )

    # --- solve ---------------------------------------------------------

    start = time.time()
    model.optimize()
    runtime = time.time() - start

    result = {
        "status": model.Status,
        "runtime": runtime,
        "num_boxes": n,
        "bin": (L, W, H),
        "bin_volume": bin_volume,
        # Volume of all boxes at their original size, as a fraction of the
        # bin. Utilization can never exceed this; when it is below 100% the
        # instance simply does not contain enough material to fill the bin.
        "volume_ratio": sum(l * w * h for (l, w, h) in boxes) / bin_volume,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        "placements": [],
        "utilization": 0.0,
        "packed_volume": 0,
        "mip_gap": None,
        "bound": None,
    }

    if model.SolCount == 0:
        return result

    result["utilization"] = model.ObjVal
    result["bound"] = model.ObjBound

    try:
        result["mip_gap"] = model.MIPGap
    except (AttributeError, gp.GurobiError):
        pass

    packed_volume = 0

    for i in B:
        if p[i].X < 0.5:
            continue

        for c in range(len(configs[i])):
            if t[i, c].X < 0.5:
                continue

            cfg = configs[i][c]

            packed_volume += cfg["wc"] * cfg["hc"] * cfg["lc"]

            result["placements"].append({
                "box": i,
                "original": boxes[i],
                "variant": cfg["variant"],
                "orientation": cfg["orientation"],
                # coordinates of the left-back-bottom corner
                "x": x[i].X,
                "y": y[i].X,
                "z": z[i].X,
                # edge lengths along X, Y, Z
                "wc": cfg["wc"],
                "hc": cfg["hc"],
                "lc": cfg["lc"],
            })
            break

    result["packed_volume"] = packed_volume

    return result


# ----------------------------------------------------------------------
# Verification and reporting
# ----------------------------------------------------------------------

def verify_solution(result, tol=1e-6):
    """
    Independent geometric check of the returned packing: every box inside
    the bin, and no two boxes overlapping. Returns a list of violations.
    """

    L, W, H = result["bin"]
    placements = result["placements"]

    violations = []

    for pl in placements:
        if pl["x"] + pl["wc"] > W + tol:
            violations.append(f"box {pl['box']} exceeds W")
        if pl["y"] + pl["hc"] > H + tol:
            violations.append(f"box {pl['box']} exceeds H")
        if pl["z"] + pl["lc"] > L + tol:
            violations.append(f"box {pl['box']} exceeds L")
        if min(pl["x"], pl["y"], pl["z"]) < -tol:
            violations.append(f"box {pl['box']} has a negative coordinate")

    for a in range(len(placements)):
        for b in range(a + 1, len(placements)):
            i, j = placements[a], placements[b]

            overlap = (
                i["x"] + i["wc"] > j["x"] + tol
                and j["x"] + j["wc"] > i["x"] + tol
                and i["y"] + i["hc"] > j["y"] + tol
                and j["y"] + j["hc"] > i["y"] + tol
                and i["z"] + i["lc"] > j["z"] + tol
                and j["z"] + j["lc"] > i["z"] + tol
            )

            if overlap:
                violations.append(
                    f"boxes {i['box']} and {j['box']} overlap"
                )

    return violations


def print_report(result, instance_name, classic):
    """Print a summary of one solved instance."""

    L, W, H = result["bin"]

    status_names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }

    problem = "3DBPP (fixed sizes)" if classic else "3DBPP-VBS"

    print()
    print("=" * 66)
    print(f"  {problem}   instance: {instance_name}")
    print("=" * 66)
    print(f"  bin (L x W x H)   : {L} x {W} x {H}")
    print(f"  boxes             : {result['num_boxes']}")

    # Utilization is capped by the total box volume: with only a few boxes
    # the bin cannot be filled no matter how well they are placed.
    ratio = result["volume_ratio"] * 100

    note = "  <- caps the utilization" if ratio < 100 else ""

    print(f"  total box volume  : {ratio:.2f} % of bin{note}")
    print(f"  model             : {result['num_vars']} vars, "
          f"{result['num_constrs']} constrs")
    print(f"  status            : "
          f"{status_names.get(result['status'], result['status'])}")
    print(f"  runtime           : {result['runtime']:.2f} s")

    if result["placements"] or result["utilization"]:
        print(f"  packed boxes      : "
              f"{len(result['placements'])} / {result['num_boxes']}")
        print(f"  space utilization : {result['utilization'] * 100:.2f} %")

        # Share of the available box volume that was actually placed. This
        # separates "packed badly" from "there were not enough boxes".
        if result["volume_ratio"] > 0:
            packed_share = result["utilization"] / result["volume_ratio"] * 100
            print(f"  of available vol. : {packed_share:.2f} %")

        if result["bound"] is not None:
            print(f"  upper bound       : {result['bound'] * 100:.2f} %")
        if result["mip_gap"] is not None:
            print(f"  MIP gap           : {result['mip_gap'] * 100:.2f} %")

        violations = verify_solution(result)

        if violations:
            print(f"  CHECK             : {len(violations)} violation(s)")
            for v in violations[:5]:
                print(f"      - {v}")
        else:
            print("  check             : feasible "
                  "(inside bin, no overlaps)")

        print()
        print("  box  variant  orient   (x, y, z)          "
              "edges X/Y/Z      original l/w/h")
        print("  " + "-" * 74)

        for pl in sorted(result["placements"], key=lambda d: d["box"]):
            l0, w0, h0 = pl["original"]
            coord = f"({pl['x']:.0f}, {pl['y']:.0f}, {pl['z']:.0f})"
            edges = f"{pl['wc']}/{pl['hc']}/{pl['lc']}"
            print(f"  {pl['box']:>3}  {pl['variant']:>7}  {pl['orientation']:>6}"
                  f"   {coord:<18} {edges:<16} {l0}/{w0}/{h0}")

    else:
        print("  no feasible solution found")

    print()


def plot_solution(result, instance_name, out_path):
    """Render the packing as a 3D plot (requires matplotlib)."""

    try:
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  plot skipped        : matplotlib is not installed "
              "(pip3 install matplotlib)")
        return

    L, W, H = result["bin"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab20")

    for idx, pl in enumerate(result["placements"]):
        x0, y0, z0 = pl["x"], pl["y"], pl["z"]
        dx, dy, dz = pl["wc"], pl["hc"], pl["lc"]

        corners = [
            (x0, y0, z0), (x0 + dx, y0, z0),
            (x0 + dx, y0 + dy, z0), (x0, y0 + dy, z0),
            (x0, y0, z0 + dz), (x0 + dx, y0, z0 + dz),
            (x0 + dx, y0 + dy, z0 + dz), (x0, y0 + dy, z0 + dz),
        ]

        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[1], corners[2], corners[6], corners[5]],
            [corners[0], corners[3], corners[7], corners[4]],
        ]

        ax.add_collection3d(Poly3DCollection(
            faces,
            facecolors=cmap(idx % 20),
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        ))

    ax.set_xlim(0, W)
    ax.set_ylim(0, L)
    ax.set_zlim(0, H)
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Z (length)")
    ax.set_zlabel("Y (height)")
    ax.set_box_aspect((W, L, H))

    ax.set_title(
        f"{instance_name}\n"
        f"{len(result['placements'])}/{result['num_boxes']} boxes, "
        f"utilization {result['utilization'] * 100:.2f}%"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  plot saved to     : {out_path}")


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MILP model (Section 3.2) for the 3DBPP-VBS, "
                    "solved on the Martello and BR instances."
    )

    parser.add_argument(
        "--instance",
        default="Martello/class_1/class1_n10_instance1.txt",  # BR/BR1/1.txt
        help="path to an instance: a BR/Martello text file, or one of the "
             "original BR *.json files "
             "(default: Martello/class_1/class1_n10_instance1.txt)",
    )
    parser.add_argument(
        "--max-boxes", type=int, default=0,
        help="use only the first k boxes of the instance; the MILP grows "
             "quadratically and is intractable for full BR instances, so "
             "reduce this for those (default: 0 = all boxes)",
    )
    parser.add_argument(
        "--classic", action="store_true",
        help="solve the standard 3DBPP with fixed box sizes "
             "(variant 1 only) instead of the 3DBPP-VBS",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="solve both the 3DBPP and the 3DBPP-VBS and report the "
             "improvement Diff of formula (18)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=300,
        help="Gurobi time limit in seconds",
    )
    parser.add_argument(
        "--mip-gap", type=float, default=None,
        help="Gurobi relative MIP gap",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="number of threads for Gurobi",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the Gurobi solver log",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="save a 3D plot of the packing next to the script",
    )
    parser.add_argument(
        "--batch", nargs="+", default=None,
        help="solve several instances in sequence, e.g. "
             "--batch BR/BR1/*.txt; results are summarized in a table "
             "and written to the file given by --csv",
    )
    parser.add_argument(
        "--csv", default=None,
        help="write the batch results to this CSV file",
    )

    args = parser.parse_args()

    if args.batch:
        run_batch(args)
        return

    path = Path(args.instance)
    L, W, H, boxes = load_instance(path)

    if args.max_boxes and args.max_boxes > 0:
        boxes = boxes[:args.max_boxes]

    instance_name = str(path)

    if args.both:
        runs = [(True, "3DBPP"), (False, "3DBPP-VBS")]
    else:
        runs = [(args.classic, None)]

    results = {}

    for classic, _label in runs:

        result = solve_3dbpp_vbs(
            L, W, H, boxes,
            classic=classic,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            verbose=not args.quiet,
        )

        print_report(result, instance_name, classic)

        results["3DBPP" if classic else "3DBPP-VBS"] = result

        if args.plot and result["placements"]:
            suffix = "3dbpp" if classic else "3dbpp_vbs"
            out = Path(f"{path.stem}_{suffix}.png")
            plot_solution(result, instance_name, out)

    if args.both:
        r1 = results["3DBPP"]["utilization"]
        r2 = results["3DBPP-VBS"]["utilization"]

        print("=" * 66)
        print("  Comparison (formula (18))")
        print("=" * 66)
        print(f"  3DBPP     utilization : {r1 * 100:.2f} %")
        print(f"  3DBPP-VBS utilization : {r2 * 100:.2f} %")

        if r1 > 0:
            print(f"  Diff                  : {(r2 - r1) / r1 * 100:.2f} %")

        print()


def run_batch(args):
    """
    Solve a list of instances in sequence and summarize the results.

    With --both, each instance is solved as 3DBPP and as 3DBPP-VBS and the
    improvement Diff of formula (18) is reported per instance and on average.
    """

    import csv

    rows = []

    header = (
        f"  {'instance':<24} {'n':>4} {'packed':>7} "
        f"{'util%':>8} {'gap%':>7} {'time(s)':>9}"
    )

    if args.both:
        header = (
            f"  {'instance':<24} {'n':>4} "
            f"{'3DBPP%':>8} {'VBS%':>8} {'Diff%':>8} {'time(s)':>9}"
        )

    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for spec in args.batch:

        path = Path(spec)

        if not path.exists():
            print(f"  {spec}: not found")
            continue

        L, W, H, boxes = load_instance(path)

        if args.max_boxes and args.max_boxes > 0:
            boxes = boxes[:args.max_boxes]

        if args.both:
            modes = [True, False]
        else:
            modes = [args.classic]

        solved = {}

        for classic in modes:
            solved[classic] = solve_3dbpp_vbs(
                L, W, H, boxes,
                classic=classic,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                threads=args.threads,
                verbose=False,
            )

        name = f"{path.parent.name}/{path.stem}"

        # Keep the CSV rows identifiable: the full name goes to the file,
        # only the printed table is shortened. Instances of one class share
        # a long prefix and differ at the end, so drop characters from the
        # front rather than the back.
        full_name = name

        if len(name) > 24:
            name = "~" + name[-23:]

        if args.both:
            r1 = solved[True]["utilization"]
            r2 = solved[False]["utilization"]
            diff = (r2 - r1) / r1 * 100 if r1 > 0 else float("nan")
            total_time = solved[True]["runtime"] + solved[False]["runtime"]

            print(f"  {name:<24} {len(boxes):>4} "
                  f"{r1 * 100:>8.2f} {r2 * 100:>8.2f} {diff:>8.2f} "
                  f"{total_time:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": len(boxes),
                "util_3dbpp": r1 * 100,
                "util_3dbpp_vbs": r2 * 100,
                "diff_pct": diff,
                "time_s": total_time,
            })

        else:
            res = solved[args.classic]
            gap = res["mip_gap"] * 100 if res["mip_gap"] is not None else \
                float("nan")

            print(f"  {name:<24} {len(boxes):>4} "
                  f"{len(res['placements']):>7} "
                  f"{res['utilization'] * 100:>8.2f} {gap:>7.2f} "
                  f"{res['runtime']:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": len(boxes),
                "packed": len(res["placements"]),
                "util_pct": res["utilization"] * 100,
                "gap_pct": gap,
                "time_s": res["runtime"],
            })

    if rows:
        print("=" * len(header))

        if args.both:
            avg1 = sum(r["util_3dbpp"] for r in rows) / len(rows)
            avg2 = sum(r["util_3dbpp_vbs"] for r in rows) / len(rows)
            avg_d = sum(r["diff_pct"] for r in rows) / len(rows)

            print(f"  {'average':<24} {'':>4} "
                  f"{avg1:>8.2f} {avg2:>8.2f} {avg_d:>8.2f}")
        else:
            avg = sum(r["util_pct"] for r in rows) / len(rows)
            print(f"  {'average':<24} {'':>4} {'':>7} {avg:>8.2f}")

        print()

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"  results written to {args.csv}")
        print()


if __name__ == "__main__":
    main()
