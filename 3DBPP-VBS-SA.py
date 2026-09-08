#!/usr/bin/env python3
# Simulated Annealing for the 3DBPP-VBS — random single-neighbour variant
# =========================================================================
# Adapted from 3dp-cptp-SA.py (3DP-CPTP, routing + packing) to the problem of
#
#     Xu, X., Wu, B., Ma, Z., Yu, Y. (2026)
#     "The three-dimensional bin packing problem with variable box size"
#     Transportation Research Part E 214, 105038.
#
# The MILP of Section 3.2 is in 3DBPP-VBS.py; this file is the heuristic
# counterpart, in the role the paper's BSA plays.
#
# What carried over from 3dp-cptp-SA.py
# -------------------------------------
# The SA machinery is unchanged in spirit:
#
#     1. pick ONE move operator uniformly at random (by weight)
#     2. sample ONE random neighbour from that operator
#     3. accept or reject by the Metropolis criterion
#     4. cool down once per iteration, regardless of the outcome
#
#   - Improving moves are always accepted
#   - Worsening moves are accepted with probability exp(delta / T)
#   - Temperature decreases geometrically: T *= ALPHA each iteration
#   - Reheating + diversification after prolonged stagnation (REHEAT_THRESHOLD)
#   - Operators drawn without replacement, so a failed draw does not waste
#     the whole iteration (sample_random_move)
#
# What had to change
# ------------------
# The 3DP-CPTP is a routing problem in which packing appears only as a
# yes/no feasibility check per vehicle. The 3DBPP-VBS has no routing at all:
# one bin, and the decision is which boxes to load, in which orientation, and
# in which size variant, maximizing space utilization. So:
#
#   * Solution representation. Routes/served/unserved are replaced by a
#     LOADING SEQUENCE: a permutation of the boxes plus, per box, a chosen
#     orientation and size variant. A deterministic placement heuristic turns
#     that sequence into an actual packing; boxes that do not fit are simply
#     left out. This is the standard encoding for bin packing under SA and
#     makes every solution feasible by construction, so no move can ever
#     produce an invalid packing.
#
#   * Objective. Maximize packed volume / bin volume (formula (2)), instead
#     of profit minus travel cost.
#
#   * Move operators. relocate/swap/2-opt act on the loading sequence and are
#     kept. insert/remove made no sense (a box is not "served"), so they are
#     replaced by CHANGE_ORIENTATION and CHANGE_VARIANT — the latter is the
#     operator specific to the VBS mechanism, switching a box between its
#     three compression variants (Section 3.1).
#
#   * Packing evaluation. 3dp-cptp-SA.py calls Gurobi for every feasibility
#     check. That is affordable when a check is rare and a route holds a
#     handful of items; here EVERY iteration needs a full packing of up to
#     ~150 boxes, and an SA run does 10^4-10^6 iterations. A Gurobi call per
#     iteration would cap the run at a few hundred iterations. The placement
#     is therefore done by a deterministic extreme-point heuristic (Crainic
#     et al. 2008, as cited in the paper's Section 2.1), which packs a
#     sequence in milliseconds. Gurobi is not used in this file.
#
#   * Delta evaluation. In the routing code a move's delta could be computed
#     locally from arc costs. A packing is not decomposable that way: moving
#     one box in the sequence changes where everything after it lands. Each
#     candidate is therefore fully re-packed and the delta measured. This is
#     why the placement heuristic has to be fast.
#
# Usage
# -----
#     python3 3DBPP-VBS-SA.py                      # Martello class 1, inst. 1
#     python3 3DBPP-VBS-SA.py --classic            # fixed sizes (plain 3DBPP)
#     python3 3DBPP-VBS-SA.py --both               # 3DBPP vs 3DBPP-VBS, Diff
#     python3 3DBPP-VBS-SA.py --instance BR/BR1/1.txt --time-limit 60
#     python3 3DBPP-VBS-SA.py --batch Martello/class_1/*.txt --both --csv r.csv
#
# The environment is the shared project .venv; see 3DBPP-VBS.py.

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path


# Run under the project's .venv even when started with a different
# interpreter, mirroring 3DBPP-VBS.py.
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


# Instance loading and the configuration model are shared with the MILP so
# that both solvers are guaranteed to see exactly the same instances and the
# same box variants. 3DBPP-VBS.py is not a valid module name (it starts with
# a digit and contains dashes), so it is loaded by path.
def _load_model_module():

    import importlib.util

    path = Path(__file__).resolve().parent / "3DBPP-VBS.py"

    spec = importlib.util.spec_from_file_location("dbpp_vbs_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_model = _load_model_module()

load_instance = _model.load_instance
box_variants = _model.box_variants
COMPRESSION_RATIOS = _model.COMPRESSION_RATIOS


# -------------------------
# Parameters
# -------------------------
T_INIT = 0.02              # initial temperature
ALPHA = 0.9995             # geometric cooling factor (T *= ALPHA each iteration)
T_MIN = 1e-6               # minimum temperature before stopping / reheating
MAX_ITERATIONS = 50000000  # maximum SA iterations
TIME_LIMIT = 60            # total wall-clock seconds

# NOTE on the temperature scale being far below the 3DP-CPTP values
# (T_INIT=100 there): the objective here is a UTILIZATION RATIO in [0, 1],
# not a profit in the hundreds. A typical worsening move costs delta ~ -0.01,
# so exp(delta / T) with T=100 is exp(-0.0001) ~ 1.0 — every move would be
# accepted and SA would degrade into a random walk. T_INIT=0.02 puts the
# initial acceptance of such a move near exp(-0.5) ~ 60%, which is the
# regime the schedule is supposed to start in. Temperature must always be
# scaled to the objective; this is the single most important change when
# porting an SA between problems.

# Temperature restored on reheating, as a fraction of T_INIT. As in the
# source file, a reheat has to RESET the temperature rather than scale the
# annealed one, which by then is close to T_MIN.
REHEAT_FRACTION = 0.25
T_REHEAT = REHEAT_FRACTION * T_INIT

# Iterations for one full geometric sweep T_INIT -> T_MIN at the current
# ALPHA (~20k for ALPHA=0.9995).
COOLING_SWEEP_ITERS = math.log(T_MIN / T_INIT) / math.log(ALPHA)

# Reheat only after a meaningful fraction of a full sweep without a new
# global best, so that annealing is never reset before it can cool.
REHEAT_THRESHOLD = max(500, int(0.75 * COOLING_SWEEP_ITERS))

# Relative probability of drawing each move operator per iteration. Set an
# entry to 0 to disable it. These are weights, not probabilities.
#
# "variant" is the operator that exercises the box size variation mechanism
# of Section 3.1; with --classic it is dropped automatically, since only
# variant 1 exists there.
MOVE_WEIGHTS = {
    "relocate":    1.0,
    "swap":        1.0,
    "2opt":        1.0,
    "orientation": 1.0,
    "variant":     1.5,
}


# -------------------------
# Instance preparation
# -------------------------
def prepare_instance(path, max_boxes=0, classic=False):
    """
    Load an instance and pre-compute, for every box, the list of size
    variants and the unique orientations of each variant.

    A "configuration" of box i is the pair (variant index, orientation
    index); the resulting edge lengths along X/Y/Z are looked up in
    box_shapes[i][variant][orientation]. Duplicate orientations (from equal
    edges) are removed so that the search does not waste moves on
    indistinguishable states.
    """

    L, W, H, boxes = load_instance(path)

    if max_boxes and max_boxes > 0:
        boxes = boxes[:max_boxes]

    box_shapes = []

    for (l, w, h) in boxes:

        per_variant = []

        for _k, l_k, w_k, h_k in box_variants(l, w, h, classic=classic):

            # (X, Y, Z) = (width, height, length), matching the MILP's axis
            # convention in 3DBPP-VBS.py.
            orientations = [
                (w_k, h_k, l_k),
                (h_k, w_k, l_k),
                (l_k, h_k, w_k),
                (h_k, l_k, w_k),
                (l_k, w_k, h_k),
                (w_k, l_k, h_k),
            ]

            per_variant.append(list(dict.fromkeys(orientations)))

        box_shapes.append(per_variant)

    return {
        "name": str(path),
        "L": L,
        "W": W,
        "H": H,
        "bin_volume": L * W * H,
        "boxes": boxes,
        "n": len(boxes),
        "shapes": box_shapes,
        "classic": classic,
        "total_box_volume": sum(l * w * h for (l, w, h) in boxes),
    }


# -------------------------
# Placement heuristic (extreme points)
# -------------------------
# Replaces the Gurobi feasibility sub-model of 3dp-cptp-SA.py. Boxes are
# placed one at a time, in the order given by the loading sequence, at the
# first extreme point where they fit. Extreme points are the corners
# generated by already-placed boxes (Crainic et al. 2008); the point list is
# kept sorted so that placement is deterministic — the same sequence always
# yields the same packing, which is what lets SA compare two sequences.
def pack_sequence(inst, sequence, orientations, variants):
    """
    Pack boxes in the given order and return (placements, packed_volume).

    A box that fits nowhere is skipped, exactly as in the model, where p_i
    may be 0. Returns placements as dicts holding the box index, its
    position, and its oriented edge lengths.
    """

    L, W, H = inst["L"], inst["W"], inst["H"]
    shapes = inst["shapes"]

    # Candidate positions, as (x, y, z). The empty bin offers just the
    # origin; every placed box adds the three corners in +X, +Y and +Z.
    points = [(0, 0, 0)]

    # Placed boxes are kept as flat parallel lists of their extents rather
    # than as dicts. This function is the inner loop of the whole search
    # (one full call per SA iteration), and the overlap test below runs
    # points x placed times per box, so attribute and key lookups there
    # dominate the runtime. Dicts are built once at the end instead.
    px0 = []
    py0 = []
    pz0 = []
    px1 = []
    py1 = []
    pz1 = []

    placed_boxes = []
    packed_volume = 0

    for i in sequence:

        dx, dy, dz = shapes[i][variants[i]][orientations[i]]

        best_idx = -1

        for idx, (qx, qy, qz) in enumerate(points):

            ax1 = qx + dx
            ay1 = qy + dy
            az1 = qz + dz

            if ax1 > L or ay1 > W or az1 > H:
                continue

            # Overlap test against everything already placed. Two boxes
            # overlap only if they overlap on all three axes.
            fits = True

            for k in range(len(placed_boxes)):
                if (qx < px1[k] and px0[k] < ax1
                        and qy < py1[k] and py0[k] < ay1
                        and qz < pz1[k] and pz0[k] < az1):
                    fits = False
                    break

            if fits:
                best_idx = idx
                break

        if best_idx < 0:
            continue

        qx, qy, qz = points.pop(best_idx)

        px0.append(qx)
        py0.append(qy)
        pz0.append(qz)
        px1.append(qx + dx)
        py1.append(qy + dy)
        pz1.append(qz + dz)

        placed_boxes.append((i, qx, qy, qz, dx, dy, dz,
                             variants[i], orientations[i]))

        packed_volume += dx * dy * dz

        for p in ((qx + dx, qy, qz), (qx, qy + dy, qz), (qx, qy, qz + dz)):
            if p[0] < L and p[1] < W and p[2] < H and p not in points:
                points.append(p)

        # Sorting by (z, y, x) fills the bin bottom-up and keeps placement
        # deterministic for a given sequence.
        points.sort(key=lambda p: (p[2], p[1], p[0]))

    placements = [
        {"box": b, "x": x, "y": y, "z": z, "dx": ex, "dy": ey, "dz": ez,
         "variant": v, "orientation": o}
        for (b, x, y, z, ex, ey, ez, v, o) in placed_boxes
    ]

    return placements, packed_volume


# -------------------------
# Solution representation & objective
# -------------------------
# A solution is the triple (sequence, orientations, variants); the packing
# itself is derived, never stored as the primary state.
def evaluate(inst, sol):
    """Pack the solution and return (utilization, placements)."""

    placements, volume = pack_sequence(
        inst, sol["sequence"], sol["orientations"], sol["variants"]
    )

    return volume / inst["bin_volume"], placements


def copy_solution(sol):
    return {
        "sequence": list(sol["sequence"]),
        "orientations": list(sol["orientations"]),
        "variants": list(sol["variants"]),
    }


def build_initial_solution(inst):
    """
    Start from the boxes sorted by decreasing volume, every box in its
    original size (variant 1) and first orientation. Largest-first is the
    classic bin-packing construction and gives SA a sane starting point.
    """

    order = sorted(range(inst["n"]),
                   key=lambda i: -(inst["boxes"][i][0]
                                   * inst["boxes"][i][1]
                                   * inst["boxes"][i][2]))

    return {
        "sequence": order,
        "orientations": [0] * inst["n"],
        "variants": [0] * inst["n"],
    }


# -------------------------
# Move operators
# -------------------------
# Each sampler returns a NEW solution (or None). Unlike the routing code
# there is no cheap local delta: a change anywhere in the sequence shifts
# every later placement, so the caller re-packs and measures the delta.
def sample_relocate(inst, sol):
    """Move one box to a different position in the loading sequence."""

    n = inst["n"]

    if n < 2:
        return None

    new = copy_solution(sol)

    i = random.randrange(n)
    box = new["sequence"].pop(i)

    j = random.randrange(n)
    new["sequence"].insert(j, box)

    return new


def sample_swap(inst, sol):
    """Exchange two boxes in the loading sequence."""

    n = inst["n"]

    if n < 2:
        return None

    new = copy_solution(sol)

    i, j = random.sample(range(n), 2)

    new["sequence"][i], new["sequence"][j] = \
        new["sequence"][j], new["sequence"][i]

    return new


def sample_2opt(inst, sol):
    """Reverse a segment of the loading sequence."""

    n = inst["n"]

    if n < 3:
        return None

    new = copy_solution(sol)

    i, j = sorted(random.sample(range(n), 2))

    new["sequence"][i:j + 1] = reversed(new["sequence"][i:j + 1])

    return new


def sample_orientation(inst, sol):
    """Give one box a different orthogonal orientation."""

    n = inst["n"]

    i = random.randrange(n)

    choices = len(inst["shapes"][i][sol["variants"][i]])

    if choices < 2:
        return None

    new = copy_solution(sol)

    options = [o for o in range(choices) if o != sol["orientations"][i]]
    new["orientations"][i] = random.choice(options)

    return new


def sample_variant(inst, sol):
    """
    Switch one box to a different size variant (Section 3.1).

    This is the operator specific to the 3DBPP-VBS. The orientation index is
    clamped, because a compressed variant can have fewer unique orientations
    than the original (e.g. when compression makes two edges equal).
    """

    n = inst["n"]

    i = random.randrange(n)

    n_variants = len(inst["shapes"][i])

    if n_variants < 2:
        return None

    new = copy_solution(sol)

    options = [k for k in range(n_variants) if k != sol["variants"][i]]
    k = random.choice(options)

    new["variants"][i] = k
    new["orientations"][i] = min(sol["orientations"][i],
                                 len(inst["shapes"][i][k]) - 1)

    return new


_SAMPLERS = {
    "relocate":    sample_relocate,
    "swap":        sample_swap,
    "2opt":        sample_2opt,
    "orientation": sample_orientation,
    "variant":     sample_variant,
}


def sample_random_move(inst, sol, move_weights=None):
    """
    Draw ONE random neighbour: pick a random operator, then a random move.

    As in 3dp-cptp-SA.py, operators are drawn WITHOUT replacement by weight,
    so that if the chosen one cannot produce a move the next is tried rather
    than wasting the iteration. Returns None only if no operator yields one.
    """

    if move_weights is None:
        move_weights = MOVE_WEIGHTS

    pool = [(name, w) for name, w in move_weights.items()
            if w > 0 and name in _SAMPLERS]

    # With fixed sizes there is only one variant, so that operator can never
    # fire; dropping it keeps the weights meaningful.
    if inst["classic"]:
        pool = [(name, w) for name, w in pool if name != "variant"]

    while pool:

        total = sum(w for _, w in pool)
        r = random.uniform(0.0, total)

        upto = 0.0
        pick = len(pool) - 1

        for idx, (_, w) in enumerate(pool):
            upto += w
            if r <= upto:
                pick = idx
                break

        name, _ = pool.pop(pick)

        move = _SAMPLERS[name](inst, sol)

        if move is not None:
            return move

    return None


def _diversify(inst, sol):
    """
    Shake the solution after prolonged stagnation: shuffle a random slice of
    the loading sequence and re-randomize some variants/orientations.
    """

    n = inst["n"]

    if n >= 4:
        i, j = sorted(random.sample(range(n), 2))

        if j - i >= 2:
            segment = sol["sequence"][i:j + 1]
            random.shuffle(segment)
            sol["sequence"][i:j + 1] = segment

    for _ in range(max(1, n // 5)):

        i = random.randrange(n)

        if not inst["classic"]:
            sol["variants"][i] = random.randrange(len(inst["shapes"][i]))

        sol["orientations"][i] = random.randrange(
            len(inst["shapes"][i][sol["variants"][i]])
        )

    return sol


# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(inst, max_iterations=MAX_ITERATIONS,
                        time_limit=TIME_LIMIT, t_init=T_INIT, alpha=ALPHA,
                        t_min=T_MIN, reheat_threshold=REHEAT_THRESHOLD,
                        t_reheat=T_REHEAT, seed=None, verbose=True):
    """
    Simulated Annealing for the 3DBPP-VBS — single-neighbour variant.

    Per iteration exactly ONE random neighbour is drawn (random operator +
    random move) and then accepted or rejected.

    Acceptance criterion:
      - Improving moves (delta > 0): always accepted
      - Worsening moves: accepted with probability exp(delta / T)
    Temperature schedule:
      - Geometric cooling: T *= alpha every iteration (also on rejection)
      - Reheating + diversification after prolonged stagnation

    Unlike the 3DP-CPTP version there is no packing-feasibility rejection:
    the placement heuristic simply leaves unfittable boxes out, so every
    candidate is a valid packing and no iteration is ever wasted on an
    infeasible draw.
    """

    if seed is not None:
        random.seed(seed)

    t_start = time.time()
    deadline = t_start + time_limit

    current = build_initial_solution(inst)
    current_obj, current_pl = evaluate(inst, current)

    best = copy_solution(current)
    best_obj = current_obj
    best_pl = current_pl

    if verbose:
        print(f"Initial solution: utilization={current_obj * 100:.2f}%, "
              f"boxes={len(current_pl)}/{inst['n']}")

    T = t_init
    no_improve_count = 0
    accepted_count = 0
    rejected_count = 0
    no_move_count = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):

        if time.time() >= deadline:
            if verbose:
                print(f"Time limit reached at iteration {iteration}.")
            break

        candidate = sample_random_move(inst, current)

        if candidate is None:
            # Nothing can be done to this solution at all — shake and retry.
            no_move_count += 1
            no_improve_count += 1

            if no_improve_count >= reheat_threshold:
                T = t_reheat
                _diversify(inst, current)
                current_obj, _ = evaluate(inst, current)
                no_improve_count = 0

            T = max(T * alpha, t_min)
            continue

        cand_obj, cand_pl = evaluate(inst, candidate)
        delta = cand_obj - current_obj

        if delta > 0:
            accept = True
        elif T > 1e-12:
            accept = random.random() < math.exp(delta / T)
        else:
            accept = False

        if accept:
            current = candidate
            current_obj = cand_obj
            accepted_count += 1

            if current_obj > best_obj + 1e-9:
                best = copy_solution(current)
                best_obj = current_obj
                best_pl = cand_pl
                no_improve_count = 0

                if verbose:
                    print(f"[iter {iteration}] *** NEW BEST: "
                          f"utilization={best_obj * 100:.2f}%, "
                          f"boxes={len(best_pl)}/{inst['n']}, T={T:.6f}, "
                          f"elapsed={time.time() - t_start:.1f}s")
            else:
                no_improve_count += 1
        else:
            rejected_count += 1
            no_improve_count += 1

        # ---- Cool down once per iteration, whatever happened ----
        T = max(T * alpha, t_min)

        # ---- Reheat + diversify on prolonged stagnation ----
        if no_improve_count >= reheat_threshold:
            T = t_reheat
            _diversify(inst, current)
            current_obj, _ = evaluate(inst, current)
            no_improve_count = 0

            if verbose:
                print(f"[iter {iteration}] Reheat + diversify. T={T:.6f}, "
                      f"utilization={current_obj * 100:.2f}%")

    elapsed = time.time() - t_start

    if verbose:
        print(f"\nSimulated Annealing finished in {elapsed:.2f}s after "
              f"{iteration} iterations.")
        print(f"Accepted: {accepted_count}, Rejected: {rejected_count}, "
              f"No-move draws: {no_move_count}")

    return {
        "utilization": best_obj,
        "placements": best_pl,
        "solution": best,
        "iterations": iteration,
        "runtime": elapsed,
        "accepted": accepted_count,
        "rejected": rejected_count,
    }


# -------------------------
# Verification and reporting
# -------------------------
def verify_solution(inst, placements):
    """
    Independent geometric check: every box inside the bin, no two boxes
    overlapping, and no box used twice. Mirrors the check in 3DBPP-VBS.py so
    that both solvers are validated the same way.
    """

    L, W, H = inst["L"], inst["W"], inst["H"]

    violations = []

    seen = set()

    for pl in placements:

        if pl["box"] in seen:
            violations.append(f"box {pl['box']} placed more than once")
        seen.add(pl["box"])

        if pl["x"] + pl["dx"] > L or pl["y"] + pl["dy"] > W \
                or pl["z"] + pl["dz"] > H:
            violations.append(f"box {pl['box']} exceeds the bin")

        if min(pl["x"], pl["y"], pl["z"]) < 0:
            violations.append(f"box {pl['box']} has a negative coordinate")

    for a in range(len(placements)):
        for b in range(a + 1, len(placements)):

            i, j = placements[a], placements[b]

            if (i["x"] < j["x"] + j["dx"] and j["x"] < i["x"] + i["dx"]
                    and i["y"] < j["y"] + j["dy"] and j["y"] < i["y"] + i["dy"]
                    and i["z"] < j["z"] + j["dz"]
                    and j["z"] < i["z"] + i["dz"]):
                violations.append(
                    f"boxes {i['box']} and {j['box']} overlap"
                )

    return violations


def print_report(inst, result, classic):
    """Print a summary of one solved instance."""

    problem = "3DBPP (fixed sizes)" if classic else "3DBPP-VBS"

    ratio = inst["total_box_volume"] / inst["bin_volume"] * 100
    note = "  <- caps the utilization" if ratio < 100 else ""

    print()
    print("=" * 66)
    print(f"  {problem} SA   instance: {inst['name']}")
    print("=" * 66)
    print(f"  bin (L x W x H)   : {inst['L']} x {inst['W']} x {inst['H']}")
    print(f"  boxes             : {inst['n']}")
    print(f"  total box volume  : {ratio:.2f} % of bin{note}")
    print(f"  iterations        : {result['iterations']}")
    print(f"  runtime           : {result['runtime']:.2f} s")
    print(f"  packed boxes      : "
          f"{len(result['placements'])} / {inst['n']}")
    print(f"  space utilization : {result['utilization'] * 100:.2f} %")

    if ratio > 0:
        print(f"  of available vol. : "
              f"{result['utilization'] / (ratio / 100) * 100:.2f} %")

    violations = verify_solution(inst, result["placements"])

    if violations:
        print(f"  CHECK             : {len(violations)} violation(s)")
        for v in violations[:5]:
            print(f"      - {v}")
    else:
        print("  check             : feasible (inside bin, no overlaps)")

    print()
    print("  box  variant  orient   (x, y, z)          edges X/Y/Z      "
          "original l/w/h")
    print("  " + "-" * 74)

    for pl in sorted(result["placements"], key=lambda d: d["box"]):

        l0, w0, h0 = inst["boxes"][pl["box"]]

        coord = f"({pl['x']}, {pl['y']}, {pl['z']})"
        edges = f"{pl['dx']}/{pl['dy']}/{pl['dz']}"

        print(f"  {pl['box']:>3}  {pl['variant'] + 1:>7}  "
              f"{pl['orientation'] + 1:>6}   {coord:<18} {edges:<16} "
              f"{l0}/{w0}/{h0}")

    print()


def plot_solution(inst, result, out_path):
    """Render the packing as a 3D plot (matplotlib)."""

    try:
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  plot skipped        : matplotlib is not installed")
        return

    L, W, H = inst["L"], inst["W"], inst["H"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab20")

    for idx, pl in enumerate(result["placements"]):

        x0, y0, z0 = pl["x"], pl["y"], pl["z"]
        dx, dy, dz = pl["dx"], pl["dy"], pl["dz"]

        c = [
            (x0, y0, z0), (x0 + dx, y0, z0),
            (x0 + dx, y0 + dy, z0), (x0, y0 + dy, z0),
            (x0, y0, z0 + dz), (x0 + dx, y0, z0 + dz),
            (x0 + dx, y0 + dy, z0 + dz), (x0, y0 + dy, z0 + dz),
        ]

        faces = [
            [c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]],
            [c[0], c[1], c[5], c[4]], [c[2], c[3], c[7], c[6]],
            [c[1], c[2], c[6], c[5]], [c[0], c[3], c[7], c[4]],
        ]

        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=cmap(idx % 20), edgecolors="black",
            linewidths=0.5, alpha=0.9,
        ))

    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Y (height)")
    ax.set_zlabel("Z (length)")
    ax.set_box_aspect((L, W, H))

    ax.set_title(f"{inst['name']}\n"
                 f"{len(result['placements'])}/{inst['n']} boxes, "
                 f"utilization {result['utilization'] * 100:.2f}%")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  plot saved to     : {out_path}")


# -------------------------
# main
# -------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Simulated Annealing for the 3DBPP-VBS, adapted from "
                    "3dp-cptp-SA.py."
    )

    parser.add_argument(
        "--instance",
        default="Martello/class_1/class1_n10_instance1.txt",
        help="path to an instance (default: Martello class 1, instance 1)",
    )
    parser.add_argument(
        "--max-boxes", type=int, default=0,
        help="use only the first k boxes (default: 0 = all)",
    )
    parser.add_argument(
        "--classic", action="store_true",
        help="solve the standard 3DBPP with fixed box sizes",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="solve both problems and report the Diff of formula (18)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=TIME_LIMIT,
        help=f"seconds per SA run (default: {TIME_LIMIT})",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="independent SA runs per problem; the best is reported "
             "(default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible runs",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-iteration SA log",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="save a 3D plot of the packing",
    )
    parser.add_argument(
        "--batch", nargs="+", default=None,
        help="solve several instances in sequence",
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

    modes = [True, False] if args.both else [args.classic]

    results = {}

    for classic in modes:

        inst = prepare_instance(path, max_boxes=args.max_boxes,
                                classic=classic)

        best = None

        for r in range(args.runs):

            seed = None if args.seed is None else args.seed + r

            result = simulated_annealing(
                inst,
                time_limit=args.time_limit,
                seed=seed,
                verbose=not args.quiet,
            )

            if best is None or result["utilization"] > best["utilization"]:
                best = result

        print_report(inst, best, classic)

        results["3DBPP" if classic else "3DBPP-VBS"] = best

        if args.plot and best["placements"]:
            suffix = "3dbpp" if classic else "3dbpp_vbs"
            plot_solution(inst, best, Path(f"{path.stem}_sa_{suffix}.png"))

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
    """Solve a list of instances in sequence and summarize the results."""

    import csv

    rows = []

    if args.both:
        header = (f"  {'instance':<24} {'n':>4} "
                  f"{'3DBPP%':>8} {'VBS%':>8} {'Diff%':>8} {'time(s)':>9}")
    else:
        header = (f"  {'instance':<24} {'n':>4} {'packed':>7} "
                  f"{'util%':>8} {'iters':>9} {'time(s)':>9}")

    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for spec in args.batch:

        path = Path(spec)

        if not path.exists():
            print(f"  {spec}: not found")
            continue

        modes = [True, False] if args.both else [args.classic]

        solved = {}

        for classic in modes:

            inst = prepare_instance(path, max_boxes=args.max_boxes,
                                    classic=classic)

            best = None

            for r in range(args.runs):

                seed = None if args.seed is None else args.seed + r

                result = simulated_annealing(
                    inst, time_limit=args.time_limit, seed=seed,
                    verbose=False,
                )

                if best is None or result["utilization"] > best["utilization"]:
                    best = result

            solved[classic] = (inst, best)

        full_name = f"{path.parent.name}/{path.stem}"

        name = full_name if len(full_name) <= 24 else "~" + full_name[-23:]

        if args.both:

            r1 = solved[True][1]["utilization"]
            r2 = solved[False][1]["utilization"]

            diff = (r2 - r1) / r1 * 100 if r1 > 0 else float("nan")
            total_time = solved[True][1]["runtime"] + solved[False][1]["runtime"]

            n = solved[True][0]["n"]

            print(f"  {name:<24} {n:>4} {r1 * 100:>8.2f} {r2 * 100:>8.2f} "
                  f"{diff:>8.2f} {total_time:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": n,
                "util_3dbpp": r1 * 100,
                "util_3dbpp_vbs": r2 * 100,
                "diff_pct": diff,
                "time_s": total_time,
            })

        else:

            inst, res = solved[args.classic]

            print(f"  {name:<24} {inst['n']:>4} "
                  f"{len(res['placements']):>7} "
                  f"{res['utilization'] * 100:>8.2f} "
                  f"{res['iterations']:>9} {res['runtime']:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": inst["n"],
                "packed": len(res["placements"]),
                "util_pct": res["utilization"] * 100,
                "iterations": res["iterations"],
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
