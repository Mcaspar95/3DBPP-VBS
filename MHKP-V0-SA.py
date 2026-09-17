#!/usr/bin/env python3
# Simulated Annealing for the MHKP under submodel V0 of Deplano et al. (2019)
# =========================================================================
# Heuristic counterpart to submodel V0 of
#
#     Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
#     "A mixed-integer linear model for the multiple heterogeneous knapsack
#      problem with realistic container loading constraints and bins' priority"
#     Intl. Trans. in Op. Res. 28(6), 3244-3275.
#
# V0 is the paper's Section 3.2 submodel holding constraints (1)-(10r): the
# geometry, the bin-priority objective, and STATIC STABILITY - but not the
# centre-of-mass distribution (V1) nor load bearing (V2). It is the natural
# first target for an SA port, because stability is the only constraint it adds
# over the geometry our existing decoder already enforces; weight, CoM and
# load-bearing data are not consulted at all.
#
# The SA engine is imported wholesale from 3DBPP-VBS-SA.py. What this file
# replaces is the DECODER and the VERIFIER; everything about how solutions are
# sampled, accepted, cooled and reheated is unchanged, so any difference in the
# results is attributable to the problem, not to the search.
#
# -------------------------------------------------------------------------
# WHAT V0 CHANGES RELATIVE TO THE 3DBPP-VBS
# -------------------------------------------------------------------------
#   * Multiple heterogeneous bins, not one. The solution therefore carries a
#     BIN ORDER, and bins are filled one at a time in that order. This is the
#     same structure 3DMHKP-SA.py uses for its containers.
#
#   * A priority objective, formula (1):
#
#         min  sum_j p_j * (bin_volume_j - packed_volume_j)
#
#     With p_j = 1/bin_volume_j (the paper's case study, formula (2)) this
#     prioritises filling SMALL bins, and the authors prove it equivalent to
#     maximising packing efficiency. SA maximises the negation, so that larger
#     is better and the engine's acceptance test needs no change.
#
#   * Two orientations, not six. The paper allows orthogonal rotation around
#     the vertical axis only (their Table 2), because the Physical Internet
#     H-container has a male/female locking mechanism on its top and bottom
#     faces. Length and width may swap; height is fixed.
#
#   * No size variants. Boxes are rigid here - the compression mechanism is
#     Xu et al.'s, not Deplano's - so the "variant" axis of the solution is
#     pinned to a single entry and its move operator never fires.
#
#   * STATIC STABILITY, constraints (10a)-(10r). This is what V0 adds and the
#     only genuinely new logic in this file. See is_supported().
#
#   * Grid positioning. Items sit on a beta-grid (their Definition 4, with
#     beta the gcd of the item sides); beta = 1 recovers free positioning.
#
# -------------------------------------------------------------------------
# THE STABILITY RULE (constraints (10a)-(10r))
# -------------------------------------------------------------------------
# An item is stable if it stands on the bin floor (10a) or on top of one or
# more other items (10b)-(10p). Resting on an item k requires:
#
#   * exact surface contact, z_i = z'_k, from (10b)-(10c) - the supporting
#     top face must be at precisely the supported bottom face's height; and
#
#   * at least TWO of the four bottom corners supported, from (10l)-(10o).
#     Read carefully, those four constraints are a pair of conditions: one of
#     the two x-side corner indicators must hold AND one of the two y-side
#     indicators must hold. A single corner, or two corners on one edge, is
#     not enough.
#
# Constraint (10q) then requires every packed item to have at least one
# support, which is what forbids floating. Note the paper allows the two
# corners to come from DIFFERENT supporting items - theta_{i,k} is summed over
# k in (10q) - so is_supported() accumulates corner coverage across every item
# at the right height rather than demanding a single one carry the box.
#
# -------------------------------------------------------------------------
# WHY THIS IS NOT A REPRODUCTION OF THEIR NUMBERS
# -------------------------------------------------------------------------
# The paper's instances are drawn from the Physical Internet catalogue of
# Landschuetzer et al. (2015) - 24 bin types and 440 item types - which is not
# published with the paper and is not in this repository. There is also no data
# availability statement. The generator below therefore follows the RULES the
# paper states (Section 5) but cannot reproduce their draws, so absolute
# objective values are NOT comparable with their Table 7. What IS comparable is
# the qualitative behaviour: the cost of the stability constraint, and how much
# of the packing SA can recover relative to an unconstrained decode.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#     python3 mhkp_instances.py                    # write the corpus first
#
#     python3 MHKP-V0-SA.py                        # the default group
#     python3 MHKP-V0-SA.py --group n10_b1          # their Table 7 size
#     python3 MHKP-V0-SA.py --group n40_b5          # multi-bin, priority active
#     python3 MHKP-V0-SA.py --no-stability          # ablate V0 -> geometry only
#     python3 MHKP-V0-SA.py 'MHKP/n18_b1/*.txt'     # explicit files
#     python3 MHKP-V0-SA.py --beta 1                # free positioning
#     python3 MHKP-V0-SA.py --csv out.csv

import argparse
import importlib.util
import math
import random
import statistics
import sys
import time
from bisect import insort
from math import gcd
from pathlib import Path


# The SA engine lives in 3DBPP-VBS-SA.py, which is not a valid module name
# (it starts with a digit and contains dashes), so it is loaded by path.
def _load_sa_module():

    path = Path(__file__).resolve().parent / "3DBPP-VBS-SA.py"

    spec = importlib.util.spec_from_file_location("dbpp_vbs_sa", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dbpp_vbs_sa"] = module
    spec.loader.exec_module(module)

    return module


_sa = _load_sa_module()

# The instance format, generator and reader live next to this file.
import mhkp_instances as _instances


# -------------------------
# Parameters
# -------------------------
# The objective here is the priority-weighted FILL RATIO of formula (1),
# renormalised into [0, 1] (see objective_scale below), so the temperature
# scale of the 3DBPP-VBS carries over unchanged and is instance-independent.
T_INIT = _sa.T_INIT
ALPHA = _sa.ALPHA
T_MIN = _sa.T_MIN
TIME_LIMIT = 10.0

# The paper's Table 7 setup: 10 items, one 120x120x120 bin, 1000 instances.
DEFAULT_BIN = (120, 120, 120)

# Relative move weights. "variant" is dropped: Deplano's boxes are rigid, so
# that operator can never fire. "orient" survives but only ever toggles between
# the two vertical-axis rotations of Table 2.
MOVE_WEIGHTS = {
    "relocate":  1.0,
    "swap":      1.0,
    "2opt":      1.0,
    "promote":   1.5,
    "demote":    1.5,
    "orient":    1.0,
    "variant":   0.0,   # rigid boxes - no size variants in this problem
}


# =========================================================================
# Instance loading
# =========================================================================
# Instances live in the MHKP/ directory as text files; mhkp_instances.py
# generates them and documents the format and its relation to the paper's own
# (unpublished) Physical Internet draws. They are read from disk rather than
# regenerated in memory so that a result can be re-checked against the exact
# input that produced it, and so that changing the generator cannot silently
# invalidate earlier runs.
def build_instance(bins, items, beta=None, name=""):
    """
    Build the internal instance structure from raw bins and items.

    `bins` are (L, W, H, priority) and `items` are (l, w, h), i.e. exactly what
    mhkp_instances.read_instance returns. Weight, CoM and load-bearing
    attributes are absent by design: submodel V0 reads none of them.
    """

    bin_list = []

    for (L, W, H, priority) in bins:
        bin_list.append({
            "dims": (L, W, H),          # length (Z), width (X), height (Y)
            "volume": L * W * H,
            # Formula (2) writes p_j = 1 / volume, but the priority is read
            # from the file rather than recomputed, so that an instance can
            # carry any weighting the model allows.
            "priority": priority,
        })

    item_list = [{"dims": tuple(d), "volume": d[0] * d[1] * d[2]}
                 for d in items]

    if beta is None:
        # Definition 4: beta is the gcd of all item sides, which keeps items
        # aligned on a common grid.
        beta = 0
        for it in item_list:
            for d in it["dims"]:
                beta = gcd(beta, d)
        beta = max(1, beta)

    inst = {
        "name": name,
        "items": item_list,
        "bins": bin_list,
        "n": len(item_list),
        "beta": beta,
        "total_item_volume": sum(it["volume"] for it in item_list),
        "total_bin_volume": sum(b["volume"] for b in bin_list),
    }

    _precompute(inst)

    return inst


def load_instance(path, beta=None):
    """Read an instance file and build the internal structure."""

    bins, items = _instances.read_instance(path)

    return build_instance(bins, items, beta=beta, name=str(path))


def _fits_any_rotation(dims, bin_dims):
    l, w, h = dims
    L, W, H = bin_dims
    # Table 2: only the two vertical-axis rotations; height stays height.
    return h <= H and ((l <= L and w <= W) or (w <= L and l <= W))


def rotations(dims):
    """
    The two orthogonal rotations of Table 2, as (dx, dy, dz).

    dx -> X-axis (width), dy -> Y-axis (height), dz -> Z-axis (length).
    Rotation is around the vertical axis only, so length and width swap while
    height is invariant. Duplicates (a square footprint) are removed.
    """

    l, w, h = dims

    out = [(w, h, l), (l, h, w)]

    return list(dict.fromkeys(out))


def _precompute(inst):
    """Cache, per item, the rotations that fit each bin at all."""

    feasible = []

    for it in inst["items"]:
        per_bin = []
        for b in inst["bins"]:
            L, W, H = b["dims"]
            per_bin.append(tuple(
                (dx, dy, dz) for (dx, dy, dz) in rotations(it["dims"])
                if dx <= W and dy <= H and dz <= L
            ))
        feasible.append(per_bin)

    inst["feasible"] = feasible
    inst["fits_at_all"] = [any(r for r in per_bin) for per_bin in feasible]

    # The best achievable value of formula (1) is a full pack of every bin;
    # the worst is an empty pack. Both are needed to renormalise (see below).
    inst["empty_cost"] = sum(b["priority"] * b["volume"] for b in inst["bins"])


# =========================================================================
# Objective: formula (1) with the priority of formula (2)
# =========================================================================
def objective_scale(inst, packed_by_bin):
    """
    Return the priority-weighted fill ratio in [0, 1].

    The paper MINIMISES wasted space weighted by priority, formula (1):

        sum_j p_j * (V_j - packed_j)

    Our SA engine maximises, and its temperature scale assumes an objective in
    [0, 1]. Both are achieved by reporting the complement, normalised by the
    cost of the empty solution:

        1 - [ sum_j p_j (V_j - packed_j) ] / [ sum_j p_j V_j ]
          =     sum_j p_j packed_j        /   sum_j p_j V_j

    This is a strictly decreasing affine function of formula (1), so it induces
    exactly the same ranking over solutions and the same optimum; it only
    rescales. With p_j = 1/V_j it is the mean fill ratio across bins, which is
    why filling a small bin pays as much as filling a large one - the paper's
    intended effect.
    """

    if inst["empty_cost"] <= 0:
        return 0.0

    gain = 0.0

    for k, b in enumerate(inst["bins"]):
        gain += b["priority"] * packed_by_bin.get(k, 0)

    return gain / inst["empty_cost"]


def wasted_space_objective(inst, packed_by_bin):
    """Formula (1) itself, for reporting in the paper's own units."""

    total = 0.0

    for k, b in enumerate(inst["bins"]):
        total += b["priority"] * (b["volume"] - packed_by_bin.get(k, 0))

    return total


# =========================================================================
# Stability, constraints (10a)-(10r)
# =========================================================================
# Two readings of constraints (10l)-(10o), which the paper does not disambiguate.
#
#   "corners"  at least TWO of the four base corners of the item lie within a
#              supporting top face. This is the paper's PROSE: "at least two
#              bottom corners of k should be on the top surface of i".
#
#   "peraxis"  the footprints overlap along x AND along y. This is what
#              constraints (10d)-(10k) literally ENCODE: the paper introduces
#              them as "the criteria for defining whether an item is on top of
#              another in the x/y plane [...] in (10d)-(10g) for the x-axis and
#              in (10h)-(10k) for the y-axis", i.e. one condition per axis, and
#              a single corner can satisfy both.
#
# The two differ: a box overhanging so that only one corner is covered passes
# "peraxis" but fails "corners". The MILP in mhkp_v0_milp.py implements the
# constraints as written and so realises "peraxis"; comparing it against an SA
# using "corners" would be comparing two different problems. STABILITY_RULE
# therefore selects, and any MILP-vs-SA experiment must set both to the same
# reading. "corners" remains the default because it is the stricter, safer
# packing and because earlier results in this repository were measured under it.
STABILITY_RULE = "corners"


def is_supported(x, y, z, dx, dz, placed, tol=0, rule=None):
    """
    True if a box at (x, y, z) with footprint dx by dz is stably supported.

    Constraint (10a): standing on the bin floor (y == 0) is always stable.

    Otherwise (10b)-(10p): the box must rest on one or more items whose TOP
    face is at exactly this box's bottom height, and at least two of its four
    bottom corners must be covered - specifically at least one of the two
    corners on the x-side and one of the two on the y-side, which is what
    (10l)-(10o) encode as two separate pairs.

    The corners may be covered by DIFFERENT supporting items, because (10q)
    sums theta over all k, so coverage is accumulated rather than required from
    a single item.
    """

    if y == 0:
        return True

    if rule is None:
        rule = STABILITY_RULE

    if rule == "peraxis":
        # Constraints (10d)-(10k) as written: overlap on each axis separately.
        for (qx, qy, qz, qdx, qdy, qdz) in placed:
            if qy + qdy != y:
                continue
            if (min(x + dx, qx + qdx) > max(x, qx)
                    and min(z + dz, qz + qdz) > max(z, qz)):
                return True
        return False

    # The four bottom corners, in the (x, z) plane the box rests on.
    corners = [
        (x,          z),
        (x + dx,     z),
        (x,          z + dz),
        (x + dx,     z + dz),
    ]

    covered = [False, False, False, False]

    for (qx, qy, qz, qdx, qdy, qdz) in placed:

        # (10b)-(10c): exact surface contact.
        if qy + qdy != y:
            continue

        for idx, (cx, cz) in enumerate(corners):
            if covered[idx]:
                continue
            # A corner counts as supported when it lies within the supporting
            # item's top face (closed interval: touching an edge still bears).
            if qx <= cx <= qx + qdx and qz <= cz <= qz + qdz:
                covered[idx] = True

    # (10l)-(10m) with (10n)-(10o): one corner from each axis pair.
    x_side = covered[0] or covered[1]
    y_side = covered[2] or covered[3]

    return x_side and y_side


# =========================================================================
# Decoder: sequence + bin order -> feasible, stable packing
# =========================================================================
def decode(inst, sol, stability=True, rule=None):
    """
    Deepest-bottom-left extreme-point packer with a V0 stability filter.

    Bins are filled one at a time in sol["bin_order"]; within a bin the still
    unplaced items are offered in sol["sequence"], each going to the first
    extreme point and rotation where it fits without overlapping AND is stably
    supported. An item that fits nowhere is left out, exactly as C_{i,j} = 0.

    Returns (placements, packed_by_bin).
    """

    beta = inst["beta"]
    feasible = inst["feasible"]
    rot_pref = sol["orientations"]

    placements = []
    packed_by_bin = {}
    assigned = set()

    for k in sol["bin_order"]:

        L, W, H = inst["bins"][k]["dims"]

        points = [(0, 0, 0)]        # extreme points as (z, y, x), kept sorted
        placed = []                 # (x, y, z, dx, dy, dz)
        packed_volume = 0

        for i in sol["sequence"]:

            if i in assigned:
                continue

            orients = feasible[i][k]

            nr = len(orients)
            if nr == 0:
                continue

            start = rot_pref[i] % nr
            hit = None

            for (qz, qy, qx) in points:

                for r in range(nr):

                    dx, dy, dz = orients[(start + r) % nr]

                    # Definition 4: x and z sit on the beta grid. y is free,
                    # since stacking height is determined by what is below.
                    if qx % beta or qz % beta:
                        continue

                    if qx + dx > W or qy + dy > H or qz + dz > L:
                        continue

                    ok = True

                    for (px, py, pz, pdx, pdy, pdz) in placed:
                        if (qx < px + pdx and px < qx + dx
                                and qy < py + pdy and py < qy + dy
                                and qz < pz + pdz and pz < qz + dz):
                            ok = False
                            break

                    if not ok:
                        continue

                    if stability and not is_supported(qx, qy, qz, dx, dz,
                                                      placed, rule=rule):
                        continue

                    hit = (qx, qy, qz, dx, dy, dz, (start + r) % nr)
                    break

                if hit is not None:
                    break

            if hit is None:
                continue

            qx, qy, qz, dx, dy, dz, r = hit

            placed.append((qx, qy, qz, dx, dy, dz))
            placements.append({
                "item": i, "bin": k, "x": qx, "y": qy, "z": qz,
                "dx": dx, "dy": dy, "dz": dz, "rotation": r,
            })

            assigned.add(i)
            packed_volume += dx * dy * dz

            points.remove((qz, qy, qx))

            for nx, ny, nz in ((qx + dx, qy, qz), (qx, qy + dy, qz),
                               (qx, qy, qz + dz)):

                if nx >= W or ny >= H or nz >= L:
                    continue

                p = (nz, ny, nx)

                if p in points:
                    continue

                inside = False

                for (px, py, pz, pdx, pdy, pdz) in placed:
                    if (px <= nx < px + pdx and py <= ny < py + pdy
                            and pz <= nz < pz + pdz):
                        inside = True
                        break

                if not inside:
                    insort(points, p)

        packed_by_bin[k] = packed_volume

    return placements, packed_by_bin


def evaluate(inst, sol, stability=True, rule=None):
    """Decode, store the packing on the solution, and return the objective."""

    placements, packed_by_bin = decode(inst, sol, stability=stability, rule=rule)

    sol["placements"] = placements
    sol["packed"] = set(p["item"] for p in placements)
    sol["packed_by_bin"] = packed_by_bin
    sol["utilization"] = objective_scale(inst, packed_by_bin)

    return sol["utilization"]


# =========================================================================
# Verification: re-check a packing against constraints (1)-(10r)
# =========================================================================
def verify(inst, placements, stability=True, rule=None):
    """Independently re-check a packing. Returns a list of violation strings."""

    errors = []
    seen = set()

    by_bin = {}

    for p in placements:

        if p["item"] in seen:
            errors.append(f"item {p['item']} placed more than once")
        seen.add(p["item"])

        by_bin.setdefault(p["bin"], []).append(p)

    for k, group in by_bin.items():

        L, W, H = inst["bins"][k]["dims"]

        for p in group:

            if p["x"] + p["dx"] > W or p["y"] + p["dy"] > H \
                    or p["z"] + p["dz"] > L:
                errors.append(f"item {p['item']} exceeds bin {k}")

            if min(p["x"], p["y"], p["z"]) < 0:
                errors.append(f"item {p['item']} has a negative coordinate")

            if p["x"] % inst["beta"] or p["z"] % inst["beta"]:
                errors.append(f"item {p['item']} is off the beta grid")

            if (p["dx"], p["dy"], p["dz"]) not in rotations(
                    inst["items"][p["item"]]["dims"]):
                errors.append(f"item {p['item']} uses a disallowed rotation")

        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                if (i["x"] < j["x"] + j["dx"] and j["x"] < i["x"] + i["dx"]
                        and i["y"] < j["y"] + j["dy"] and j["y"] < i["y"] + i["dy"]
                        and i["z"] < j["z"] + j["dz"] and j["z"] < i["z"] + i["dz"]):
                    errors.append(
                        f"items {i['item']} and {j['item']} overlap in bin {k}")

        if stability:
            # Re-check (10a)-(10r) against every OTHER item in the bin.
            for p in group:
                others = [(q["x"], q["y"], q["z"], q["dx"], q["dy"], q["dz"])
                          for q in group if q is not p]
                if not is_supported(p["x"], p["y"], p["z"],
                                    p["dx"], p["dz"], others, rule=rule):
                    errors.append(f"item {p['item']} floats in bin {k}")

    return errors


# =========================================================================
# Solution representation and SA glue
# =========================================================================
def build_initial_solution(inst, stability=True):
    """
    Largest-volume-first sequence, bins in priority order (smallest first).

    Filling small bins first is the paper's own WFBF ordering, so this starts
    SA from the same construction philosophy their heuristic uses.
    """

    order = sorted(range(inst["n"]), key=lambda i: -inst["items"][i]["volume"])
    bin_order = sorted(range(len(inst["bins"])),
                       key=lambda k: inst["bins"][k]["volume"])

    sol = {
        "sequence": order,
        "orientations": [0] * inst["n"],
        "variants": [0] * inst["n"],
        "bin_order": bin_order,
        "placements": [],
        "packed": set(),
        "packed_by_bin": {},
        "utilization": 0.0,
    }

    evaluate(inst, sol, stability=stability)

    return sol


def copy_solution(sol):
    return {
        "sequence": list(sol["sequence"]),
        "orientations": list(sol["orientations"]),
        "variants": list(sol["variants"]),
        "bin_order": list(sol["bin_order"]),
        "placements": list(sol["placements"]),
        "packed": set(sol["packed"]),
        "packed_by_bin": dict(sol["packed_by_bin"]),
        "utilization": sol["utilization"],
    }


def _adapt(inst, stability):
    """
    Present this problem to the SA engine through the interface it expects.

    The engine reads inst["n"], inst["shapes"], inst["feasible"],
    inst["fits_at_all"] and inst["classic"] in its samplers. Here "feasible" is
    per (item, bin) rather than per (item, variant), so a view keyed on the
    FIRST bin in the order is supplied for the orientation sampler to size its
    choices - the rotation count is the same in every bin it fits.
    """

    view = dict(inst)

    view["classic"] = True          # rigid boxes: disables the variant sampler
    view["shapes"] = [[r] for r in
                      (rotations(it["dims"]) for it in inst["items"])]

    # feasible[i][0] is what the samplers index; give them the rotations that
    # fit the roomiest bin, so an orientation move is never wrongly rejected.
    widest = max(range(len(inst["bins"])),
                 key=lambda k: inst["bins"][k]["volume"])
    view["feasible"] = [[inst["feasible"][i][widest]] for i in range(inst["n"])]
    view["boxes"] = [it["dims"] for it in inst["items"]]

    return view


def sample_bin_move(inst, sol):
    """
    Exchange two bins in the filling order.

    Only bins of different shape are worth exchanging: filling two identical
    bins in the other order decodes to the same packing. This is the operator
    that explores what the paper's randomisedOrderWithPriority explores, and it
    is the direct answer to the under-filled-last-bin weakness they document
    for WFBF - SA can reorder the bins, WFBF cannot.
    """

    order = sol["bin_order"]

    if len(order) < 2:
        return None

    for _ in range(12):

        a = random.randrange(len(order))
        b = random.randrange(len(order))

        if a == b:
            continue

        if inst["bins"][order[a]]["dims"] == inst["bins"][order[b]]["dims"]:
            continue

        return {"type": "bin", "a": a, "b": b}

    return None


def _diversify(inst, sol, stability=True):
    """
    Shake the solution after prolonged stagnation.

    The 3DBPP-VBS version cannot be reused directly: its tail re-evaluates
    through that module's decoder, which expects a single-bin instance keyed on
    "L"/"W"/"H". The perturbation itself is the same idea - promote rejected
    cargo, scramble a block of the sequence, re-randomise some rotations - plus
    a reshuffle of the bin order, which is this problem's extra degree of
    freedom.
    """

    sequence = sol["sequence"]
    n = inst["n"]

    unpacked = [i for i in sequence
                if i not in sol["packed"] and inst["fits_at_all"][i]]

    if unpacked:

        n_promote = max(1, len(unpacked) // 3)
        chosen = set(random.sample(unpacked, min(n_promote, len(unpacked))))

        rest = [i for i in sequence if i not in chosen]
        head = max(1, n // 3)

        for i in chosen:
            rest.insert(random.randrange(head + 1), i)

        sequence = rest
        sol["sequence"] = sequence

    if n >= 4:
        seg = max(2, n // 5)
        start = random.randrange(n - seg + 1)
        block = sequence[start:start + seg]
        random.shuffle(block)
        sequence[start:start + seg] = block

    for _ in range(max(1, n // 5)):
        i = random.randrange(n)
        nr = len(rotations(inst["items"][i]["dims"]))
        if nr > 1:
            sol["orientations"][i] = random.randrange(nr)

    random.shuffle(sol["bin_order"])

    evaluate(inst, sol, stability=stability)

    return sol


def simulated_annealing(inst, time_limit=TIME_LIMIT, t_init=T_INIT,
                        alpha=ALPHA, t_min=T_MIN, seed=None, stability=True,
                        verbose=False):
    """
    SA over (sequence, rotation, bin order) for the MHKP under V0.

    The loop mirrors 3DBPP-VBS-SA.simulated_annealing exactly - one random
    neighbour per iteration, Metropolis acceptance, geometric cooling, reheat
    with diversification on stagnation - so results are attributable to the
    problem rather than to a changed search. It is reimplemented rather than
    imported only because the move set gains a bin-order operator and the
    decoder is this file's.
    """

    if seed is not None:
        random.seed(seed)

    view = _adapt(inst, stability)

    t_start = time.time()
    deadline = t_start + time_limit

    reheat_threshold = _sa.default_reheat_threshold(alpha)
    t_reheat = _sa.REHEAT_FRACTION * t_init

    current = build_initial_solution(inst, stability=stability)
    initial = current["utilization"]

    best = copy_solution(current)

    T = t_init
    no_improve = 0
    accepted = rejected = iteration = 0

    while True:

        iteration += 1

        if time.time() >= deadline:
            break

        # Draw one neighbour: the shared operators, plus the bin-order swap.
        if len(inst["bins"]) > 1 and random.random() < 0.15:
            move = sample_bin_move(inst, current)
        else:
            move = _sa.sample_random_move(view, current, MOVE_WEIGHTS)

        if move is None:
            T = max(T * alpha, t_min)
            continue

        trial = copy_solution(current)

        if move["type"] == "bin":
            bo = trial["bin_order"]
            a, b = move["a"], move["b"]
            bo[a], bo[b] = bo[b], bo[a]
        else:
            _sa.apply_move(view, trial, move)

        delta = evaluate(inst, trial, stability=stability) - current["utilization"]

        if delta > 0:
            accept = True
        elif T > 1e-12:
            accept = random.random() < math.exp(delta / T)
        else:
            accept = False

        if accept:
            current = trial
            accepted += 1
            if current["utilization"] > best["utilization"] + 1e-12:
                best = copy_solution(current)
                no_improve = 0
            else:
                no_improve += 1
        else:
            rejected += 1
            no_improve += 1

        T = max(T * alpha, t_min)

        if no_improve >= reheat_threshold:
            T = t_reheat
            _diversify(inst, current, stability=stability)
            no_improve = 0

    elapsed = time.time() - t_start

    return {
        "utilization": best["utilization"],
        "placements": best["placements"],
        "packed_by_bin": best["packed_by_bin"],
        "solution": best,
        "n_packed": len(best["packed"]),
        "initial": initial,
        "iterations": iteration,
        "accepted": accepted,
        "rejected": rejected,
        "runtime": elapsed,
        "wasted": wasted_space_objective(inst, best["packed_by_bin"]),
    }


# =========================================================================
# Metrics of Section 5 (equations (19)-(21))
# =========================================================================
def eccentricity(dims):
    """Equation (19): how far a box is from being a cube."""

    l, w, h = dims

    return max(
        abs((l - w) / (l + w)),
        abs((h - w) / (h + w)),
        abs((l - h) / (l + h)),
    )


def relative_volume(item_volume, bin_volume):
    """Equation (21)."""
    return item_volume / bin_volume if bin_volume else 0.0


# =========================================================================
# Driver
# =========================================================================
def collect_instances(args):
    """
    Resolve the instance files to run.

    --instances takes paths or glob patterns; with none given, the whole group
    directory is used. Missing files are an error rather than a silent skip, so
    that a typo cannot quietly shrink an experiment.
    """

    root = Path(args.instance_dir)

    if args.instances:
        paths = []
        for spec in args.instances:
            matched = sorted(Path().glob(spec)) if any(
                c in spec for c in "*?[") else [Path(spec)]
            if not matched:
                raise SystemExit(f"no instance matches: {spec}")
            for m in matched:
                if not m.exists():
                    raise SystemExit(f"instance not found: {m}")
            paths.extend(matched)
        return paths

    group_dir = root / args.group

    if not group_dir.is_dir():
        raise SystemExit(
            f"no such instance group: {group_dir}\n"
            f"generate the corpus first:  python3 mhkp_instances.py")

    paths = sorted(group_dir.glob("*.txt"))

    if not paths:
        raise SystemExit(f"no .txt instances in {group_dir}")

    return paths


def run_experiment(args):

    paths = collect_instances(args)

    if args.limit:
        paths = paths[:args.limit]

    rows = []

    print("=" * 78)
    print("MHKP under submodel V0 of Deplano et al. (2019) - SA")
    print(f"{len(paths)} instance(s) | {args.time_limit}s each "
          f"| stability={not args.no_stability}")
    print("=" * 78)
    print(f"{'instance':<26}{'packed':>9}{'greedy':>9}{'SA':>9}"
          f"{'wasted(1)':>12}{'iters':>8}{'viol':>6}")

    t0 = time.time()

    for t, path in enumerate(paths):

        inst = load_instance(path, beta=args.beta)

        res = simulated_annealing(
            inst,
            time_limit=args.time_limit,
            seed=args.seed + t,
            stability=not args.no_stability,
        )

        errs = verify(inst, res["placements"], stability=not args.no_stability)

        rows.append({
            "instance": path.name,
            "packed": res["n_packed"],
            "items": inst["n"],
            "greedy": res["initial"],
            "sa": res["utilization"],
            "wasted": res["wasted"],
            "iterations": res["iterations"],
            "violations": len(errs),
            "mean_ecc": statistics.fmean(
                eccentricity(it["dims"]) for it in inst["items"]),
            "mean_volrel": statistics.fmean(
                relative_volume(it["volume"], inst["bins"][0]["volume"])
                for it in inst["items"]),
        })

        if args.verbose or t < 12 or errs:
            label = path.stem
            label = label if len(label) <= 25 else "~" + label[-24:]
            print(f"{label:<26}{res['n_packed']:>5}/{inst['n']:<3}"
                  f"{res['initial'] * 100:>8.2f}%{res['utilization'] * 100:>8.2f}%"
                  f"{res['wasted']:>12.4f}{res['iterations']:>8}{len(errs):>6}")
            if errs:
                print(f"    !! {errs[0]}")

    elapsed = time.time() - t0

    print("-" * 78)

    n = len(rows)
    mean_packed = statistics.fmean(r["packed"] for r in rows)
    mean_greedy = statistics.fmean(r["greedy"] for r in rows)
    mean_sa = statistics.fmean(r["sa"] for r in rows)
    mean_wasted = statistics.fmean(r["wasted"] for r in rows)
    total_viol = sum(r["violations"] for r in rows)
    improved = sum(1 for r in rows if r["sa"] > r["greedy"] + 1e-12)

    print(f"instances                  : {n}")
    mean_items = statistics.fmean(r["items"] for r in rows)
    print(f"mean packed items          : {mean_packed:.3f} / {mean_items:.0f}")
    print(f"mean greedy objective      : {mean_greedy * 100:.2f} %")
    print(f"mean SA objective          : {mean_sa * 100:.2f} % "
          f"({(mean_sa - mean_greedy) * 100:+.2f} pts)")
    print(f"mean wasted space, eq. (1) : {mean_wasted:.4f}")
    print(f"SA improved over greedy    : {improved}/{n}")
    print(f"mean eccentricity, eq. (19): "
          f"{statistics.fmean(r['mean_ecc'] for r in rows):.4f}")
    print(f"mean VolRel, eq. (21)      : "
          f"{statistics.fmean(r['mean_volrel'] for r in rows):.5f}")
    print(f"verification               : "
          f"{'all feasible' if total_viol == 0 else f'{total_viol} VIOLATIONS'}")
    print(f"total wall clock           : {elapsed:.1f}s")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"results written to {args.csv}")

    return rows


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Simulated annealing for the MHKP under submodel V0 "
                    "(stability only) of Deplano et al. (2019).")

    parser.add_argument("instances", nargs="*", default=None, metavar="INSTANCE",
                        help="instance files or glob patterns; with none given, "
                             "the whole --group directory is run")
    parser.add_argument("--instance-dir", default="MHKP",
                        help="root directory of the instance corpus "
                             "(default: MHKP, written by mhkp_instances.py)")
    parser.add_argument("--group", default="n18_b1",
                        help="instance group to run when no files are named "
                             "(default: n18_b1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N instances of the selection")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help=f"SA seconds per instance (default {TIME_LIMIT})")
    parser.add_argument("--beta", type=int, default=None,
                        help="grid cell size; default = gcd of item sides, "
                             "1 = free positioning")
    parser.add_argument("--no-stability", action="store_true",
                        help="drop constraints (10a)-(10r): ablates V0 down to "
                             "geometry only, to price the stability constraint")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    run_experiment(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
