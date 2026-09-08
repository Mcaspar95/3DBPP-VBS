#!/usr/bin/env python3
# Simulated Annealing for the 3DBPP-VBS — sequence-decoder variant
# =========================================================================
# Heuristic counterpart to the exact MILP in 3DBPP-VBS.py for the problem of
#
#     Xu, X., Wu, B., Ma, Z., Yu, Y. (2026)
#     "The three-dimensional bin packing problem with variable box size"
#     Transportation Research Part E 214, 105038.
#
# One bin of size L x W x H. Each box may be compressed vertically into one of
# three size variants (Section 3.1) and rotated into one of six orthogonal
# orientations, giving |C_i| = 18 configurations. A subset of the boxes is
# loaded so that the used volume fraction of the bin, formula (2), is maximal.
# This file plays the role the paper's BSA plays; the MILP is not called.
#
# The SA engine is the one shared with 3DMHKP-SA.py and 3dp-cptp-SA.py:
#
#     1. pick ONE move operator at random (weighted by MOVE_WEIGHTS)
#     2. sample ONE random neighbour from that operator
#     3. accept or reject that single neighbour by the Metropolis criterion
#     4. cool down once per iteration, regardless of the outcome
#     5. reheat + diversify after prolonged stagnation
#
#   - Improving moves are always accepted
#   - Worsening moves are accepted with probability exp(delta / T)
#   - Temperature decreases geometrically: T *= ALPHA each iteration
#
# -------------------------------------------------------------------------
# SOLUTION REPRESENTATION
# -------------------------------------------------------------------------
# SA does not search over packings, it searches over the input of a
# deterministic constructive DECODER:
#
#     solution = (loading sequence over the boxes,
#                 size variant per box,
#                 orientation preference per box)
#
#     decode(solution) -> a genuine, verified feasible packing + its utilization
#
# Every solution maps to a feasible packing by construction — a box that fits
# nowhere is simply left out, exactly as p_i = 0 in the model — so there are no
# infeasible neighbours to reject and no Gurobi call anywhere in this file.
# The price is that the objective is not an analytic delta: a change anywhere
# in the sequence shifts every later placement, so each drawn neighbour is
# fully re-decoded before the Metropolis test can be applied.
#
# The decoder is a deepest-bottom-left extreme-point packer (Crainic et al.
# 2008, cited in the paper's Section 2.1): boxes are offered in the loading
# sequence, each going to the first (lowest, then front, then left) extreme
# point where its chosen configuration fits. The initial sequence is the
# classic largest-volume-first construction, so SA starts from that greedy
# packing and can only improve on it.
#
# -------------------------------------------------------------------------
# WHAT CHANGES RELATIVE TO 3DMHKP-SA.py
# -------------------------------------------------------------------------
# The 3DMHKP fills SEVERAL heterogeneous containers with boxes of a few types
# and maximizes stowed VALUE. Here there is ONE bin and the objective is the
# volume fraction of formula (2). That drives four changes:
#
#   * No container order. The 3DMHKP solution carried a container filling
#     sequence and a "container" swap operator; with a single bin both are
#     dropped, and the decoder's outer loop over containers disappears.
#
#   * A variant dimension. The 3DMHKP had one immutable size per box; the VBS
#     mechanism of Section 3.1 gives each box three, so the solution carries a
#     variant per box and a CHANGE_VARIANT operator — the operator specific to
#     this problem. Compression trades height for footprint at constant
#     volume, so a variant change never alters the objective contribution of a
#     box, only whether it and its successors still fit.
#
#   * A different objective scale. The 3DMHKP objective is a stowed value
#     spanning orders of magnitude across instances, so T_INIT there is a
#     fraction of f(S_0). Here the objective is a RATIO in [0, 1], bounded and
#     comparable across every instance, so a fixed T_INIT is meaningful again
#     and T_INIT_FRACTION is gone. See the note at T_INIT.
#
#   * Degeneracy filters instead of type filters. The 3DMHKP instances hold
#     200 boxes of 2-6 types, so its samplers reject draws that permute
#     identical boxes. The BR and Martello instances here are heterogeneous,
#     but the same principle applies to a weaker equivalence: boxes with equal
#     dimensions in equal configurations are indistinguishable to the decoder,
#     and _key/_is_noop_span filter those draws.
#
# -------------------------------------------------------------------------
# THE MOVES (6 operators)
# -------------------------------------------------------------------------
#   relocate  — move one box to another position in the loading sequence
#   swap      — exchange the sequence positions of two boxes
#   2opt      — reverse a segment of the loading sequence
#   promote   — pull a currently UNPACKED box forward, so it can displace
#               boxes that are currently in the bin
#   demote    — push a currently PACKED box back, freeing the space it holds
#   orient    — change the orientation preference of one box
#   variant   — switch one box between its three compression variants
#
# promote/demote are what actually exchange loaded against rejected cargo; on
# instances whose boxes do not all fit they carry most of the improvement, and
# they are weighted accordingly. With --classic only variant 1 exists and the
# "variant" operator is dropped automatically.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#     python3 3DBPP-VBS-SA.py                      # Martello class 1, inst. 1
#     python3 3DBPP-VBS-SA.py --classic            # fixed sizes (plain 3DBPP)
#     python3 3DBPP-VBS-SA.py --both               # 3DBPP vs 3DBPP-VBS, Diff
#     python3 3DBPP-VBS-SA.py --instance BR/BR1/1.txt --time-limit 60
#     python3 3DBPP-VBS-SA.py --batch Martello/class_1/*.txt --both --csv r.csv
#     python3 3DBPP-VBS-SA.py --greedy-only        # decode the initial order only
#
# The environment is the shared project .venv; see 3DBPP-VBS.py.

import argparse
import math
import os
import random
import sys
import time
from bisect import insort
from pathlib import Path


# Run under the project's .venv even when started with a different
# interpreter, so that matplotlib is always available for --plot.
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


# Instance loading and the size-variant model are shared with the MILP so that
# both solvers are guaranteed to see exactly the same instances and the same
# box variants. 3DBPP-VBS.py is not a valid module name (it starts with a digit
# and contains dashes), so it is loaded by path.
#
# Importing it pulls in gurobipy, which the MILP needs at module level but this
# file never uses. When gurobipy is missing or unlicensed the import is skipped
# and the two functions actually needed are re-implemented below, so the
# heuristic stays runnable in an environment without Gurobi.
def _load_model_module():

    import importlib.util

    path = Path(__file__).resolve().parent / "3DBPP-VBS.py"

    if not path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("dbpp_vbs_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ImportError:
        return None


_model = _load_model_module()


# Compression ratios s^k for the three variants (Section 3.1).
COMPRESSION_RATIOS = ([1.0, 0.85, 0.8] if _model is None
                      else _model.COMPRESSION_RATIOS)


if _model is not None:

    load_instance = _model.load_instance
    box_variants = _model.box_variants

else:

    # Fallback definitions, identical to those in 3DBPP-VBS.py. They exist only
    # so that this file runs without gurobipy installed; when the MILP module
    # imports, its versions are used and these are dead code.

    def load_instance(path):
        """Load a BR *.json instance or a simple "n W H L" text instance."""

        import json

        path = Path(path)

        if path.suffix.lower() == ".json":

            with open(path) as f:
                data = json.load(f)

            obj = data["Objects"][0]
            L, W, H = int(obj["Length"]), int(obj["Depth"]), int(obj["Height"])

            boxes = []

            for item in data["Items"]:
                dims = (int(item["Length"]), int(item["Depth"]),
                        int(item["Height"]))
                boxes.extend([dims] * int(item["Demand"]))

            return L, W, H, boxes

        with open(path) as f:
            tokens = f.read().split()

        n = int(tokens[0])
        W, H, L = int(tokens[1]), int(tokens[2]), int(tokens[3])

        boxes = []

        for i in range(n):
            w, h, l = (int(v) for v in tokens[4 + 3 * i: 7 + 3 * i])
            boxes.append((l, w, h))

        return L, W, H, boxes

    def box_variants(l, w, h, classic=False):
        """The three size variants of a box, formula (1)."""

        ratios = [1.0] if classic else COMPRESSION_RATIOS

        variants = []

        for k, s in enumerate(ratios, start=1):
            factor = math.sqrt(1.0 / s)
            variants.append((k, int(factor * l), int(factor * w), int(s * h)))

        return variants


# -------------------------
# Parameters
# -------------------------
T_INIT = 0.02              # initial temperature
ALPHA = 0.9995             # geometric cooling factor (T *= ALPHA each iteration)
T_MIN = 1e-6               # minimum temperature
MAX_ITERATIONS = 50000000  # maximum SA iterations
TIME_LIMIT = 60.0          # total wall-clock seconds per run
EP_LIMIT = 0               # max extreme points scanned per box, 0 = unlimited

# Why the temperature scale is far below the 3DP-CPTP values (T_INIT=100
# there) and is a fixed number rather than the fraction of f(S_0) that
# 3DMHKP-SA.py uses: the objective here is a UTILIZATION RATIO in [0, 1], not a
# profit in the hundreds nor a stowed value spanning orders of magnitude. A
# typical worsening move costs delta ~ -0.01, so exp(delta / T) with T=100 is
# exp(-1e-4) ~ 1 — every move would be accepted and SA would degrade into a
# random walk. T_INIT=0.02 puts the initial acceptance of such a move near
# exp(-0.5) ~ 60%, the regime the schedule is supposed to start in. Because the
# objective is bounded by 1 on every instance, that single number transfers
# across the whole benchmark set, which is exactly what does NOT hold in
# 3DMHKP-SA.py and is why it derives T0 from the instance instead.
#
# Temperature must always be scaled to the objective; this is the single most
# important thing to get right when porting an SA between problems.

REHEAT_FRACTION = 0.25     # T is reset to REHEAT_FRACTION * T_INIT on a reheat.
                           # A reheat has to RESET the temperature, not scale
                           # the annealed one, which by then is near T_MIN.

# Iterations of one full geometric sweep T_INIT -> T_MIN: ~20k at ALPHA=0.9995.
# REHEAT_THRESHOLD is derived from that sweep length rather than fixed: an
# absolute threshold far below the sweep resets T before it can ever cool,
# which pins acceptance near 100% and degrades SA into a random walk.
REHEAT_THRESHOLD = None    # None = max(500, 0.75 * sweep length at ALPHA)

# Relative probability of drawing each move operator per iteration. Set an entry
# to 0 to disable that operator. These are weights, not probabilities — they are
# normalised.
MOVE_WEIGHTS = {
    "relocate":  1.0,
    "swap":      1.0,
    "2opt":      1.0,
    "promote":   1.5,   # slightly favoured: these two are the operators that
    "demote":    1.5,   # actually exchange loaded against rejected cargo
    "orient":    1.0,
    "variant":   1.5,   # the operator specific to the VBS mechanism
}

_SAMPLE_TRIES = 12         # retries per sampler before it gives up this iteration


def cooling_sweep_iters(alpha, t_min=T_MIN, t_init=T_INIT):
    """Iterations of one full geometric sweep t_init -> t_min at this alpha."""
    return math.log(t_min / t_init) / math.log(alpha)


def default_reheat_threshold(alpha):
    return max(500, int(0.75 * cooling_sweep_iters(alpha)))


# =========================================================================
# Instance preparation
# =========================================================================
def prepare_instance(path, max_boxes=0, classic=False):
    """
    Load an instance and pre-compute, for every box, the list of size variants
    and the unique orientations of each variant.

    A "configuration" of box i is the pair (variant, orientation); the edge
    lengths along the axes are shapes[i][variant][orientation] as (dx, dy, dz)
    with

        dx -> X-axis, bounded by W (width)
        dy -> Y-axis, bounded by H (height)
        dz -> Z-axis, bounded by L (length)

    which is the axis convention of the MILP in 3DBPP-VBS.py, so that a
    placement produced here can be checked against the model's own bounds
    (4)-(6) without permuting anything.

    Duplicate orientations (which arise when a variant has equal edges, and
    compression makes that common) are removed so that the search does not
    waste moves on indistinguishable states.
    """

    L, W, H, boxes = load_instance(path)

    if max_boxes and max_boxes > 0:
        boxes = boxes[:max_boxes]

    box_shapes = []

    for (l, w, h) in boxes:

        per_variant = []

        for _k, l_k, w_k, h_k in box_variants(l, w, h, classic=classic):

            # The 6 orthogonal orientations, as (X, Y, Z) = (width, height,
            # length): every assignment of the three edges to the three axes.
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

    inst = {
        "name": str(path),
        "L": L,
        "W": W,
        "H": H,
        "bin_volume": L * W * H,
        "boxes": boxes,
        "n": len(boxes),
        "shapes": box_shapes,
        "classic": classic,
        "volumes": [l * w * h for (l, w, h) in boxes],
        "total_box_volume": sum(l * w * h for (l, w, h) in boxes),
    }

    _precompute_fit_tables(inst)

    return inst


def _precompute_fit_tables(inst):
    """
    Cache, per box and variant, the orientations that fit the empty bin at all,
    and mark boxes that fit in no configuration whatsoever.

    An orientation too large for the empty bin can never be placed, so dropping
    it here removes it from every one of the ~1e5 decodes a run performs
    instead of re-testing it each time. `fits_at_all` is what lets the decoder
    skip a hopeless box in O(1).
    """

    L, W, H = inst["L"], inst["W"], inst["H"]

    feasible = []

    for shapes in inst["shapes"]:

        per_variant = []

        for orientations in shapes:
            per_variant.append(tuple(
                (dx, dy, dz) for (dx, dy, dz) in orientations
                if dx <= W and dy <= H and dz <= L
            ))

        feasible.append(per_variant)

    inst["feasible"] = feasible
    inst["fits_at_all"] = [any(v for v in per_variant) for per_variant in feasible]


# =========================================================================
# Bound
# =========================================================================
def volume_bound(inst):
    """
    Trivial upper bound on the objective (2): the utilization can exceed
    neither 1 nor the share of the bin the available box volume could fill.

    Compression preserves volume exactly in the continuous formula (1), so the
    total box volume is variant-independent and this bound holds for the 3DBPP
    and the 3DBPP-VBS alike. Rounding the edge lengths to integers perturbs it
    slightly, which is why it is a guide rather than a certificate: a decode
    that reaches it is optimal, but the converse does not hold.
    """

    if inst["bin_volume"] <= 0:
        return 0.0

    return min(1.0, inst["total_box_volume"] / inst["bin_volume"])


# =========================================================================
# Decoder: loading sequence -> feasible packing
# =========================================================================
def decode(inst, sol, ep_limit=EP_LIMIT):
    """
    Deepest-bottom-left extreme-point packer driven by the SA solution.

    Boxes are offered in sol["sequence"]; each is placed at the first extreme
    point (sorted by z, then y, then x) where its configuration — variant
    sol["variants"][i], starting from orientation preference
    sol["orientations"][i] — fits inside the bin without overlapping an
    already-placed box. A box that fits nowhere is skipped, exactly as p_i = 0
    in the model.

    Returns (placements, packed_volume).

    Extreme points are the corners opened up by already-placed boxes (Crainic
    et al. 2008); the list is kept sorted so that the packing is a deterministic
    function of the sequence, which is what lets SA compare two solutions at
    all.
    """

    L, W, H = inst["L"], inst["W"], inst["H"]
    feasible = inst["feasible"]
    fits_at_all = inst["fits_at_all"]
    volumes = inst["volumes"]
    variants = sol["variants"]
    orientations = sol["orientations"]

    # Extreme points as (z, y, x) so that plain sorted order IS the
    # deepest-then-lowest-then-leftmost order the packer wants; insort then
    # keeps the list sorted without re-sorting it after every placement.
    points = [(0, 0, 0)]

    # Placed boxes are kept as flat parallel lists of their extents rather than
    # as dicts. This function is the inner loop of the whole search (one full
    # call per SA iteration) and the overlap test runs points x placed times per
    # box, so attribute and key lookups there dominate the runtime. The dicts
    # are built once at the end instead.
    px0, py0, pz0 = [], [], []
    px1, py1, pz1 = [], [], []

    placed_boxes = []
    packed_volume = 0
    free_volume = inst["bin_volume"]

    for i in sol["sequence"]:

        if not fits_at_all[i]:
            continue

        if volumes[i] > free_volume:
            continue

        orients = feasible[i][variants[i]]

        nr = len(orients)
        if nr == 0:
            continue

        start = orientations[i] % nr

        hit = None

        for scanned, (qz, qy, qx) in enumerate(points):

            if ep_limit and scanned >= ep_limit:
                break

            for r in range(nr):

                dx, dy, dz = orients[(start + r) % nr]

                ax1 = qx + dx
                ay1 = qy + dy
                az1 = qz + dz

                if ax1 > W or ay1 > H or az1 > L:
                    continue

                # Two boxes overlap only if they overlap on all three axes.
                ok = True

                for k in range(len(placed_boxes)):
                    if (qx < px1[k] and px0[k] < ax1
                            and qy < py1[k] and py0[k] < ay1
                            and qz < pz1[k] and pz0[k] < az1):
                        ok = False
                        break

                if ok:
                    hit = (qx, qy, qz, dx, dy, dz, (start + r) % nr)
                    break

            if hit is not None:
                break

        if hit is None:
            continue

        qx, qy, qz, dx, dy, dz, r = hit

        px0.append(qx)
        py0.append(qy)
        pz0.append(qz)
        px1.append(qx + dx)
        py1.append(qy + dy)
        pz1.append(qz + dz)

        placed_boxes.append((i, qx, qy, qz, dx, dy, dz, variants[i], r))

        packed_volume += dx * dy * dz
        free_volume -= dx * dy * dz

        points.remove((qz, qy, qx))

        # The three extreme points this box opens up. A point strictly inside
        # an already-placed box can never host anything, so it is dropped right
        # away instead of being rescanned for every later box.
        for nx, ny, nz in ((qx + dx, qy, qz), (qx, qy + dy, qz),
                           (qx, qy, qz + dz)):

            if nx >= W or ny >= H or nz >= L:
                continue

            p = (nz, ny, nx)

            if p in points:
                continue

            inside = False

            for k in range(len(placed_boxes)):
                if (px0[k] <= nx < px1[k] and py0[k] <= ny < py1[k]
                        and pz0[k] <= nz < pz1[k]):
                    inside = True
                    break

            if not inside:
                insort(points, p)

    placements = [
        {"box": b, "x": x, "y": y, "z": z, "dx": ex, "dy": ey, "dz": ez,
         "variant": v, "orientation": o}
        for (b, x, y, z, ex, ey, ez, v, o) in placed_boxes
    ]

    return placements, packed_volume


def evaluate(inst, sol, ep_limit=EP_LIMIT):
    """Decode a solution, store the packing on it, and return its utilization."""

    placements, volume = decode(inst, sol, ep_limit=ep_limit)

    sol["placements"] = placements
    sol["packed"] = set(pl["box"] for pl in placements)
    sol["utilization"] = volume / inst["bin_volume"] if inst["bin_volume"] else 0.0

    return sol["utilization"]


# =========================================================================
# Solution representation
# =========================================================================
def copy_solution(sol):
    return {
        "sequence": list(sol["sequence"]),
        "orientations": list(sol["orientations"]),
        "variants": list(sol["variants"]),
        "placements": list(sol["placements"]),
        "packed": set(sol["packed"]),
        "utilization": sol["utilization"],
    }


def build_initial_solution(inst, ep_limit=EP_LIMIT):
    """
    Largest-volume-first sequence, every box in its original size (variant 1)
    and first orientation.

    Largest-first is the classic bin-packing construction: big boxes are the
    ones that need contiguous space, so they are placed while the bin is still
    empty. SA starts from that greedy packing and can only improve on it.
    """

    order = sorted(range(inst["n"]), key=lambda i: -inst["volumes"][i])

    sol = {
        "sequence": order,
        "orientations": [0] * inst["n"],
        "variants": [0] * inst["n"],
        "placements": [],
        "packed": set(),
        "utilization": 0.0,
    }

    evaluate(sol=sol, inst=inst, ep_limit=ep_limit)

    return sol


# =========================================================================
# Random single-move sampling
# =========================================================================
# Each sampler returns ONE randomly drawn move, or None when no move of that
# type can change the current solution. Two boxes with identical dimensions in
# identical configurations are indistinguishable to the decoder, so every
# sampler filters draws that would decode to the same packing; on instances
# with repeated box types (BR expands each type by its Demand, so they are the
# common case) that filter is what keeps iterations from being spent
# re-decoding the solution SA already has.

def _key(inst, sol, pos):
    """What the decoder can actually tell apart at a position in the sequence."""

    i = sol["sequence"][pos]
    v = sol["variants"][i]
    orients = inst["feasible"][i][v]

    return (inst["boxes"][i], v,
            sol["orientations"][i] % len(orients) if orients else 0)


def _is_noop_span(inst, sol, src, dst):
    """
    True if moving the box at `src` to `dst` cannot change the packing.

    It cannot when every box it jumps over is indistinguishable from it.
    """

    if src == dst:
        return True

    key = _key(inst, sol, src)
    lo, hi = (src + 1, dst) if dst > src else (dst, src - 1)

    return all(_key(inst, sol, p) == key for p in range(lo, hi + 1))


def sample_relocate_move(inst, sol):
    """Move one box to another position in the loading sequence."""

    n = inst["n"]

    if n < 2:
        return None

    for _ in range(_SAMPLE_TRIES):

        src = random.randrange(n)
        dst = random.randrange(n)

        if _is_noop_span(inst, sol, src, dst):
            continue

        return {"type": "relocate", "src": src, "dst": dst}

    return None


def sample_swap_move(inst, sol):
    """Exchange the sequence positions of two boxes."""

    n = inst["n"]

    if n < 2:
        return None

    for _ in range(_SAMPLE_TRIES):

        a = random.randrange(n)
        b = random.randrange(n)

        if a == b or _key(inst, sol, a) == _key(inst, sol, b):
            continue

        return {"type": "swap", "a": a, "b": b}

    return None


def sample_2opt_move(inst, sol):
    """
    Reverse a segment of the loading sequence.

    There is no tour here, but reversing a segment is the natural large-step
    perturbation: it flips the loading precedence of a whole block of boxes at
    once, which no single relocate can do.
    """

    n = inst["n"]

    if n < 3:
        return None

    for _ in range(_SAMPLE_TRIES):

        i = random.randrange(n - 1)
        j = random.randrange(i + 1, n)

        keys = {_key(inst, sol, p) for p in range(i, j + 1)}

        if len(keys) < 2:      # homogeneous block — reversing changes nothing
            continue

        return {"type": "2opt", "i": i, "j": j}

    return None


def sample_promote_move(inst, sol):
    """
    Pull a currently UNPACKED box forward.

    Moving it ahead of boxes that are currently in the bin is what lets it
    displace them; whether that trade pays is whatever the decoder makes of it.
    """

    sequence = sol["sequence"]

    unpacked = [p for p, i in enumerate(sequence)
                if i not in sol["packed"] and inst["fits_at_all"][i]]

    if not unpacked:
        return None

    random.shuffle(unpacked)

    for src in unpacked[:_SAMPLE_TRIES]:

        if src == 0:
            continue

        dst = random.randrange(src)

        if _is_noop_span(inst, sol, src, dst):
            continue

        return {"type": "promote", "src": src, "dst": dst}

    return None


def sample_demote_move(inst, sol):
    """Push a currently PACKED box back, freeing the space it holds."""

    sequence = sol["sequence"]
    n = inst["n"]

    packed = [p for p, i in enumerate(sequence) if i in sol["packed"]]

    if not packed:
        return None

    random.shuffle(packed)

    for src in packed[:_SAMPLE_TRIES]:

        if src == n - 1:
            continue

        dst = random.randrange(src + 1, n)

        if _is_noop_span(inst, sol, src, dst):
            continue

        return {"type": "demote", "src": src, "dst": dst}

    return None


def sample_orient_move(inst, sol):
    """
    Give one box a different orthogonal orientation.

    This does not touch the sequence at all, only how the decoder first tries
    to lay the box down, which decides what shape of free space is left behind.
    """

    n = inst["n"]

    for _ in range(_SAMPLE_TRIES):

        i = random.randrange(n)

        nr = len(inst["feasible"][i][sol["variants"][i]])

        if nr < 2:
            continue

        new = random.randrange(nr)

        if new == sol["orientations"][i] % nr:
            continue

        return {"type": "orient", "box": i, "pref": new}

    return None


def sample_variant_move(inst, sol):
    """
    Switch one box to a different size variant (Section 3.1).

    This is the operator specific to the 3DBPP-VBS. Compression preserves
    volume, so the move never changes what the box contributes to the
    objective — it changes the SHAPE of that volume, trading height for
    footprint, and so what still fits around it.
    """

    n = inst["n"]

    for _ in range(_SAMPLE_TRIES):

        i = random.randrange(n)

        n_variants = len(inst["shapes"][i])

        if n_variants < 2:
            continue

        k = random.randrange(n_variants)

        if k == sol["variants"][i]:
            continue

        if not inst["feasible"][i][k]:   # this variant fits the bin in no
            continue                     # orientation — never worth trying

        return {"type": "variant", "box": i, "variant": k}

    return None


_SAMPLERS = {
    "relocate": sample_relocate_move,
    "swap":     sample_swap_move,
    "2opt":     sample_2opt_move,
    "promote":  sample_promote_move,
    "demote":   sample_demote_move,
    "orient":   sample_orient_move,
    "variant":  sample_variant_move,
}


def sample_random_move(inst, sol, move_weights=None):
    """
    Draw ONE random neighbour: pick a random operator, then a random move.

    Operators are drawn without replacement (by weight) so that if the chosen
    one cannot produce a move in the current solution, the next one is tried
    instead of wasting the whole iteration. Returns None only if no operator
    yields a move.
    """

    if move_weights is None:
        move_weights = MOVE_WEIGHTS

    pool = [(name, w) for name, w in move_weights.items()
            if w > 0 and name in _SAMPLERS]

    # With fixed sizes there is only one variant, so that operator can never
    # fire; dropping it keeps the remaining weights meaningful.
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


# -------------------------
# Apply a move to a solution (in-place)
# -------------------------
def apply_move(inst, sol, move):

    mtype = move["type"]

    if mtype in ("relocate", "promote", "demote"):
        sequence = sol["sequence"]
        i = sequence.pop(move["src"])
        sequence.insert(move["dst"], i)

    elif mtype == "swap":
        sequence = sol["sequence"]
        a, b = move["a"], move["b"]
        sequence[a], sequence[b] = sequence[b], sequence[a]

    elif mtype == "2opt":
        sequence = sol["sequence"]
        i, j = move["i"], move["j"]
        sequence[i:j + 1] = sequence[i:j + 1][::-1]

    elif mtype == "orient":
        sol["orientations"][move["box"]] = move["pref"]

    elif mtype == "variant":
        i, k = move["box"], move["variant"]
        sol["variants"][i] = k
        # A compressed variant can have fewer unique orientations than the
        # original (compression can make two edges equal), so the preference is
        # clamped into the new variant's range rather than left dangling.
        sol["orientations"][i] = min(sol["orientations"][i],
                                     len(inst["feasible"][i][k]) - 1)


# -------------------------
# Diversification
# -------------------------
def _diversify(inst, sol, ep_limit=EP_LIMIT):
    """
    Shake the current solution after prolonged stagnation: promote rejected
    cargo, scramble a block of the sequence, and re-randomize some
    configurations.

    Promoting the rejected boxes is the part that matters on instances where
    not everything fits — it is the only way a box that has been at the tail of
    the sequence for thousands of iterations gets another chance.
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
        i = random.randrange(n - seg + 1)
        block = sequence[i:i + seg]
        random.shuffle(block)
        sequence[i:i + seg] = block

    for _ in range(max(1, n // 5)):

        i = random.randrange(n)

        if not inst["classic"]:
            candidates = [k for k in range(len(inst["shapes"][i]))
                          if inst["feasible"][i][k]]
            if candidates:
                sol["variants"][i] = random.choice(candidates)

        nr = len(inst["feasible"][i][sol["variants"][i]])

        if nr:
            sol["orientations"][i] = random.randrange(nr)

    evaluate(inst, sol, ep_limit=ep_limit)

    return sol


# =========================================================================
# Simulated Annealing main loop
# =========================================================================
def simulated_annealing(inst, max_iterations=MAX_ITERATIONS,
                        time_limit=TIME_LIMIT, t_init=T_INIT, alpha=ALPHA,
                        t_min=T_MIN, reheat_threshold=None,
                        reheat_fraction=REHEAT_FRACTION, ep_limit=EP_LIMIT,
                        seed=None, verbose=True):
    """
    Simulated Annealing for the 3DBPP-VBS over decoder inputs.

    Per iteration exactly ONE random neighbour is drawn (random operator +
    random move), decoded into a feasible packing, and then accepted or
    rejected. No neighbourhood is enumerated.

    Acceptance criterion (the objective, utilization, is MAXIMIZED):
      - Improving moves (delta > 0): always accepted
      - Worsening moves: accepted with probability exp(delta / T)
    Temperature schedule:
      - Geometric cooling: T *= alpha every iteration (also on rejection)
      - Reheating to reheat_fraction * t_init after prolonged stagnation,
        combined with a diversification shake

    There is no packing-feasibility rejection: the decoder leaves unfittable
    boxes out, so every candidate is a valid packing and no iteration is ever
    wasted on an infeasible draw.
    """

    if seed is not None:
        random.seed(seed)

    t_start = time.time()
    deadline = t_start + time_limit

    if reheat_threshold is None:
        reheat_threshold = default_reheat_threshold(alpha)

    # ---- Initial solution ----
    current = build_initial_solution(inst, ep_limit=ep_limit)
    initial_utilization = current["utilization"]

    if verbose:
        print(f"  initial (largest-first) decode: utilization "
              f"{initial_utilization * 100:.2f}% "
              f"({len(current['packed'])}/{inst['n']} boxes)")

    best = copy_solution(current)

    t_reheat = reheat_fraction * t_init

    T = t_init
    no_improve = 0
    accepted = rejected = no_move = reheats = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):

        if time.time() >= deadline:
            if verbose:
                print(f"  time limit reached at iteration {iteration}")
            break

        # ---- Draw ONE random neighbour ----
        move = sample_random_move(inst, current)

        if move is None:
            # Nothing can change this solution at all — shake it and retry.
            no_move += 1
            no_improve += 1

            if no_improve >= reheat_threshold:
                T = t_reheat
                _diversify(inst, current, ep_limit=ep_limit)
                no_improve = 0
                reheats += 1

            T = max(T * alpha, t_min)
            continue

        # ---- Decode it: every neighbour is feasible by construction ----
        trial = copy_solution(current)
        apply_move(inst, trial, move)

        delta = evaluate(inst, trial, ep_limit=ep_limit) - current["utilization"]

        # ---- Metropolis acceptance ----
        if delta > 0:
            accept = True                       # improving move — always accept
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

                if verbose:
                    print(f"    [iter {iteration}] *** NEW BEST: utilization "
                          f"{best['utilization'] * 100:.2f}%, boxes "
                          f"{len(best['packed'])}/{inst['n']}, T={T:.6f}, "
                          f"elapsed={time.time() - t_start:.1f}s")
            else:
                no_improve += 1

        else:
            rejected += 1
            no_improve += 1

        # ---- Cool down once per iteration, whatever happened ----
        T = max(T * alpha, t_min)

        # ---- Reheat + diversify on prolonged stagnation ----
        if no_improve >= reheat_threshold:

            T = t_reheat
            _diversify(inst, current, ep_limit=ep_limit)
            no_improve = 0
            reheats += 1

            if verbose:
                print(f"    [iter {iteration}] reheat + diversify: T={T:.6f}, "
                      f"utilization={current['utilization'] * 100:.2f}%")

    elapsed = time.time() - t_start

    if verbose:
        acc_rate = accepted / max(1, accepted + rejected)
        print(f"  SA finished: {iteration} iterations in {elapsed:.1f}s "
              f"({iteration / elapsed if elapsed > 0 else 0:.0f} it/s), "
              f"accepted {accepted} ({acc_rate:.1%}), rejected {rejected}, "
              f"no-move draws {no_move}, reheats {reheats}")

    return {
        "utilization": best["utilization"],
        "placements": best["placements"],
        "solution": best,
        "iterations": iteration,
        "runtime": elapsed,
        "accepted": accepted,
        "rejected": rejected,
        "no_move": no_move,
        "reheats": reheats,
        "t_init": t_init,
        "alpha": alpha,
        "reheat_threshold": reheat_threshold,
        "initial_utilization": initial_utilization,
        "iters_per_sec": iteration / elapsed if elapsed > 0 else 0.0,
    }


# =========================================================================
# Verification and reporting
# =========================================================================
def verify_solution(inst, placements):
    """
    Independent geometric check: every box inside the bin, no two boxes
    overlapping, no box placed twice, and every placement using a configuration
    the box actually has. Mirrors the check in 3DBPP-VBS.py so that both
    solvers are validated the same way.
    """

    L, W, H = inst["L"], inst["W"], inst["H"]

    violations = []
    seen = set()

    for pl in placements:

        if pl["box"] in seen:
            violations.append(f"box {pl['box']} placed more than once")

        seen.add(pl["box"])

        if pl["x"] + pl["dx"] > W or pl["y"] + pl["dy"] > H \
                or pl["z"] + pl["dz"] > L:
            violations.append(f"box {pl['box']} exceeds the bin")

        if min(pl["x"], pl["y"], pl["z"]) < 0:
            violations.append(f"box {pl['box']} has a negative coordinate")

        shapes = inst["shapes"][pl["box"]]

        if not (0 <= pl["variant"] < len(shapes)):
            violations.append(f"box {pl['box']} uses an unknown variant")
        elif (pl["dx"], pl["dy"], pl["dz"]) not in shapes[pl["variant"]]:
            violations.append(
                f"box {pl['box']} uses edges that are not a configuration "
                f"of its variant {pl['variant'] + 1}"
            )

    for a in range(len(placements)):
        for b in range(a + 1, len(placements)):

            i, j = placements[a], placements[b]

            if (i["x"] < j["x"] + j["dx"] and j["x"] < i["x"] + i["dx"]
                    and i["y"] < j["y"] + j["dy"] and j["y"] < i["y"] + i["dy"]
                    and i["z"] < j["z"] + j["dz"]
                    and j["z"] < i["z"] + i["dz"]):
                violations.append(f"boxes {i['box']} and {j['box']} overlap")

    return violations


def print_report(inst, result, classic):
    """Print a summary of one solved instance."""

    problem = "3DBPP (fixed sizes)" if classic else "3DBPP-VBS"

    ratio = (inst["total_box_volume"] / inst["bin_volume"] * 100
             if inst["bin_volume"] else 0.0)

    note = "  <- caps the utilization" if ratio < 100 else ""

    unfittable = sum(1 for ok in inst["fits_at_all"] if not ok)

    print()
    print("=" * 66)
    print(f"  {problem} SA   instance: {inst['name']}")
    print("=" * 66)
    print(f"  bin (L x W x H)   : {inst['L']} x {inst['W']} x {inst['H']}")
    print(f"  boxes             : {inst['n']}")
    print(f"  total box volume  : {ratio:.2f} % of bin{note}")

    if unfittable:
        print(f"  note              : {unfittable} box(es) fit the empty bin "
              f"in no configuration - never packable")

    print(f"  iterations        : {result['iterations']} "
          f"({result['iters_per_sec']:.0f} it/s)")
    print(f"  runtime           : {result['runtime']:.2f} s")
    print(f"  packed boxes      : {len(result['placements'])} / {inst['n']}")
    print(f"  greedy (initial)  : "
          f"{result['initial_utilization'] * 100:.2f} %")
    print(f"  space utilization : {result['utilization'] * 100:.2f} % "
          f"({(result['utilization'] - result['initial_utilization']) * 100:+.2f} "
          f"over greedy)")

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

    # The axes carry the bin's own bounds: X is the width W, Y the height H
    # and Z the length L, as in the MILP's constraints (4)-(6).
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, L)
    ax.set_xlabel("X (width W)")
    ax.set_ylabel("Y (height H)")
    ax.set_zlabel("Z (length L)")
    ax.set_box_aspect((W, H, L))

    ax.set_title(f"{inst['name']}\n"
                 f"{len(result['placements'])}/{inst['n']} boxes, "
                 f"utilization {result['utilization'] * 100:.2f}%")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  plot saved to     : {out_path}")


# =========================================================================
# Driver
# =========================================================================
def solve_one(path, classic, args, verbose=True):
    """
    Prepare an instance and run SA on it --runs times, returning the best run.

    Restarts are independent: SA is a stochastic search whose outcome varies
    with the seed, so on a fixed time budget several short runs often beat one
    long one. --runs 1 (the default) spends the whole budget on a single run.
    """

    inst = prepare_instance(path, max_boxes=args.max_boxes, classic=classic)

    best = None

    for r in range(args.runs):

        seed = None if args.seed is None else args.seed + r

        if args.greedy_only:

            t0 = time.time()
            sol = build_initial_solution(inst, ep_limit=args.ep_limit)

            result = {
                "utilization": sol["utilization"],
                "placements": sol["placements"],
                "solution": sol,
                "iterations": 0,
                "runtime": time.time() - t0,
                "accepted": 0, "rejected": 0, "no_move": 0, "reheats": 0,
                "t_init": args.t_init, "alpha": args.alpha,
                "reheat_threshold": 0,
                "initial_utilization": sol["utilization"],
                "iters_per_sec": 0.0,
            }

        else:

            result = simulated_annealing(
                inst,
                max_iterations=args.max_iterations,
                time_limit=args.time_limit,
                t_init=args.t_init,
                alpha=args.alpha,
                t_min=args.t_min,
                reheat_threshold=args.reheat_threshold,
                reheat_fraction=args.reheat_fraction,
                ep_limit=args.ep_limit,
                seed=seed,
                verbose=verbose,
            )

        if best is None or result["utilization"] > best["utilization"]:
            best = result

    return inst, best


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Simulated annealing for the three-dimensional bin "
                    "packing problem with variable box size (Xu et al. 2026)."
    )

    parser.add_argument(
        "--instance",
        default="Martello/class_1/class1_n10_instance1.txt",
        help="path to an instance: a BR/Martello text file, or one of the "
             "original BR *.json files "
             "(default: Martello/class_1/class1_n10_instance1.txt)",
    )
    parser.add_argument(
        "--max-boxes", type=int, default=0,
        help="use only the first k boxes of the instance (default: 0 = all)",
    )
    parser.add_argument(
        "--classic", action="store_true",
        help="solve the standard 3DBPP with fixed box sizes (variant 1 only) "
             "instead of the 3DBPP-VBS",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="solve both the 3DBPP and the 3DBPP-VBS and report the "
             "improvement Diff of formula (18)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=TIME_LIMIT,
        help=f"SA seconds per run (default: {TIME_LIMIT})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help="maximum SA iterations per run",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="independent SA runs per problem; the best is reported "
             "(default: 1)",
    )
    parser.add_argument(
        "--t-init", type=float, default=T_INIT,
        help=f"initial temperature (default: {T_INIT}; the objective is a "
             f"ratio in [0, 1], so this scale is instance-independent)",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"geometric cooling factor (default: {ALPHA})",
    )
    parser.add_argument(
        "--t-min", type=float, default=T_MIN,
        help=f"minimum temperature (default: {T_MIN})",
    )
    parser.add_argument(
        "--reheat-threshold", type=int, default=REHEAT_THRESHOLD,
        help="iterations without a new best before reheating (default: "
             "three quarters of a cooling sweep at the given alpha)",
    )
    parser.add_argument(
        "--reheat-fraction", type=float, default=REHEAT_FRACTION,
        help=f"T is reset to this fraction of T0 on a reheat "
             f"(default: {REHEAT_FRACTION})",
    )
    parser.add_argument(
        "--ep-limit", type=int, default=EP_LIMIT,
        help="max extreme points scanned per box (0 = unlimited); a small cap "
             "trades packing quality for iterations on large instances",
    )
    parser.add_argument(
        "--greedy-only", action="store_true",
        help="only decode the initial largest-first sequence, no SA",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible runs",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the per-improvement SA log",
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

    args = parser.parse_args(argv)

    if args.seed is None:
        args.seed = random.randrange(2 ** 31)

    random.seed(args.seed)

    if args.batch:
        return run_batch(args)

    path = Path(args.instance)

    if not path.exists():
        print(f"instance not found: {path}", file=sys.stderr)
        return 1

    reheat = (args.reheat_threshold if args.reheat_threshold is not None
              else default_reheat_threshold(args.alpha))

    print("=" * 66)
    print("3DBPP-VBS Simulated Annealing (random single-neighbour, "
          "sequence decoder)")
    print(f"SA params: T0={args.t_init}, alpha={args.alpha}, "
          f"T_min={args.t_min}, reheat_threshold={reheat}, "
          f"reheat_fraction={args.reheat_fraction}")
    print(f"Move weights: {MOVE_WEIGHTS}")
    print(f"Time limit: {args.time_limit}s per run, {args.runs} run(s), "
          f"seed {args.seed}")
    print("=" * 66)

    modes = [True, False] if args.both else [args.classic]

    results = {}

    for classic in modes:

        inst, best = solve_one(path, classic, args, verbose=not args.quiet)

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

    return 0


def run_batch(args):
    """
    Solve a list of instances in sequence and summarize the results.

    With --both, each instance is solved as 3DBPP and as 3DBPP-VBS and the
    improvement Diff of formula (18) is reported per instance and on average.
    """

    import csv

    rows = []

    if args.both:
        header = (f"  {'instance':<24} {'n':>4} "
                  f"{'3DBPP%':>8} {'VBS%':>8} {'Diff%':>8} {'time(s)':>9}")
    else:
        header = (f"  {'instance':<24} {'n':>4} {'packed':>7} "
                  f"{'greedy%':>8} {'util%':>8} {'iters':>9} {'time(s)':>9}")

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
            solved[classic] = solve_one(path, classic, args, verbose=False)

        full_name = f"{path.parent.name}/{path.stem}"
        name = full_name if len(full_name) <= 24 else "~" + full_name[-23:]

        if args.both:

            inst = solved[True][0]

            r1 = solved[True][1]["utilization"]
            r2 = solved[False][1]["utilization"]

            diff = (r2 - r1) / r1 * 100 if r1 > 0 else float("nan")
            total_time = solved[True][1]["runtime"] + solved[False][1]["runtime"]

            print(f"  {name:<24} {inst['n']:>4} {r1 * 100:>8.2f} "
                  f"{r2 * 100:>8.2f} {diff:>8.2f} {total_time:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": inst["n"],
                "util_3dbpp": r1 * 100,
                "util_3dbpp_vbs": r2 * 100,
                "diff_pct": diff,
                "time_s": total_time,
            })

        else:

            inst, res = solved[args.classic]

            print(f"  {name:<24} {inst['n']:>4} "
                  f"{len(res['placements']):>7} "
                  f"{res['initial_utilization'] * 100:>8.2f} "
                  f"{res['utilization'] * 100:>8.2f} "
                  f"{res['iterations']:>9} {res['runtime']:>9.2f}")

            rows.append({
                "instance": full_name,
                "boxes": inst["n"],
                "packed": len(res["placements"]),
                "greedy_pct": res["initial_utilization"] * 100,
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
            avg_g = sum(r["greedy_pct"] for r in rows) / len(rows)
            avg = sum(r["util_pct"] for r in rows) / len(rows)

            print(f"  {'average':<24} {'':>4} {'':>7} "
                  f"{avg_g:>8.2f} {avg:>8.2f}")

        print()

    if args.csv and rows:

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"  results written to {args.csv}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
