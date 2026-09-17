"""
Weight First Best Fit (WFBF) of Deplano et al. (2019), Section 4
===============================================================

Faithful implementation of Algorithms 1 and 2 of

    Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
    "A mixed-integer linear model for the multiple heterogeneous knapsack
     problem with realistic container loading constraints and bins' priority"
    Intl. Trans. in Op. Res. 28(6), 3244-3275.

WFBF is the paper's own constructive heuristic, the counterpart to its MIP. It
exists here so that our SA (MHKP-V0-SA.py) can be compared against the paper's
method ON THE SAME INSTANCES - the only comparison that means anything, since
the paper's instances are not published and its objective values therefore
cannot be read against ours.

Scope: submodel V0
------------------
This implements WFBF under V0, constraints (1)-(10r): geometry, the bin
priority objective, and static stability. Algorithm 1 line 21 also calls
isLoadBearingValid and isCDMValid; both belong to V1/V2, which introduce weight,
arbitrary centres of mass and load-bearing limits. Under V0 those predicates are
vacuously true, so they are omitted rather than stubbed, and the file says so
instead of pretending to a generality it does not have.

The algorithm
-------------
Items are sorted by WEIGHT descending (Algorithm 1, "Require"). V0 carries no
weights, and the paper's items draw weight from volume (Section 5, equation
(17): the item's maximal weight is proportional to its volume). Sorting by
volume descending is therefore the faithful V0 reading of "by weight
descending", and it is what `sort_key="volume"` does. `sort_key="none"` keeps
the file order, for isolating the effect of that sort.

Bins are filled ONE AT A TIME in randomised-priority order (line 10). For each
bin, every remaining item is offered in turn; for each item every (position,
rotation, reflection) triple is scored by the ranking function (line 20)

    rank = ( z_i + |tau_i^{x,y} - theta_j^{x,y}| / 2 ) ^ 2

and the feasible triple with the SMALLEST rank wins (first best fit). The rank
rewards placements that are low in the bin and near the vertical axis through
the bin's centre. An item with no feasible placement goes to `discarded` and is
retried in the NEXT bin (line 30, 33) - never in this one, which is the source
of the weakness the authors document.

Reflection (r in {0, 1}, line 19) is a pi-radian rotation that affects only an
item's centre of mass. With homogeneous items - V0 has no CoM data - it does not
change geometry or rank, so the loop over r collapses to a single pass. It is
kept in the signature and documented rather than silently dropped.

Position enumeration
--------------------
Line 15 enumerates ALL positions in the bin, of cardinality H*W*L/beta. On the
paper's Physical Internet instances the modular dimensions give a large beta and
this is tractable. On our instances beta is the gcd of the item sides, which is
typically 1, giving 120^3 = 1.7 million positions per item per rotation - far
beyond what can be run. Two enumerations are therefore provided:

  mode="grid"    the literal Algorithm 1: every position on the beta-grid.
                 Faithful, and the one to quote, but needs a coarse beta (pass
                 `grid_step`) to finish in reasonable time.

  mode="corner"  positions restricted to corner points: the origin, and the
                 three corners opened by each placed item. The ranking function
                 and the selection rule are untouched.

`mode="corner"` is a near-equivalence, not an identity: the rank falls with
lower z AND with proximity to the bin axis, and those two can conflict, so a
grid position that is off-corner can in principle win. It is offered because it
makes WFBF runnable at beta = 1, and the two are compared directly in the
experiments rather than assumed interchangeable.

Multi-start
-----------
`randomisedOrderWithPriority` (Section 4) partitions the bins into equal-priority
classes, shuffles within each class, and flattens in priority order. It is only
useful when a class holds bins of equal priority but different shapes, which is
what `is_randomizable` tests (line 4); when it is false the paper sets
repeatLimit to 1, and so does this.
"""

import argparse
import random
from pathlib import Path


def rotations(dims):
    """
    The two orthogonal rotations of Table 2, as (dx, dy, dz).

    Rotation is around the vertical axis only: length and width may swap, the
    height is invariant. Duplicates (a square footprint) are removed.
    """

    l, w, h = dims

    return list(dict.fromkeys([(w, h, l), (l, h, w)]))


def is_randomizable(bins):
    """
    Algorithm 1 line 4.

    True if some priority class holds at least two bins of DIFFERENT shape;
    shuffling identical bins cannot change anything.
    """

    classes = {}

    for (L, W, H, p) in bins:
        classes.setdefault(p, []).append((L, W, H))

    for shapes in classes.values():
        if len(shapes) >= 2 and len(set(shapes)) >= 2:
            return True

    return False


def randomised_order_with_priority(bins, rng):
    """
    Section 4: partition into equal-priority classes, shuffle within each
    class, flatten in priority order.

    The paper sorts "randomised priority ascending" when p_j < 1 and descending
    when p_j >= 1. With the case-study priority p_j = 1/volume every priority is
    far below 1, so ascending priority applies - which puts the LARGEST bins
    first by priority value, i.e. the smallest volume last. The paper's stated
    intent throughout is the opposite: fill SMALL bins first, and it describes
    the first bin as "the one with the lowest volume". Sorting by DESCENDING
    priority delivers that, so that is what is implemented, matching the prose
    and the worked example rather than the sign convention in the sentence.
    """

    classes = {}

    for b in bins:
        classes.setdefault(b[3], []).append(b)

    order = []

    for priority in sorted(classes, reverse=True):     # smallest volume first
        group = list(classes[priority])
        rng.shuffle(group)
        order.extend(group)

    return order


def enumerate_positions(L, W, H, step):
    """
    Algorithm 1 line 15, the literal reading: every position on the grid.

    Yields (x, y, z) with x along W, y along H, z along L, ordered so that low
    y (low in the bin) comes first - which matches the direction the ranking
    function prefers and makes the scan cache-friendly, without changing which
    placement wins.
    """

    for y in range(0, H, step):
        for z in range(0, L, step):
            for x in range(0, W, step):
                yield (x, y, z)


def corner_positions(placed, L, W, H, beta):
    """
    Positions restricted to corner points: the origin plus the three corners
    each placed item opens. See the module docstring on why this is offered.

    `beta` is the instance's own grid (Definition 4), NOT the coarseness used
    to make grid enumeration tractable. Corner points are induced by the items
    already placed, so with beta = 1 - the usual case here - every corner is
    admissible and none is filtered. Conflating the two silently discards
    legitimate corners and cripples the heuristic.
    """

    pts = {(0, 0, 0)}

    for (px, py, pz, pdx, pdy, pdz) in placed:
        for p in ((px + pdx, py, pz), (px, py + pdy, pz), (px, py, pz + pdz)):
            if p[0] < W and p[1] < H and p[2] < L:
                if p[0] % beta == 0 and p[2] % beta == 0:
                    pts.add(p)

    # Sorted so that ties in rank resolve deterministically.
    return sorted(pts, key=lambda p: (p[1], p[2], p[0]))


def _overlaps(x, y, z, dx, dy, dz, placed):
    for (px, py, pz, pdx, pdy, pdz) in placed:
        if (x < px + pdx and px < x + dx
                and y < py + pdy and py < y + dy
                and z < pz + pdz and pz < z + dz):
            return True
    return False


def _is_supported(x, y, z, dx, dz, placed):
    """
    Constraints (10a)-(10r), the same rule MHKP-V0-SA.py enforces.

    On the floor, or resting with exact surface contact on items that together
    cover at least one of the two x-side bottom corners AND one of the two
    y-side bottom corners.
    """

    if y == 0:
        return True

    corners = [(x, z), (x + dx, z), (x, z + dz), (x + dx, z + dz)]
    covered = [False] * 4

    for (px, py, pz, pdx, pdy, pdz) in placed:

        if py + pdy != y:
            continue

        for idx, (cx, cz) in enumerate(corners):
            if not covered[idx] and px <= cx <= px + pdx and pz <= cz <= pz + pdz:
                covered[idx] = True

    return (covered[0] or covered[1]) and (covered[2] or covered[3])


def rank_of(x, y, z, dx, dy, dz, bin_dims):
    """
    The ranking function, Algorithm 1 line 20:

        rank = ( z_i + |tau_i^{x,y} - theta_j^{x,y}| / 2 ) ^ 2

    tau is the item's centre of mass in the bin's frame; with homogeneous items
    that is its geometric centre. theta is the vertex of the bin's feasible CoM
    region, which the paper suggests placing at the bin's centre (Section 3.1).

    NOTE on axes: the paper's z_i is the coordinate of the item's bottom face,
    i.e. the vertical one. In this file's convention the vertical axis is y, so
    y enters the rank where the paper writes z. The horizontal distance is taken
    in the plane orthogonal to the bin floor, which is the (x, z) plane here.
    """

    L, W, H = bin_dims

    cx = x + dx / 2.0
    cz = z + dz / 2.0

    theta_x = W / 2.0
    theta_z = L / 2.0

    horizontal = ((cx - theta_x) ** 2 + (cz - theta_z) ** 2) ** 0.5

    return (y + horizontal / 2.0) ** 2


def pack_one_bin(bin_dims, items, item_ids, mode, step, beta=1):
    """
    Algorithm 1 lines 14-32: fill ONE bin, returning (placed, discarded).

    Items are offered in the order given; each takes its best-ranked feasible
    placement, or is discarded for the next bin.
    """

    L, W, H, _p = bin_dims
    dims3 = (L, W, H)

    placed = []          # (x, y, z, dx, dy, dz)
    result = []          # (item_id, x, y, z, dx, dy, dz, rotation)
    discarded = []

    for idx, i in enumerate(item_ids):

        best = float("inf")
        winning = None

        if mode == "corner":
            positions = corner_positions(placed, L, W, H, beta)
        else:
            positions = list(enumerate_positions(L, W, H, step))

        for r, (dx, dy, dz) in enumerate(rotations(items[i])):

            if dx > W or dy > H or dz > L:
                continue

            for (x, y, z) in positions:

                if x + dx > W or y + dy > H or z + dz > L:
                    continue

                score = rank_of(x, y, z, dx, dy, dz, dims3)

                # First best fit: only test feasibility if the rank can win,
                # which is the order Algorithm 1 line 21 uses too.
                if score >= best:
                    continue

                if _overlaps(x, y, z, dx, dy, dz, placed):
                    continue

                if not _is_supported(x, y, z, dx, dz, placed):
                    continue

                best = score
                winning = (x, y, z, dx, dy, dz, r)

        if winning is None:
            discarded.append(i)
            continue

        x, y, z, dx, dy, dz, r = winning
        placed.append((x, y, z, dx, dy, dz))
        result.append((i, x, y, z, dx, dy, dz, r))

    return result, discarded


def wfbf(bins, items, max_limit=10, mode="corner", grid_step=None,
         sort_key="volume", seed=None):
    """
    Algorithms 1 and 2 in full.

    `bins` are (L, W, H, priority), `items` are (l, w, h) - exactly what
    mhkp_instances.read_instance returns.

    Returns a dict with the placements, the objective (1), and the count of
    items packed.
    """

    rng = random.Random(seed)

    # The instance's own grid, Definition 4. Corner points respect this; the
    # grid enumeration's coarseness is a separate, purely computational knob.
    beta = _grid_beta(items)

    if grid_step is None:
        grid_step = beta

    # "items sorted by weight in descending order, removing the items that do
    # not fit in any available bin" (Algorithm 1, Require).
    order = list(range(len(items)))

    if sort_key == "volume":
        order.sort(key=lambda i: -(items[i][0] * items[i][1] * items[i][2]))

    order = [i for i in order
             if any(_fits(items[i], b) for b in bins)]

    repeat_limit = max_limit if is_randomizable(bins) else 1

    best_solution = None
    best_objective = float("inf")

    for _repeat in range(repeat_limit):

        items_left = list(order)
        solution = []

        bin_order = randomised_order_with_priority(bins, rng)

        # Bin identity is needed for the objective, so carry the original index.
        index_of = {}
        used = {}
        for b in bin_order:
            for k, orig in enumerate(bins):
                if orig is b and k not in used:
                    index_of[id(b)] = k
                    used[k] = True
                    break

        for b in bin_order:

            if not items_left:
                break

            placed, discarded = pack_one_bin(b, items, items_left, mode,
                                             grid_step, beta)

            k = index_of[id(b)]

            for entry in placed:
                solution.append((k,) + entry)

            # Line 33: only the discarded items go on to the next bin.
            items_left = discarded

        objective = objective_1(bins, solution)

        if objective < best_objective:
            best_objective = objective
            best_solution = solution

    return {
        "placements": [
            {"bin": k, "item": i, "x": x, "y": y, "z": z,
             "dx": dx, "dy": dy, "dz": dz, "rotation": r}
            for (k, i, x, y, z, dx, dy, dz, r) in best_solution
        ],
        "objective": best_objective,
        "n_packed": len(best_solution),
        "repeats": repeat_limit,
    }


def _grid_beta(items):
    """Definition 4: beta is the gcd of all item sides."""

    from math import gcd

    beta = 0

    for dims in items:
        for d in dims:
            beta = gcd(beta, d)

    return max(1, beta)


def _fits(dims, bin_row):
    l, w, h = dims
    L, W, H, _ = bin_row
    return h <= H and ((l <= L and w <= W) or (w <= L and l <= W))


def objective_1(bins, solution):
    """
    Formula (1): sum_j p_j * (V_j - packed_j).

    With p_j = 1/V_j and a single bin this is the wasted fraction, which is the
    unit the paper's Tables 8 and 11 report.
    """

    packed = {}

    for (k, _i, _x, _y, _z, dx, dy, dz, _r) in solution:
        packed[k] = packed.get(k, 0) + dx * dy * dz

    total = 0.0

    for k, (L, W, H, p) in enumerate(bins):
        total += p * (L * W * H - packed.get(k, 0))

    return total


def verify(bins, items, placements):
    """Independent geometric and stability check, mirroring MHKP-V0-SA.verify."""

    errors = []
    seen = set()
    by_bin = {}

    for p in placements:

        if p["item"] in seen:
            errors.append(f"item {p['item']} placed more than once")
        seen.add(p["item"])

        by_bin.setdefault(p["bin"], []).append(p)

    for k, group in by_bin.items():

        L, W, H, _ = bins[k]

        for p in group:

            if p["x"] + p["dx"] > W or p["y"] + p["dy"] > H \
                    or p["z"] + p["dz"] > L:
                errors.append(f"item {p['item']} exceeds bin {k}")

            if min(p["x"], p["y"], p["z"]) < 0:
                errors.append(f"item {p['item']} has a negative coordinate")

            if (p["dx"], p["dy"], p["dz"]) not in rotations(items[p["item"]]):
                errors.append(f"item {p['item']} uses a disallowed rotation")

        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                if (i["x"] < j["x"] + j["dx"] and j["x"] < i["x"] + i["dx"]
                        and i["y"] < j["y"] + j["dy"] and j["y"] < i["y"] + i["dy"]
                        and i["z"] < j["z"] + j["dz"] and j["z"] < i["z"] + i["dz"]):
                    errors.append(
                        f"items {i['item']} and {j['item']} overlap in bin {k}")

        for p in group:
            others = [(q["x"], q["y"], q["z"], q["dx"], q["dy"], q["dz"])
                      for q in group if q is not p]
            if not _is_supported(p["x"], p["y"], p["z"], p["dx"], p["dz"], others):
                errors.append(f"item {p['item']} floats in bin {k}")

    return errors


def main():

    import mhkp_instances as mi

    parser = argparse.ArgumentParser(
        description="WFBF of Deplano et al. (2019) under submodel V0.")

    parser.add_argument("instances", nargs="+")
    parser.add_argument("--mode", choices=("corner", "grid"), default="corner",
                        help="position enumeration; 'grid' is the literal "
                             "Algorithm 1 and needs a coarse --grid-step")
    parser.add_argument("--grid-step", type=int, default=None,
                        help="grid spacing; default 1 (the paper's beta)")
    parser.add_argument("--max-limit", type=int, default=10,
                        help="multi-start repetitions (paper's maxLimit)")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    for spec in args.instances:

        bins, items = mi.read_instance(spec)

        res = wfbf(bins, items, max_limit=args.max_limit, mode=args.mode,
                   grid_step=args.grid_step, seed=args.seed)

        errs = verify(bins, items, res["placements"])

        print(f"{Path(spec).stem:<28} obj {res['objective']:.4f}  "
              f"packed {res['n_packed']}/{len(items)}  "
              f"{'ok' if not errs else str(len(errs)) + ' VIOLATIONS'}")


if __name__ == "__main__":
    main()
