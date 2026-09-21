#!/usr/bin/env python3
# Beam Search for the MHKP under submodel V2 of Deplano et al. (2019)
# =========================================================================
# V2 counterpart to MHKP-V0-BS.py and MHKP-V1-BS.py, and the heuristic
# counterpart to mhkp_v2_milp.py, for the problem of
#
#     Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
#     "A mixed-integer linear model for the multiple heterogeneous knapsack
#      problem with realistic container loading constraints and bins' priority"
#     Intl. Trans. in Op. Res. 28(6), 3244-3275.
#
#     V0  (1)-(10r)   geometry, priority objective, static stability
#     V1  (1)-(12n)   V0 + centre-of-mass distribution
#     V2  (1)-(13g)   V1 + load bearing
#
# This file reuses MHKP-V1-BS.py - which itself reuses MHKP-V0-BS.py - and
# changes only what load bearing changes.
#
# -------------------------------------------------------------------------
# LOAD BEARING AS A PLACEMENT TEST
# -------------------------------------------------------------------------
# Constraint (13g) says the weight an item carries may not exceed its
# capacity Lambda_i. The weight item i carries is the sum of the weights of
# the items whose CoM falls inside i's footprint and which sit at or above
# i's top face - not merely those touching it, because load accumulates down
# a column.
#
# Unlike the CoM constraint of V1, this one IS local enough to test at
# placement time: placing a new box k can only ever ADD load, and only to
# the boxes beneath it. So the test is "does placing k here overload
# anything under it", which is decided by walking the boxes below k in the
# same bin. That makes load bearing MONOTONE in the way stability is and
# the CoM is not: a packing that violates (13g) cannot be repaired by
# adding more boxes, since adding boxes only increases load. There is
# therefore no strict/relaxed distinction here - rejecting an overloading
# placement never rules out a completion that would have been feasible.
#
# The V1 CoM machinery is inherited unchanged, including its strict/relaxed
# switch and the fallback, because the CoM constraint's non-monotonicity is
# a property of V1 that V2 inherits along with it.
#
# -------------------------------------------------------------------------
# COST
# -------------------------------------------------------------------------
# The load test is O(items already in the bin) per candidate placement,
# because it must find every box under the candidate. That is the same order
# as the overlap test V0 already performs, so it does not change the
# complexity of a placement check - but it does roughly double its constant,
# which shows up directly as fewer rollouts in the same time budget.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#     python3 MHKP-V2-BS.py                        # default group
#     python3 MHKP-V2-BS.py --group t8_n20 --time-limit 3
#     python3 MHKP-V2-BS.py --factor 1.5           # weaker boxes
#     python3 MHKP-V2-BS.py --compare              # against V1 and V0

import argparse
import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path


def _load(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_bs1 = _load("mhkp_v1_bs", "MHKP-V1-BS.py")
_v2 = _load("mhkp_v2_milp", "mhkp_v2_milp.py")

_bs0 = _bs1._bs
_sa = _bs1._sa

load_instance = _sa.load_instance
objective_scale = _sa.objective_scale
wasted_space_objective = _sa.wasted_space_objective

bin_order = _bs0.bin_order
oriented_com = _bs1.oriented_com
com_ok = _bs1.com_ok
StateV1 = _bs1.StateV1

derive_v2_data = _v2.derive_v2_data
DEFAULT_ALPHA = _v2.DEFAULT_ALPHA
LOAD_BEARING_FACTOR = _v2.LOAD_BEARING_FACTOR

BEAM_WIDTH = _bs0.BEAM_WIDTH
MAX_ACTIONS = _bs0.MAX_ACTIONS
TIME_LIMIT = _bs0.TIME_LIMIT


# =========================================================================
# V2 instance data
# =========================================================================
def attach_v2_data(inst, bins_raw, items_raw, seed=1, alpha=DEFAULT_ALPHA,
                   factor=LOAD_BEARING_FACTOR):
    """
    Attach V1's weights, CoMs and pyramid, plus the load-bearing limits.

    Derived by mhkp_v2_milp.derive_v2_data so that the beam search and the
    MILP see identical data for a given instance, seed and factor.
    """

    omega, kappa, varrho, xi, Lambda = derive_v2_data(
        bins_raw, items_raw, seed=seed, alpha=alpha, factor=factor)

    inst["omega"] = omega
    inst["kappa"] = kappa
    inst["varrho"] = varrho
    inst["xi"] = xi
    inst["Lambda"] = Lambda

    return inst


# =========================================================================
# Load bearing, constraint (13g)
# =========================================================================
def _com_of(inst, item, dims, pos, rot, refl):
    """The absolute CoM of a placed box."""

    x, y, z = pos
    ox, oy, oz = oriented_com(inst["kappa"][item], dims, rot, refl)

    return x + ox, y + oy, z + oz


def load_ok(state, inst, k, item, dims, pos, rot, refl, tol=1e-6):
    """
    True if placing `item` here overloads anything, in either direction.

    Two things have to be checked, and checking only the first is a bug that
    silently produces infeasible packings:

      1. The candidate loads every box it sits OVER. Straightforward: charge
         its weight to each such box and compare against that box's capacity.

      2. The candidate may itself be sat over by boxes ALREADY PLACED. Boxes
         are not placed in bottom-up order - the free-space list is sorted by
         DBL, but a later placement can still land underneath an earlier one,
         for instance in a space beside a tall stack. So a new box can arrive
         beneath existing boxes and must absorb their weight immediately.

    Ignoring (2) is what made the search accept packings the verifier then
    rejected: the state reported a borne load of zero for boxes that were in
    fact carrying tens of millions, because nothing ever charged them.
    """

    cx, cy, _cz = _com_of(inst, item, dims, pos, rot, refl)

    x, y, z = pos
    dx, dy, dz = dims
    w = inst["omega"][item]

    top = z + dz

    incoming = 0.0

    for rec in state.loads.get(k, ()):
        # rec = [item, x, y, z, dx, dy, dz, borne, com_x, com_y]
        low_item, lx, ly, lz, ldx, ldy, ldz, borne, _ocx, _ocy = rec

        # (1) the candidate sits over this box.
        if (z >= lz + ldz - tol
                and lx - tol <= cx <= lx + ldx + tol
                and ly - tol <= cy <= ly + ldy + tol):

            if borne + w > inst["Lambda"][low_item] + tol:
                return False

        # (2) this box sits over the candidate. Its CoM is stored on the
        # record (fields 8 and 9) when it was placed, so no rotation or
        # reflection has to be re-derived here.
        elif lz >= top - tol:

            ocx, ocy = rec[8], rec[9]

            if (x - tol <= ocx <= x + dx + tol
                    and y - tol <= ocy <= y + dy + tol):
                incoming += inst["omega"][low_item]

    if incoming > inst["Lambda"][item] + tol:
        return False

    return True


class StateV2(StateV1):
    """
    A V1 state plus, per bin, the running load carried by each placed box.

    `loads[k]` is a list of mutable records, one per box in bin k:
        [item, x, y, z, dx, dy, dz, borne]

    Keeping the borne weight on the record makes the load test O(boxes in
    the bin) rather than O(boxes squared), which matters because the test
    sits in the innermost loop of the search.
    """

    __slots__ = ("loads",)

    def __init__(self, base, mass=None, wx=None, wy=None, wz=None,
                 loads=None):
        super().__init__(base, mass, wx, wy, wz)
        self.loads = loads if loads is not None else {}

    def copy(self):
        return StateV2(
            self.base.copy(), dict(self.mass), dict(self.wx), dict(self.wy),
            dict(self.wz),
            {k: [list(r) for r in v] for k, v in self.loads.items()},
        )


def initial_state_v2(inst):
    return StateV2(_bs0.initial_state(inst))


def can_place_v2(state, k, space, dims, inst, item, rot, refl, rule=None,
                 strict=True):
    """V1's test, plus load bearing."""

    pos = _bs1.can_place_v1(state, k, space, dims, inst, item, rot, refl,
                            rule=rule, strict=strict)

    if pos is None:
        return None

    if not load_ok(state, inst, k, item, dims, pos, rot, refl):
        return None

    return pos


def apply_action_v2(state, inst, k, space, item, dims, pos, rot, refl):
    """Place an item, updating the V1 state and the borne loads."""

    _bs1.apply_action_v1(state, inst, k, space, item, dims, pos, rot, refl)

    cx, cy, _cz = _com_of(inst, item, dims, pos, rot, refl)

    x, y, z = pos
    dx, dy, dz = dims
    w = inst["omega"][item]

    tol = 1e-6
    top = z + dz

    recs = state.loads.setdefault(k, [])

    borne_by_new = 0.0

    for rec in recs:
        _li, lx, ly, lz, ldx, ldy, ldz, _borne, ocx, ocy = rec

        # The new box sits over this one: charge it.
        if (z >= lz + ldz - tol
                and lx - tol <= cx <= lx + ldx + tol
                and ly - tol <= cy <= ly + ldy + tol):
            rec[7] += w

        # This box sits over the new one: the new box absorbs its weight.
        elif (lz >= top - tol
                and x - tol <= ocx <= x + dx + tol
                and y - tol <= ocy <= y + dy + tol):
            borne_by_new += inst["omega"][_li]

    # Fields 8 and 9 carry the CoM so that later placements landing beneath
    # this box can test containment without re-deriving its orientation.
    recs.append([item, x, y, z, dx, dy, dz, borne_by_new, cx, cy])


# =========================================================================
# Rollout and expansion
# =========================================================================
def fch_v2(state, inst, rule=None, strict=True):
    """Greedy completion under V2; the shape is V1's, the tests are V2's."""

    s = state.copy()

    volumes = [it["volume"] for it in inst["items"]]

    for k in bin_order(inst):

        while True:

            s.spaces[k].sort(key=lambda sp: (sp[1], sp[2], sp[0]))

            best = None

            for space in s.spaces[k]:
                for item in sorted(s.remaining, key=lambda i: -volumes[i]):
                    for rot, dims in enumerate(inst["feasible"][item][k]):
                        for refl in (0, 1):
                            pos = can_place_v2(s, k, space, dims, inst, item,
                                               rot, refl, rule=rule,
                                               strict=strict)
                            if pos is not None:
                                best = (space, item, dims, pos, rot, refl)
                                break
                        if best:
                            break
                    if best:
                        break
                if best:
                    break

            if best is None:
                break

            space, item, dims, pos, rot, refl = best
            apply_action_v2(s, inst, k, space, item, dims, pos, rot, refl)

    return objective_scale(inst, s.packed_by_bin), s


def expand_v2(state, inst, max_actions, rng, rule=None, strict=True):
    """Up to `max_actions` feasible V2 actions, as V1 but with the load test."""

    actions = []
    volumes = [it["volume"] for it in inst["items"]]

    per_space = max(1, max_actions // 4)

    for k in bin_order(inst):

        spaces = sorted(state.spaces[k], key=lambda sp: (sp[1], sp[2], sp[0]))

        for space in spaces:

            feasible_items = []

            for item in state.remaining:
                found = False
                for rot, dims in enumerate(inst["feasible"][item][k]):
                    for refl in (0, 1):
                        if can_place_v2(state, k, space, dims, inst, item,
                                        rot, refl, rule=rule, strict=strict):
                            feasible_items.append(item)
                            found = True
                            break
                    if found:
                        break

            if not feasible_items:
                continue

            pool = list(feasible_items)
            chosen = []

            while pool and len(chosen) < per_space:
                total = sum(volumes[i] for i in pool)
                if total <= 0:
                    chosen.extend(pool[:per_space - len(chosen)])
                    break
                r = rng.uniform(0, total)
                upto = 0.0
                pick = pool[-1]
                for i in pool:
                    upto += volumes[i]
                    if r <= upto:
                        pick = i
                        break
                chosen.append(pick)
                pool.remove(pick)

            for item in chosen:
                for rot, dims in enumerate(inst["feasible"][item][k]):
                    for refl in (0, 1):
                        pos = can_place_v2(state, k, space, dims, inst, item,
                                           rot, refl, rule=rule, strict=strict)
                        if pos is not None:
                            actions.append((k, space, item, dims, pos, rot,
                                            refl))
                            if len(actions) >= max_actions:
                                return actions

            if actions:
                return actions

    return actions


# =========================================================================
# Beam search
# =========================================================================
def beam_search_v2(inst, width=BEAM_WIDTH, max_actions=MAX_ACTIONS,
                   time_limit=TIME_LIMIT, seed=None, rule=None, strict=True,
                   verbose=False):
    """Beam search over partial packings under V2."""

    rng = random.Random(seed)
    t0 = time.time()

    root = initial_state_v2(inst)

    best_score, best_state = fch_v2(root, inst, rule=rule, strict=strict)

    beam = [root]
    rollouts = 1
    levels = 0

    while beam and time.time() - t0 < time_limit:

        candidates = []

        for node in beam:

            if time.time() - t0 >= time_limit:
                break

            for action in expand_v2(node, inst, max_actions, rng, rule=rule,
                                    strict=strict):

                k, space, item, dims, pos, rot, refl = action

                child = node.copy()
                apply_action_v2(child, inst, k, space, item, dims, pos, rot,
                                refl)

                score, rolled = fch_v2(child, inst, rule=rule, strict=strict)
                rollouts += 1

                if score > best_score:
                    best_score, best_state = score, rolled

                candidates.append((score, child))

                if time.time() - t0 >= time_limit:
                    break

        if not candidates:
            break

        candidates.sort(key=lambda t: -t[0])
        beam = [c for _s, c in candidates[:width]]
        levels += 1

        if verbose:
            print(f"  level {levels}: {len(candidates)} candidates, "
                  f"best {best_score:.4f}, {rollouts} rollouts, "
                  f"{time.time() - t0:.1f}s")

    return {
        "objective": best_score,
        "wasted": wasted_space_objective(inst, best_state.packed_by_bin),
        "placements": best_state.placements,
        "n_packed": len(best_state.placements),
        "rollouts": rollouts,
        "levels": levels,
        "runtime": time.time() - t0,
        "strict": strict,
    }


def beam_search_v2_auto(inst, width=BEAM_WIDTH, max_actions=MAX_ACTIONS,
                        time_limit=TIME_LIMIT, seed=None, rule=None,
                        verbose=False):
    """
    Strict CoM first, falling back to the relaxed reading if it packs
    nothing. See MHKP-V1-BS.beam_search_v1_auto: the non-monotonicity is
    the CoM constraint's, inherited here. Load bearing needs no such
    fallback, being monotone.
    """

    r = beam_search_v2(inst, width=width, max_actions=max_actions,
                       time_limit=time_limit, seed=seed, rule=rule,
                       strict=True, verbose=verbose)

    if r["n_packed"] > 0:
        return r

    relaxed = beam_search_v2(inst, width=width, max_actions=max_actions,
                             time_limit=time_limit, seed=seed, rule=rule,
                             strict=False, verbose=verbose)

    if relaxed["n_packed"] > 0 and not verify_v2(inst, relaxed["placements"],
                                                 rule=rule):
        relaxed["fell_back"] = True
        return relaxed

    return r


# =========================================================================
# Verification
# =========================================================================
def verify_v2(inst, placements, rule=None, tol=1e-6):
    """
    Everything V1 checks, plus load bearing recomputed from the placements.

    Independent of the search's own borne-load bookkeeping, so an error in
    the incremental accounting cannot verify itself.
    """

    violations = list(_bs1.verify_v1(inst, placements, rule=rule, tol=tol))

    by_bin = {}
    for pl in placements:
        by_bin.setdefault(pl["bin"], []).append(pl)

    for k, placed in by_bin.items():

        for lower in placed:

            top = lower["z"] + lower["dz"]
            load = 0.0

            for upper in placed:

                if upper is lower:
                    continue

                if upper["z"] < top - tol:
                    continue

                dims = (upper["dx"], upper["dy"], upper["dz"])
                cx, cy, _cz = _com_of(inst, upper["item"], dims,
                                      (upper["x"], upper["y"], upper["z"]),
                                      upper["rotation"],
                                      upper.get("reflected", 0))

                if not (lower["x"] - tol <= cx <= lower["x"] + lower["dx"] + tol):
                    continue
                if not (lower["y"] - tol <= cy <= lower["y"] + lower["dy"] + tol):
                    continue

                load += inst["omega"][upper["item"]]

            cap = inst["Lambda"][lower["item"]]

            if load > cap + tol:
                violations.append(
                    f"bin {k}, item {lower['item']}: bears {load:.1f} "
                    f"> capacity {cap:.1f}")

    return violations


# =========================================================================
# main
# =========================================================================
def main():
    import mhkp_instances as mi

    parser = argparse.ArgumentParser(
        description="Beam search for the MHKP under submodel V2 "
                    "(stability + centre of mass + load bearing).")

    parser.add_argument("--group", default="t8_n18")
    parser.add_argument("--width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--factor", type=float, default=LOAD_BEARING_FACTOR,
                        help=f"load bearing as a multiple of the maximal "
                             f"weight (default {LOAD_BEARING_FACTOR})")
    parser.add_argument("--stability-rule", choices=("corners", "peraxis"),
                        default="peraxis")
    parser.add_argument("--relaxed-intermediate", action="store_true")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="also run the V1 and V0 beam searches")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    _sa.STABILITY_RULE = args.stability_rule

    strict = not args.relaxed_intermediate

    root = Path(__file__).resolve().parent
    files = sorted((root / "MHKP" / args.group).glob("*.txt"))

    if not files:
        print(f"no instances in MHKP/{args.group}")
        return

    header = (f"{'instance':<26}{'V2 obj':>9}{'packed':>8}{'rollouts':>10}"
              f"{'time':>7}{'viol':>6}")

    if args.compare:
        header += f"{'V1 obj':>9}{'V0 obj':>9}"

    print("=" * len(header))
    print(f"Beam search, submodel V2 | w={args.width}, m={args.actions}, "
          f"{args.time_limit}s")
    print(f"alpha={args.alpha}, Lambda={args.factor}x, seed={args.seed} | "
          f"intermediate CoM: {'strict' if strict else 'relaxed'}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    v2s, v1s, v0s, packs = [], [], [], []
    total_viol = 0

    for f in files:

        inst = load_instance(str(f))
        bins_raw, items_raw = mi.read_instance(str(f))
        attach_v2_data(inst, bins_raw, items_raw, seed=args.seed,
                       alpha=args.alpha, factor=args.factor)

        if strict and not args.strict_only:
            r = beam_search_v2_auto(inst, width=args.width,
                                    max_actions=args.actions,
                                    time_limit=args.time_limit,
                                    seed=args.seed, rule=args.stability_rule,
                                    verbose=args.verbose)
        else:
            r = beam_search_v2(inst, width=args.width,
                               max_actions=args.actions,
                               time_limit=args.time_limit, seed=args.seed,
                               rule=args.stability_rule, strict=strict,
                               verbose=args.verbose)

        viol = verify_v2(inst, r["placements"], rule=args.stability_rule)
        total_viol += len(viol)

        v2s.append(r["wasted"])
        packs.append(r["n_packed"])

        mark = "*" if r.get("fell_back") else ""

        row = (f"{f.stem + mark:<26}{r['wasted']:>9.4f}{r['n_packed']:>8}"
               f"{r['rollouts']:>10}{r['runtime']:>7.2f}{len(viol):>6}")

        if args.compare:
            r1 = _bs1.beam_search_v1_auto(
                inst, width=args.width, max_actions=args.actions,
                time_limit=args.time_limit, seed=args.seed,
                rule=args.stability_rule)
            r0 = _bs0.beam_search(inst, width=args.width,
                                  max_actions=args.actions,
                                  time_limit=args.time_limit, seed=args.seed,
                                  rule=args.stability_rule)
            v1s.append(r1["wasted"])
            v0s.append(r0["wasted"])
            row += f"{r1['wasted']:>9.4f}{r0['wasted']:>9.4f}"

        print(row)

        for x in viol[:3]:
            print(f"    {x}")

    print("-" * len(header))

    summary = (f"{'mean':<26}{statistics.fmean(v2s):>9.4f}"
               f"{statistics.fmean(packs):>8.1f}{'':>10}{'':>7}"
               f"{total_viol:>6}")

    if args.compare:
        summary += (f"{statistics.fmean(v1s):>9.4f}"
                    f"{statistics.fmean(v0s):>9.4f}")

    print(summary)


if __name__ == "__main__":
    main()
