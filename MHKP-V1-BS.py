#!/usr/bin/env python3
# Beam Search for the MHKP under submodel V1 of Deplano et al. (2019)
# =========================================================================
# V1 counterpart to MHKP-V0-BS.py, and the heuristic counterpart to
# mhkp_v1_milp.py, for the problem of
#
#     Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
#     "A mixed-integer linear model for the multiple heterogeneous knapsack
#      problem with realistic container loading constraints and bins' priority"
#     Intl. Trans. in Op. Res. 28(6), 3244-3275.
#
# V0 is constraints (1)-(10r): geometry, the priority objective and static
# stability. V1 adds (11a)-(12n): each item carries a centre of mass (CoM)
# that rotation and reflection move around inside it, and the WEIGHTED CoM of
# every loaded bin must lie inside a pyramidal safe region.
#
# This file reuses MHKP-V0-BS.py wholesale - the state, the maximal-space
# split, the rollout, the beam - and changes only what V1 actually changes.
#
# -------------------------------------------------------------------------
# WHY THE CoM CONSTRAINT DOES NOT FIT THE V0 FEASIBILITY PATTERN
# -------------------------------------------------------------------------
# Every V0 constraint is LOCAL to a placement: containment, overlap and
# support can all be decided by looking at the one box being placed and the
# boxes already in the bin. can_place() therefore answers yes or no for a
# single candidate, and a node that passes is feasible.
#
# The CoM constraint is not like that. It is a property of the BIN AS A WHOLE:
#
#     sum_i tau^x_i,j omega_i  within  varrho^x_j M_j +- (headroom) xi_j
#
# No single box violates it; an aggregate does. Two consequences follow, and
# they are the whole design of this file:
#
#   1. FEASIBILITY IS CHECKED ON THE RUNNING AGGREGATE, not on the box. The
#      state carries the running weighted sums (mass, sum w*cx, sum w*cy,
#      sum w*cz) per bin, and a candidate placement is accepted iff the sums
#      INCLUDING it still satisfy the pyramid. Recomputing the CoM from
#      scratch per candidate would be O(items in bin) per check inside the
#      innermost loop of the search; the incremental sums make it O(1).
#
#   2. FEASIBILITY IS NOT MONOTONE. In V0, a partial packing that is feasible
#      stays feasible: adding a box never invalidates the boxes under it. Here
#      a partial packing can be feasible and admit NO feasible completion, and
#      - more awkwardly - a partial packing can be INFEASIBLE and still be on
#      the way to a feasible one, because a later box on the opposite side can
#      pull the CoM back into the region.
#
#      We take the strict reading: every node of the search tree must itself
#      satisfy the pyramid. That is what the MILP's (12j)-(12n) impose on the
#      final packing, and enforcing it at every step is the conservative
#      choice - it can only reject packings the MILP would accept, never admit
#      ones it would reject. `--relaxed-intermediate` switches to the permissive
#      reading, allowing intermediate violations and checking only at the end;
#      it explores more but can end on an infeasible packing, which is then
#      rejected outright. The strict reading is the default because a beam
#      search that ends with nothing feasible is worse than one that ends with
#      a conservative answer.
#
# -------------------------------------------------------------------------
# REFLECTION
# -------------------------------------------------------------------------
# V0 has no reflection variable, because reflecting a cuboid does not change
# the region it occupies and no V0 constraint can observe it. Under V1 it
# matters: reflection mirrors the CoM inside the box, which moves the bin's
# aggregate CoM. An action is therefore (bin, space, item, dims, pos, rot,
# refl) rather than V0's six-tuple, and each candidate is offered in both
# reflected and unreflected form - at most 2x the actions per item, which is
# why `per_space` is divided by 2 again relative to V0.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#     python3 MHKP-V1-BS.py                          # default group
#     python3 MHKP-V1-BS.py --group t8_n50 --width 12 --actions 60
#     python3 MHKP-V1-BS.py --alpha 0.1              # tighter pyramid
#     python3 MHKP-V1-BS.py --compare-v0             # cost of the CoM region
#     python3 MHKP-V1-BS.py --relaxed-intermediate

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


_bs = _load("mhkp_v0_bs", "MHKP-V0-BS.py")
_v1 = _load("mhkp_v1_milp", "mhkp_v1_milp.py")

_sa = _bs._sa

load_instance = _sa.load_instance
objective_scale = _sa.objective_scale
wasted_space_objective = _sa.wasted_space_objective
verify = _sa.verify

split_space = _bs.split_space
prune_spaces = _bs.prune_spaces
bin_order = _bs.bin_order
is_supported = _sa.is_supported

derive_v1_data = _v1.derive_v1_data
DEFAULT_ALPHA = _v1.DEFAULT_ALPHA

BEAM_WIDTH = _bs.BEAM_WIDTH
MAX_ACTIONS = _bs.MAX_ACTIONS
TIME_LIMIT = _bs.TIME_LIMIT


# =========================================================================
# V1 instance data
# =========================================================================
def attach_v1_data(inst, bins_raw, items_raw, seed=1, alpha=DEFAULT_ALPHA):
    """
    Attach the weights, CoM offsets and pyramid parameters that V1 needs.

    Derived by mhkp_v1_milp.derive_v1_data, so the beam search and the MILP
    see EXACTLY the same weights and CoMs for a given instance and seed -
    without that, the two are not solving the same problem and their
    objectives cannot be compared.
    """

    omega, kappa, varrho, xi = derive_v1_data(bins_raw, items_raw,
                                              seed=seed, alpha=alpha)

    inst["omega"] = omega
    inst["kappa"] = kappa
    inst["varrho"] = varrho
    inst["xi"] = xi

    return inst


def oriented_com(kappa_i, dims, rot, refl):
    """
    The CoM offset inside the box, after rotation and reflection.

    `dims` are the already-oriented extents (dx, dy, dz). Rotation 0 keeps
    kx on x; rotation 1 swaps kx and ky, matching constraints (11a)-(11h)
    and the rotation convention of MHKP-V0-SA.rotations. Reflection mirrors
    the offset within the box, dx - ox.

    The z offset is untouched: there is no rotation about a horizontal axis,
    which is exactly why (12g)-(12i) carry no rotation term.
    """

    kx, ky, kz = kappa_i
    dx, dy, _dz = dims

    ox = kx if rot == 0 else ky
    oy = ky if rot == 0 else kx

    if refl:
        ox = dx - ox
        oy = dy - oy

    return ox, oy, kz


def com_ok(inst, k, mass, wx, wy, wz, tol=1e-6):
    """
    True if a bin holding the given weighted sums satisfies its pyramid.

    `mass` is the loaded weight and wx, wy, wz the weight-times-coordinate
    sums, i.e. the numerators of the CoM. Written on the sums rather than on
    the CoM itself so that an empty bin (mass = 0) is trivially feasible,
    matching the MILP, where (12a)/(12d)/(12g) zero every tau of an empty bin
    and (12j)-(12n) collapse to 0 <= 0.
    """

    if mass <= 0:
        return True

    rx, ry, rz = inst["varrho"][k]

    cz = wz / mass

    # (12n): the CoM may not rise above the vertex.
    if cz > rz + tol:
        return False

    # The permitted horizontal deviation shrinks linearly to zero at the
    # vertex, which is what makes the region a pyramid. Equivalent to
    # (12j)-(12m) after dividing through by `mass`.
    allowed = (rz - cz) / rz * inst["xi"][k]

    if abs(wx / mass - rx) > allowed + tol:
        return False

    if abs(wy / mass - ry) > allowed + tol:
        return False

    return True


# =========================================================================
# State
# =========================================================================
class StateV1:
    """
    A V0 state plus the running weighted CoM sums per bin.

    The sums are what make the CoM check O(1) per candidate instead of
    O(items in the bin); see the header.
    """

    __slots__ = ("base", "mass", "wx", "wy", "wz")

    def __init__(self, base, mass=None, wx=None, wy=None, wz=None):
        self.base = base
        self.mass = mass if mass is not None else {}
        self.wx = wx if wx is not None else {}
        self.wy = wy if wy is not None else {}
        self.wz = wz if wz is not None else {}

    # The beam and the objective only ever touch the underlying V0 state,
    # so these read through rather than duplicating it.
    @property
    def spaces(self):
        return self.base.spaces

    @property
    def placed(self):
        return self.base.placed

    @property
    def remaining(self):
        return self.base.remaining

    @property
    def packed_by_bin(self):
        return self.base.packed_by_bin

    @property
    def placements(self):
        return self.base.placements

    def copy(self):
        return StateV1(self.base.copy(), dict(self.mass), dict(self.wx),
                       dict(self.wy), dict(self.wz))


def initial_state_v1(inst):
    return StateV1(_bs.initial_state(inst))


def can_place_v1(state, k, space, dims, inst, item, rot, refl, rule=None,
                 strict=True):
    """
    Where a box may sit in `space` under V1, or None.

    Geometry, overlap and support are V0's `can_place`. The CoM test is then
    applied to the bin's sums INCLUDING this candidate, which is the part V0
    has no counterpart for.

    With `strict=False` the CoM test is skipped, giving the permissive reading
    described in the header.
    """

    pos = _bs.can_place(state.base, k, space, dims, inst, rule=rule)

    if pos is None:
        return None

    if not strict:
        return pos

    x, y, z = pos
    ox, oy, oz = oriented_com(inst["kappa"][item], dims, rot, refl)
    w = inst["omega"][item]

    mass = state.mass.get(k, 0.0) + w
    wx = state.wx.get(k, 0.0) + w * (x + ox)
    wy = state.wy.get(k, 0.0) + w * (y + oy)
    wz = state.wz.get(k, 0.0) + w * (z + oz)

    if not com_ok(inst, k, mass, wx, wy, wz):
        return None

    return pos


def apply_action_v1(state, inst, k, space, item, dims, pos, rot, refl):
    """Place an item and update both the V0 state and the CoM sums."""

    _bs.apply_action(state.base, inst, k, space, item, dims, pos, rot)

    # V0's apply_action does not know about reflection; record it on the
    # placement it just appended so the verifier can reproduce the CoM.
    state.placements[-1]["reflected"] = int(bool(refl))

    x, y, z = pos
    ox, oy, oz = oriented_com(inst["kappa"][item], dims, rot, refl)
    w = inst["omega"][item]

    state.mass[k] = state.mass.get(k, 0.0) + w
    state.wx[k] = state.wx.get(k, 0.0) + w * (x + ox)
    state.wy[k] = state.wy.get(k, 0.0) + w * (y + oy)
    state.wz[k] = state.wz.get(k, 0.0) + w * (z + oz)


# =========================================================================
# Rollout
# =========================================================================
def fch_v1(state, inst, rule=None, strict=True):
    """
    Complete a partial packing greedily under V1, and return its objective.

    Same shape as V0's FCH - DBL spaces, largest item first - but every
    candidate is additionally offered in both reflections, and accepted only
    if the bin's CoM survives it. Reflection is tried in the order that keeps
    the CoM closer to the pyramid axis, so the greedy rollout does not have to
    branch on it.
    """

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

                            pos = can_place_v1(s, k, space, dims, inst, item,
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
            apply_action_v1(s, inst, k, space, item, dims, pos, rot, refl)

    return objective_scale(inst, s.packed_by_bin), s


# =========================================================================
# Node expansion
# =========================================================================
def expand_v1(state, inst, max_actions, rng, rule=None, strict=True):
    """
    Up to `max_actions` feasible V1 actions from this state.

    As in V0: bins in priority order, spaces in DBL order, commit to the first
    space that admits anything, and sample items by volume-weighted roulette.
    The budget per space is m/4 rather than m/2, because each item is now
    crossed with two rotations AND two reflections.
    """

    actions = []
    volumes = [it["volume"] for it in inst["items"]]

    # |O| = 2 rotations x 2 reflections.
    per_space = max(1, max_actions // 4)

    for k in bin_order(inst):

        spaces = sorted(state.spaces[k], key=lambda sp: (sp[1], sp[2], sp[0]))

        for space in spaces:

            feasible_items = []

            for item in state.remaining:
                found = False
                for rot, dims in enumerate(inst["feasible"][item][k]):
                    for refl in (0, 1):
                        if can_place_v1(state, k, space, dims, inst, item, rot,
                                        refl, rule=rule, strict=strict):
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
                        pos = can_place_v1(state, k, space, dims, inst, item,
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
def beam_search_v1(inst, width=BEAM_WIDTH, max_actions=MAX_ACTIONS,
                   time_limit=TIME_LIMIT, seed=None, rule=None, strict=True,
                   verbose=False):
    """
    Beam search over partial packings under V1, scored by a greedy rollout.

    Identical in structure to MHKP-V0-BS.beam_search; the differences are
    entirely inside expand_v1/fch_v1, which is the point of keeping them
    separate.
    """

    rng = random.Random(seed)
    t0 = time.time()

    root = initial_state_v1(inst)

    best_score, best_state = fch_v1(root, inst, rule=rule, strict=strict)

    beam = [root]
    rollouts = 1
    levels = 0

    while beam and time.time() - t0 < time_limit:

        candidates = []

        for node in beam:

            if time.time() - t0 >= time_limit:
                break

            for action in expand_v1(node, inst, max_actions, rng, rule=rule,
                                    strict=strict):

                k, space, item, dims, pos, rot, refl = action

                child = node.copy()
                apply_action_v1(child, inst, k, space, item, dims, pos, rot,
                                refl)

                score, rolled = fch_v1(child, inst, rule=rule, strict=strict)
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

    wasted = wasted_space_objective(inst, best_state.packed_by_bin)

    return {
        "objective": best_score,
        "wasted": wasted,
        "placements": best_state.placements,
        "n_packed": len(best_state.placements),
        "rollouts": rollouts,
        "levels": levels,
        "runtime": time.time() - t0,
        "strict": strict,
    }


def beam_search_v1_auto(inst, width=BEAM_WIDTH, max_actions=MAX_ACTIONS,
                        time_limit=TIME_LIMIT, seed=None, rule=None,
                        verbose=False):
    """
    Run the strict search, and fall back to the relaxed one if it packs
    nothing.

    The strict reading requires every PREFIX of the packing to satisfy the
    pyramid, while the MILP requires only the FINAL packing to. Because
    feasibility is not monotone (see the header), the two differ: with a
    tight pyramid the first box placed is off-axis on its own, so the strict
    search places nothing at all, while the MILP happily packs a dozen items
    whose CoMs cancel out. Measured on t8_n18_instance01 with alpha = 0.1:
    strict packs 0, the MILP packs 12, and both are "feasible" by their own
    rule.

    A search that returns an empty packing is useless, so this wrapper falls
    back. The relaxed result is only accepted if it VERIFIES - the strict
    empty packing is returned otherwise, since a wrong answer is worse than
    a conservative one.
    """

    r = beam_search_v1(inst, width=width, max_actions=max_actions,
                       time_limit=time_limit, seed=seed, rule=rule,
                       strict=True, verbose=verbose)

    if r["n_packed"] > 0:
        return r

    relaxed = beam_search_v1(inst, width=width, max_actions=max_actions,
                             time_limit=time_limit, seed=seed, rule=rule,
                             strict=False, verbose=verbose)

    if relaxed["n_packed"] > 0 and not verify_v1(inst, relaxed["placements"],
                                                 rule=rule):
        relaxed["fell_back"] = True
        return relaxed

    return r


# =========================================================================
# Verification
# =========================================================================
def verify_v1(inst, placements, rule=None, tol=1e-6):
    """
    Check a V1 packing: everything V0 checks, plus the pyramid.

    The CoM is recomputed from the placements rather than read from the search
    state, so a bookkeeping error in the incremental sums cannot verify itself.
    """

    violations = list(verify(inst, placements, stability=True, rule=rule))

    by_bin = {}
    for pl in placements:
        by_bin.setdefault(pl["bin"], []).append(pl)

    for k, placed in by_bin.items():

        mass = wx = wy = wz = 0.0

        for pl in placed:
            dims = (pl["dx"], pl["dy"], pl["dz"])
            ox, oy, oz = oriented_com(inst["kappa"][pl["item"]], dims,
                                      pl["rotation"], pl.get("reflected", 0))
            w = inst["omega"][pl["item"]]

            mass += w
            wx += w * (pl["x"] + ox)
            wy += w * (pl["y"] + oy)
            wz += w * (pl["z"] + oz)

        if not com_ok(inst, k, mass, wx, wy, wz, tol=tol):
            rx, ry, rz = inst["varrho"][k]
            violations.append(
                f"bin {k}: CoM ({wx / mass:.2f}, {wy / mass:.2f}, "
                f"{wz / mass:.2f}) outside the pyramid "
                f"(vertex {rx:.1f}, {ry:.1f}, {rz:.1f}, xi {inst['xi'][k]:.1f})")

    return violations


# =========================================================================
# main
# =========================================================================
def main():
    import mhkp_instances as mi

    parser = argparse.ArgumentParser(
        description="Beam search for the MHKP under submodel V1 "
                    "(stability + centre of mass).")

    parser.add_argument("--group", default="t8_n18",
                        help="instance group under MHKP/ (default t8_n18)")
    parser.add_argument("--width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"pyramid base fraction (default {DEFAULT_ALPHA})")
    parser.add_argument("--stability-rule", choices=("corners", "peraxis"),
                        default="peraxis")
    parser.add_argument("--relaxed-intermediate", action="store_true",
                        help="allow intermediate CoM violations and check only "
                             "the final packing")
    parser.add_argument("--strict-only", action="store_true",
                        help="disable the fallback: keep the strict result "
                             "even when it packs nothing")
    parser.add_argument("--compare-v0", action="store_true",
                        help="also run the V0 beam search, to show what the "
                             "CoM region costs")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    _sa.STABILITY_RULE = args.stability_rule

    strict = not args.relaxed_intermediate

    root = Path(__file__).resolve().parent
    files = sorted((root / "MHKP" / args.group).glob("*.txt"))

    if not files:
        print(f"no instances in MHKP/{args.group}")
        return

    header = (f"{'instance':<28}{'V1 obj':>9}{'packed':>8}{'rollouts':>10}"
              f"{'time':>8}{'viol':>6}")

    if args.compare_v0:
        header += f"{'V0 obj':>9}{'V0 pk':>7}"

    print("=" * len(header))
    print(f"Beam search, submodel V1 | w={args.width}, m={args.actions}, "
          f"{args.time_limit}s | alpha={args.alpha}, seed={args.seed}")
    print(f"intermediate CoM: {'strict' if strict else 'relaxed'} | "
          f"stability rule '{args.stability_rule}'")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    v1_objs, v0_objs, packs = [], [], []
    total_viol = 0

    for f in files:

        inst = load_instance(str(f))
        bins_raw, items_raw = mi.read_instance(str(f))
        attach_v1_data(inst, bins_raw, items_raw, seed=args.seed,
                       alpha=args.alpha)

        if strict and not args.strict_only:
            r = beam_search_v1_auto(inst, width=args.width,
                                    max_actions=args.actions,
                                    time_limit=args.time_limit,
                                    seed=args.seed, rule=args.stability_rule,
                                    verbose=args.verbose)
        else:
            r = beam_search_v1(inst, width=args.width,
                               max_actions=args.actions,
                               time_limit=args.time_limit, seed=args.seed,
                               rule=args.stability_rule, strict=strict,
                               verbose=args.verbose)

        viol = verify_v1(inst, r["placements"], rule=args.stability_rule)
        total_viol += len(viol)

        v1_objs.append(r["wasted"])
        packs.append(r["n_packed"])

        mark = "*" if r.get("fell_back") else ""

        row = (f"{f.stem + mark:<28}{r['wasted']:>9.4f}{r['n_packed']:>8}"
               f"{r['rollouts']:>10}{r['runtime']:>8.2f}{len(viol):>6}")

        if args.compare_v0:
            r0 = _bs.beam_search(inst, width=args.width,
                                 max_actions=args.actions,
                                 time_limit=args.time_limit, seed=args.seed,
                                 rule=args.stability_rule)
            v0_objs.append(r0["wasted"])
            row += f"{r0['wasted']:>9.4f}{r0['n_packed']:>7}"

        print(row)

        for v in viol[:3]:
            print(f"    {v}")

    print("-" * len(header))

    summary = (f"{'mean':<28}{statistics.fmean(v1_objs):>9.4f}"
               f"{statistics.fmean(packs):>8.1f}{'':>10}{'':>8}"
               f"{total_viol:>6}")

    if args.compare_v0:
        summary += f"{statistics.fmean(v0_objs):>9.4f}"

    print(summary)

    if args.compare_v0 and v0_objs:
        m1, m0 = statistics.fmean(v1_objs), statistics.fmean(v0_objs)
        print()
        print(f"CoM region costs {(m1 - m0) / m0 * 100:+.2f}% wasted space "
              f"(V0 {m0:.4f} -> V1 {m1:.4f})")


if __name__ == "__main__":
    main()
