#!/usr/bin/env python3
# Beam Search for the MHKP under submodel V0 of Deplano et al. (2019)
# =========================================================================
# Third solver for submodel V0, beside MHKP-V0-SA.py (simulated annealing) and
# MHKP-V0-HC.py (hill climbing), for the problem of
#
#     Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
#     "A mixed-integer linear model for the multiple heterogeneous knapsack
#      problem with realistic container loading constraints and bins' priority"
#     Intl. Trans. in Op. Res. 28(6), 3244-3275.
#
# The beam search framework is the BSA of
#
#     Xu, X., Wu, B., Ma, Z., Yu, Y. (2026)
#     "The three-dimensional bin packing problem with variable box size"
#     Transportation Research Part E 214, 105038, Section 4,
#
# transplanted onto the MHKP: multi-path expansion with width w, at most m
# actions per node, and a fast constructive rollout scoring every candidate.
#
# -------------------------------------------------------------------------
# WHY BEAM SEARCH IS THE INTERESTING THIRD OPTION
# -------------------------------------------------------------------------
# Measured on this decoder, 97% of an SA or HC iteration is the decode. Both of
# those methods perturb a SEQUENCE and then re-decode the whole packing to find
# out what the perturbation did, so their unit of work is one full packing and
# their iteration count is capped by decode cost. That cap is what leaves them
# a quarter of the bin empty at n=90 on instances that admit a perfect pack.
#
# Beam search does not have that shape. Its unit of work is one PLACEMENT: a
# node is a partial packing, and expanding it commits one more item to one more
# space. The expensive rollout is still a full construction, but it scores a
# whole subtree rather than a single neighbour, and the partial packing is
# built incrementally and never recomputed. So it spends its budget differently
# rather than merely spending it faster, which is the only way to escape a
# cost structure rather than optimise within it.
#
# Whether that pays here is an empirical question, and this file exists to
# answer it rather than to assume it.
#
# -------------------------------------------------------------------------
# THE FOUR COMPONENTS (Xu et al., Section 4.1)
# -------------------------------------------------------------------------
# STATE          a partial packing: the free spaces per bin, the items already
#                placed, and which items remain.
#
# TRANSITION     placing item i into free space s with rotation o removes the
#                item from the pool and re-partitions the occupied space.
#
# EXPANSION      spaces are traversed in DBL order; for the first space that
#                admits anything, the admissible items are sampled by a
#                volume-weighted roulette and crossed with the rotations, to at
#                most m actions.
#
# EVALUATION     each candidate is completed by a fast greedy rollout (FCH) and
#                scored by the resulting objective. The best w nodes survive.
#
# -------------------------------------------------------------------------
# WHAT CHANGES RELATIVE TO THE BSA OF XU ET AL.
# -------------------------------------------------------------------------
#   * OBJECTIVE. The BSA maximises the volume fraction of a single bin. V0
#     minimises the priority-weighted wasted space of formula (1) over several
#     bins. Nodes are scored by the normalised complement (objective_scale in
#     MHKP-V0-SA.py), so larger is better and the beam keeps the top w.
#
#   * MULTIPLE BINS. The BSA has one bin. Here each state carries free spaces
#     for every bin, and an action names its bin. Bins are offered in priority
#     order - smallest first under p_j = 1/V_j - so the search prefers to fill
#     cheap bins, which is what formula (1) rewards.
#
#   * TWO ROTATIONS, not six times three configurations: V0 allows rotation
#     about the vertical axis only (Deplano Table 2), and its items are rigid,
#     so there are no size variants. The m/6 cap on the roulette in the BSA's
#     Algorithm 1 is therefore m/2 here, one slot per rotation.
#
#   * STABILITY. Every placement must satisfy constraints (10a)-(10r). This is
#     the constraint the BSA has no counterpart for, and it is checked at
#     placement time so that no infeasible node ever enters the beam.
#
#   * MAXIMAL SPACES. The BSA uses the maximal-space partition of Parreno et
#     al. (2008), where placing an item splits the space it occupied into three
#     OVERLAPPING regions, each running to the far corner of the bin. That is
#     kept, because it is what makes the expansion enumerate placements rather
#     than corner points - see split_space below.
#
# -------------------------------------------------------------------------
# USAGE
# -------------------------------------------------------------------------
#     python3 MHKP-V0-BS.py                        # default group
#     python3 MHKP-V0-BS.py --group t8_n50 --width 12 --actions 60
#     python3 MHKP-V0-BS.py --compare              # against SA and HC
#     python3 MHKP-V0-BS.py --stability-rule peraxis

import argparse
import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path


def _load_sa_module():
    """Load MHKP-V0-SA.py by path; its name is not importable."""

    path = Path(__file__).resolve().parent / "MHKP-V0-SA.py"

    spec = importlib.util.spec_from_file_location("mhkp_v0_sa", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mhkp_v0_sa"] = module
    spec.loader.exec_module(module)

    return module


_sa = _load_sa_module()

load_instance = _sa.load_instance
build_instance = _sa.build_instance
verify = _sa.verify
rotations = _sa.rotations
objective_scale = _sa.objective_scale
wasted_space_objective = _sa.wasted_space_objective
is_supported = _sa.is_supported


# -------------------------
# Parameters
# -------------------------
# Xu et al. select w = 12 and m = 60 by grid search (their Section 5.2). Those
# are carried over as defaults: their instances are a different problem, but
# the two parameters trade the same things here - width against depth, and
# branching against rollout cost - so their values are a reasonable starting
# point rather than an arbitrary one.
BEAM_WIDTH = 12
MAX_ACTIONS = 60
TIME_LIMIT = 3.0


# =========================================================================
# Free spaces: the maximal-space partition (Parreno et al. 2008)
# =========================================================================
# A space is (x, y, z, dx, dy, dz) in the SA file's frame: x along the width W,
# y vertical along the height H, z along the length L.
def split_space(space, x, y, z, dx, dy, dz):
    """
    The maximal spaces left when a box occupies part of `space`.

    Unlike an extreme-point split, the three regions OVERLAP and each runs to
    the far face of the original space, which is what lets a later item use the
    full remaining extent along any one axis rather than only the sliver a
    corner point would expose.

    Returns the regions that survive with positive volume; a box that does not
    intersect `space` at all leaves it whole.
    """

    sx, sy, sz, sdx, sdy, sdz = space

    # No intersection: the space is untouched.
    if (x >= sx + sdx or x + dx <= sx
            or y >= sy + sdy or y + dy <= sy
            or z >= sz + sdz or z + dz <= sz):
        return [space]

    out = []

    # Along X: the slabs left of and right of the box.
    if x > sx:
        out.append((sx, sy, sz, x - sx, sdy, sdz))
    if x + dx < sx + sdx:
        out.append((x + dx, sy, sz, sx + sdx - (x + dx), sdy, sdz))

    # Along Y (vertical): below and above.
    if y > sy:
        out.append((sx, sy, sz, sdx, y - sy, sdz))
    if y + dy < sy + sdy:
        out.append((sx, y + dy, sz, sdx, sy + sdy - (y + dy), sdz))

    # Along Z: in front of and behind.
    if z > sz:
        out.append((sx, sy, sz, sdx, sdy, z - sz))
    if z + dz < sz + sdz:
        out.append((sx, sy, z + dz, sdx, sdy, sz + sdz - (z + dz)))

    return out


def prune_spaces(spaces):
    """
    Drop spaces wholly contained in another.

    The maximal-space split produces overlapping regions, so without this the
    list grows without bound and the expansion wastes its budget re-examining
    subsets of spaces it has already considered.
    """

    out = []

    for i, a in enumerate(spaces):
        ax, ay, az, adx, ady, adz = a
        if adx <= 0 or ady <= 0 or adz <= 0:
            continue
        contained = False
        for j, b in enumerate(spaces):
            if i == j:
                continue
            bx, by, bz, bdx, bdy, bdz = b
            if (bx <= ax and by <= ay and bz <= az
                    and ax + adx <= bx + bdx and ay + ady <= by + bdy
                    and az + adz <= bz + bdz):
                # Identical spaces contain each other; keep the first only.
                if a == b and j > i:
                    continue
                contained = True
                break
        if not contained:
            out.append(a)

    return out


# =========================================================================
# State
# =========================================================================
class State:
    """
    A partial packing.

    `spaces` maps a bin index to its list of maximal free spaces, `placed` maps
    a bin index to the boxes in it, and `remaining` is the set of unplaced
    items. States are copied on expansion, so the lists are shallow-copied per
    bin rather than deep-copied wholesale.
    """

    __slots__ = ("spaces", "placed", "remaining", "packed_by_bin",
                 "placements", "score")

    def __init__(self, spaces, placed, remaining, packed_by_bin, placements,
                 score=0.0):
        self.spaces = spaces
        self.placed = placed
        self.remaining = remaining
        self.packed_by_bin = packed_by_bin
        self.placements = placements
        self.score = score

    def copy(self):
        return State(
            {k: list(v) for k, v in self.spaces.items()},
            {k: list(v) for k, v in self.placed.items()},
            set(self.remaining),
            dict(self.packed_by_bin),
            list(self.placements),
            self.score,
        )


def initial_state(inst):
    """The empty packing: every bin is one maximal space, nothing placed."""

    spaces = {}
    placed = {}

    for k, b in enumerate(inst["bins"]):
        L, W, H = b["dims"]
        spaces[k] = [(0, 0, 0, W, H, L)]
        placed[k] = []

    return State(spaces, placed, set(range(inst["n"])), {}, [])


def bin_order(inst):
    """
    Bins in priority order, highest priority first.

    With p_j = 1/V_j that is smallest volume first, which is what formula (1)
    rewards and what the paper's own WFBF does.
    """

    return sorted(range(len(inst["bins"])),
                  key=lambda k: -inst["bins"][k]["priority"])


# =========================================================================
# Placement
# =========================================================================
def can_place(state, k, space, dims, inst, rule=None):
    """
    Where a box of `dims` can sit in `space`, or None.

    The box is pushed to the space's bottom-left-back corner, which is the DBL
    rule; only that one position per space is considered, because the maximal
    spaces already enumerate the distinct anchors.
    """

    sx, sy, sz, sdx, sdy, sdz = space
    dx, dy, dz = dims

    if dx > sdx or dy > sdy or dz > sdz:
        return None

    x, y, z = sx, sy, sz

    for (px, py, pz, pdx, pdy, pdz) in state.placed[k]:
        if (x < px + pdx and px < x + dx
                and y < py + pdy and py < y + dy
                and z < pz + pdz and pz < z + dz):
            return None

    if not is_supported(x, y, z, dx, dz, state.placed[k], rule=rule):
        return None

    return (x, y, z)


def apply_action(state, inst, k, space, item, dims, pos, rot):
    """Place an item: update the spaces, the placed list and the volume."""

    x, y, z = pos
    dx, dy, dz = dims

    new_spaces = []
    for s in state.spaces[k]:
        new_spaces.extend(split_space(s, x, y, z, dx, dy, dz))

    state.spaces[k] = prune_spaces(new_spaces)
    state.placed[k].append((x, y, z, dx, dy, dz))
    state.remaining.discard(item)
    state.packed_by_bin[k] = state.packed_by_bin.get(k, 0) + dx * dy * dz
    state.placements.append({
        "item": item, "bin": k, "x": x, "y": y, "z": z,
        "dx": dx, "dy": dy, "dz": dz, "rotation": rot,
    })


# =========================================================================
# Fast constructive heuristic (the rollout, Xu et al. Algorithm 2)
# =========================================================================
def fch(state, inst, rule=None):
    """
    Complete a partial packing greedily, and return its objective.

    Spaces are taken in DBL order and filled with the largest item that fits,
    exactly as the BSA's FCH does, until nothing more can be placed. The state
    is copied first, so the rollout never disturbs the node it scores.

    This is the expensive part: one full construction per candidate node. It
    does NOT get cheaper as the search deepens - measured on an n=50 instance,
    a rollout costs 1.31 ms at the root and 2.83 ms nine placements in, because
    the free-space list and the placed-box list both grow with depth while the
    pool of remaining items shrinks only slowly (50 -> 41). What beam search
    gains over the SA's decode is not a cheaper evaluation but a more
    informative one: each rollout scores a node that has one more item
    permanently committed, so the same work buys search depth rather than
    another sample of the same neighbourhood.
    """

    s = state.copy()

    volumes = [it["volume"] for it in inst["items"]]
    order = bin_order(inst)

    for k in order:

        while True:

            # DBL: lowest y, then z, then x.
            s.spaces[k].sort(key=lambda sp: (sp[1], sp[2], sp[0]))

            best = None

            for space in s.spaces[k]:

                candidates = sorted(s.remaining, key=lambda i: -volumes[i])

                for item in candidates:
                    for rot, dims in enumerate(inst["feasible"][item][k]):
                        pos = can_place(s, k, space, dims, inst, rule=rule)
                        if pos is not None:
                            best = (space, item, dims, pos, rot)
                            break
                    if best:
                        break
                if best:
                    break

            if best is None:
                break

            space, item, dims, pos, rot = best
            apply_action(s, inst, k, space, item, dims, pos, rot)

    return objective_scale(inst, s.packed_by_bin), s


# =========================================================================
# Node expansion (Xu et al. Algorithm 1)
# =========================================================================
def expand(state, inst, max_actions, rng, rule=None):
    """
    Up to `max_actions` feasible actions from this state.

    Bins are visited in priority order and, within a bin, spaces in DBL order.
    The first space that admits anything produces the actions, as in the BSA:
    committing to the deepest-lowest space keeps the search from scattering
    over equivalent placements elsewhere in the bin.

    Items are drawn by a volume-weighted roulette, which is the BSA's own
    device for sampling m/|O| items rather than enumerating all of them.
    """

    actions = []
    volumes = [it["volume"] for it in inst["items"]]
    per_space = max(1, max_actions // 2)      # two rotations, so m/2 items

    for k in bin_order(inst):

        spaces = sorted(state.spaces[k], key=lambda sp: (sp[1], sp[2], sp[0]))

        for space in spaces:

            feasible_items = []

            for item in state.remaining:
                for rot, dims in enumerate(inst["feasible"][item][k]):
                    if can_place(state, k, space, dims, inst, rule=rule):
                        feasible_items.append(item)
                        break

            if not feasible_items:
                continue

            # Volume-weighted roulette without replacement.
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
                    pos = can_place(state, k, space, dims, inst, rule=rule)
                    if pos is not None:
                        actions.append((k, space, item, dims, pos, rot))
                        if len(actions) >= max_actions:
                            return actions

            if actions:
                return actions

    return actions


# =========================================================================
# Beam search
# =========================================================================
def beam_search(inst, width=BEAM_WIDTH, max_actions=MAX_ACTIONS,
                time_limit=TIME_LIMIT, seed=None, rule=None, verbose=False):
    """
    Beam search over partial packings, scored by a greedy rollout.

    Each level expands every node in the beam into at most `max_actions`
    children, scores each by completing it with FCH, and keeps the best
    `width`. The best rollout seen anywhere is the answer, so the search can
    stop at any time and still return a complete, feasible packing.
    """

    if rule is not None:
        _sa.STABILITY_RULE = rule

    rng = random.Random(seed)

    t_start = time.time()
    deadline = t_start + time_limit

    root = initial_state(inst)
    beam = [root]

    best_score, best_state = fch(root, inst, rule=rule)

    if verbose:
        print(f"  root rollout: {wasted_space_objective(inst, best_state.packed_by_bin):.4f}")

    levels = 0
    rollouts = 1

    while beam and time.time() < deadline:

        levels += 1
        candidates = []

        for node in beam:

            if time.time() >= deadline:
                break

            actions = expand(node, inst, max_actions, rng, rule=rule)

            for (k, space, item, dims, pos, rot) in actions:

                if time.time() >= deadline:
                    break

                child = node.copy()
                apply_action(child, inst, k, space, item, dims, pos, rot)

                score, completed = fch(child, inst, rule=rule)
                rollouts += 1
                child.score = score

                if score > best_score:
                    best_score = score
                    best_state = completed
                    if verbose:
                        print(f"    [level {levels}] new best "
                              f"{wasted_space_objective(inst, completed.packed_by_bin):.4f} "
                              f"({time.time() - t_start:.1f}s)")

                candidates.append(child)

        if not candidates:
            break

        candidates.sort(key=lambda c: -c.score)
        beam = candidates[:width]

    elapsed = time.time() - t_start

    result = {
        "utilization": best_score,
        "wasted": wasted_space_objective(inst, best_state.packed_by_bin),
        "placements": best_state.placements,
        "packed_by_bin": best_state.packed_by_bin,
        "n_packed": len(best_state.placements),
        "levels": levels,
        "rollouts": rollouts,
        "runtime": elapsed,
        "width": width,
        "max_actions": max_actions,
    }

    if verbose:
        print(f"  BS: {levels} levels, {rollouts} rollouts in {elapsed:.1f}s")

    return result


# =========================================================================
# Driver
# =========================================================================
def collect_instances(args):

    root = Path(args.instance_dir)

    if args.instances:
        paths = []
        for spec in args.instances:
            matched = sorted(Path().glob(spec)) if any(
                c in spec for c in "*?[") else [Path(spec)]
            if not matched:
                raise SystemExit(f"no instance matches: {spec}")
            paths.extend(matched)
        return paths

    group_dir = root / args.group

    if not group_dir.is_dir():
        raise SystemExit(f"no such instance group: {group_dir}")

    return sorted(group_dir.glob("*.txt"))


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Beam search for the MHKP under submodel V0.")

    parser.add_argument("instances", nargs="*", default=None, metavar="INSTANCE")
    parser.add_argument("--instance-dir", default="MHKP")
    parser.add_argument("--group", default="t8_n30")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stability-rule", choices=("corners", "peraxis"),
                        default="corners")
    parser.add_argument("--compare", action="store_true",
                        help="also run SA and HC on the same instances")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    paths = collect_instances(args)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no instances selected")

    _sa.STABILITY_RULE = args.stability_rule

    hc = None
    if args.compare:
        spec = importlib.util.spec_from_file_location(
            "mhkp_v0_hc", Path(__file__).resolve().parent / "MHKP-V0-HC.py")
        hc = importlib.util.module_from_spec(spec)
        sys.modules["mhkp_v0_hc"] = hc
        spec.loader.exec_module(hc)

    print("=" * 92)
    print("MHKP submodel V0 - beam search "
          f"(w={args.width}, m={args.actions}, FCH rollout)")
    print(f"{len(paths)} instance(s) | {args.time_limit}s each | "
          f"stability rule '{args.stability_rule}'")
    print("=" * 92)

    head = f"{'instance':<26}{'BS':>9}{'packed':>9}{'levels':>8}{'rollouts':>10}{'viol':>6}"
    if args.compare:
        head += f"{'SA':>9}{'HC':>9}{'best':>6}"
    print(head)

    bs_objs, sa_objs, hc_objs = [], [], []
    total_viol = 0
    wins = {"BS": 0, "SA": 0, "HC": 0}

    for path in paths:

        inst = load_instance(path)

        r = beam_search(inst, width=args.width, max_actions=args.actions,
                        time_limit=args.time_limit, seed=args.seed,
                        rule=args.stability_rule, verbose=args.verbose)

        viol = len(verify(inst, r["placements"], stability=True,
                          rule=args.stability_rule))
        total_viol += viol
        bs_objs.append(r["wasted"])

        line = (f"{path.stem:<26}{r['wasted']:>9.4f}"
                f"{r['n_packed']:>6}/{inst['n']:<2}{r['levels']:>8}"
                f"{r['rollouts']:>10}{viol:>6}")

        if args.compare:
            s = _sa.simulated_annealing(inst, time_limit=args.time_limit,
                                        seed=args.seed, stability=True)
            h = hc.hill_climb(inst, time_limit=args.time_limit, seed=args.seed,
                              stability=True)
            sa_objs.append(s["wasted"]); hc_objs.append(h["wasted"])
            total_viol += len(verify(inst, s["placements"], stability=True,
                                     rule=args.stability_rule))
            total_viol += len(verify(inst, h["placements"], stability=True,
                                     rule=args.stability_rule))
            trio = [("BS", r["wasted"]), ("SA", s["wasted"]), ("HC", h["wasted"])]
            winner = min(trio, key=lambda t: t[1])[0]
            wins[winner] += 1
            line += f"{s['wasted']:>9.4f}{h['wasted']:>9.4f}{winner:>6}"

        print(line)

    print("-" * 92)
    summary = f"mean BS {statistics.fmean(bs_objs):.4f}"
    if args.compare:
        summary += (f"   mean SA {statistics.fmean(sa_objs):.4f}"
                    f"   mean HC {statistics.fmean(hc_objs):.4f}")
    print(summary)

    if args.compare:
        print(f"best-of-three: BS {wins['BS']}, SA {wins['SA']}, HC {wins['HC']}")

    print(f"verification: "
          f"{'all feasible' if total_viol == 0 else f'{total_viol} VIOLATIONS'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
