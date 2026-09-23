"""
MHKP instances for submodel V0 of Deplano et al. (2019)
=======================================================

Generator and reader for the instances used by MHKP-V0-SA.py, following

    Deplano, I., Lersteau, C., Nguyen, T.T. (2019/2021)
    "A mixed-integer linear model for the multiple heterogeneous knapsack
     problem with realistic container loading constraints and bins' priority"
    Intl. Trans. in Op. Res. 28(6), 3244-3275.

Why this file exists
--------------------
MHKP-V0-SA.py originally built its instances in memory from a seeded RNG, so
nothing was ever written to disk: a run could be repeated only by repeating the
exact seed AND the exact generator code, and a change to the generator silently
invalidated every earlier result. Persisting the instances decouples the two, so
that a packing can be re-checked, an instance inspected by hand, and results
compared across changes to the solver.

Relation to the paper's own instances
-------------------------------------
The paper draws bins and items from the Physical Internet container catalogue
of Landschuetzer et al. (2015) - 24 bin types and 440 item types. That catalogue
is not published with the paper, the paper carries no data availability
statement, and it is not in this repository. These instances therefore follow
the RULES the paper states in Section 5 but cannot reproduce its draws:

  * bins are picked from a set of types, items are picked such that every item
    fits at least one bin (Section 5, "there is no loss of generality because
    this procedure is equivalent to filtering unfitting items before the
    optimisation task");
  * the item distribution is skewed toward larger items - the paper orders the
    catalogue by volume and takes 70% from above the median and 30% from below.

Absolute objective values from these files are consequently NOT comparable with
the paper's Table 7. What they support is the comparison the port is for: the
price of the stability constraint, and solver-against-solver on fixed inputs.

The weight, load-bearing and centre-of-mass attributes of the paper's full model
are deliberately NOT generated. Submodel V0 (constraints (1)-(10r)) reads none
of them; they belong to V1 and V2 and would be invented data here. The file
format has room for them if this is ever extended.

File format
-----------
The repository's existing format (martello.py, custom_instances.py) is

    n W H L
    w_1 h_1 l_1
    ...

which describes a single bin and cannot express the MHKP's several
heterogeneous bins or their priorities. This format extends it with named
sections, keeping the same "one line per physical box" spirit:

    # comments begin with a hash
    BINS
    <length> <width> <height> <priority>
    ...
    ITEMS
    <length> <width> <height>
    ...

Lengths are integers on the axes the model uses: length along Z, width along X,
height along Y. A bin's priority is written explicitly rather than implied, so
that a file can carry the paper's p_j = 1/volume case study (formula (2)) or any
other weighting without the reader having to guess which was meant.

Usage
-----
    python3 mhkp_instances.py                      # the full corpus
    python3 mhkp_instances.py --sets 10            # a small corpus, for a
                                                   # quick check
    python3 mhkp_instances.py --output-directory MHKP
"""

import argparse
import random
from math import gcd
from pathlib import Path


# The paper's Table 7 experiments use the smallest bin, 120 x 120 x 120.
BASE_BIN = (120, 120, 120)

# A small catalogue of bin types standing in for the Physical Internet set.
# Sizes are multiples of the 120 module, as the PI containers are modular.
BIN_TYPES = [
    (120, 120, 120),
    (144, 120, 120),
    (168, 144, 120),
    (192, 168, 144),
    (240, 192, 168),
]

# The corpus: (name, item count, bin count, instances).
#
# The instance COUNTS follow the paper's own experimental design, which differs
# per table and is easy to over-generalise:
#
#   * Table 7 - ten items, one bin - is explicitly averaged over 1000 RANDOM
#     INSTANCES, because it reports mean features (packed items, weights, load
#     bearing, eccentricity, asymmetry) where a small sample would be noise.
#     n10_b1 therefore holds 1000 instances and is the group to use when
#     comparing against that table.
#
#   * Tables 8 and 10 - one bin from 18 to 90 items, and the multi-bin runs -
#     report ONE instance per size, since each is a single MIP solve against a
#     3600 s limit. The remaining groups here hold 25: enough to average a
#     stochastic search over, without pretending to a precision the paper's own
#     design does not claim at those sizes.
#
# Multi-bin groups exercise the priority objective of formula (1), which is
# inert when there is only one bin.
GROUPS = [
    ("n10_b1",  10, 1, 1000),   # Table 7: averaged over 1000 random instances
    ("n18_b1",  18, 1,   25),
    ("n30_b1",  30, 1,   25),
    ("n24_b3",  24, 3,   25),
    ("n40_b5",  40, 5,   25),
]

# Table 8 sweeps one bin from 18 to 90 items, one instance per size, and reports
# a proven optimum (Gap% = 0) for most of them. Those rows are the only hard
# targets the paper publishes for V0, so the sizes are reproduced here to sit
# beside them. The paper's own instances are not published, so these cannot be
# the same draws - see the note at the top of this file. Several instances per
# size are generated rather than one, because a single draw at a given size
# says nothing about a stochastic solver.
TABLE8_SIZES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                35, 40, 45, 50, 55, 60, 65, 70, 80, 90]

TABLE8_SETS = 10

# Table 11 of Deplano et al. (2019) is WFBF vs the V2 MILP on 1 bin, n = 18-90
# - exactly the sizes and instance shape TABLE8_SIZES already covers (the two
# tables share the same size sweep; the difference is which methods they
# compare). The t8_n* groups therefore double as the Table 11 grid; no
# separate generator is needed for it.
#
# Table 12 continues the same sweep, 1 bin, n = 100-200 in steps of 10. It is
# a distinct size range from Table 8/11, so it gets its own groups here.
TABLE12_SIZES = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

TABLE12_SETS = 10


def item_fits_any_bin(dims, bins):
    """
    True if the item fits some bin under the two rotations of Table 2.

    Rotation is around the vertical axis only, so length and width may swap
    while height is fixed.
    """

    l, w, h = dims

    for (L, W, H) in bins:
        if h <= H and ((l <= L and w <= W) or (w <= L and l <= W)):
            return True

    return False


def generate_items(n, bins, rng, large_fraction=0.7):
    """
    Draw n items that each fit at least one of `bins`.

    The paper skews its draws toward larger items: the catalogue is ordered by
    volume and an instance takes 70% from above the median and 30% from below.
    That is reproduced here by drawing each item from one of two size bands
    rather than from a single uniform range, which is what makes the instances
    hard enough to be interesting - with only small items every instance packs
    completely and the solver has nothing to decide.
    """

    max_L = max(b[0] for b in bins)
    max_W = max(b[1] for b in bins)
    max_H = max(b[2] for b in bins)

    items = []

    while len(items) < n:

        if rng.random() < large_fraction:
            lo, hi = 0.35, 0.60          # above the median
        else:
            lo, hi = 0.15, 0.35          # below it

        dims = (
            rng.randint(max(1, int(lo * max_L)), max(2, int(hi * max_L))),
            rng.randint(max(1, int(lo * max_W)), max(2, int(hi * max_W))),
            rng.randint(max(1, int(lo * max_H)), max(2, int(hi * max_H))),
        )

        if not item_fits_any_bin(dims, bins):
            continue

        items.append(dims)

    return items


def generate_instance(n_items, n_bins, rng):
    """Build one instance as (bins, items), bins carrying their priority."""

    if n_bins == 1:
        shapes = [BASE_BIN]
    else:
        shapes = BIN_TYPES[:n_bins]

    # Formula (2): priority is the inverse of the volume, which is what makes
    # filling the small bins preferable.
    bins = [(L, W, H, 1.0 / (L * W * H)) for (L, W, H) in shapes]

    items = generate_items(n_items, shapes, rng)

    return bins, items


def grid_beta(items):
    """
    Definition 4: beta is the gcd of all item sides, which keeps items aligned
    on a common grid. It is derived rather than stored, so that a file cannot
    disagree with itself.
    """

    beta = 0

    for dims in items:
        for d in dims:
            beta = gcd(beta, d)

    return max(1, beta)


def write_instance(path, bins, items):
    """Write one instance in the BINS/ITEMS format documented above."""

    with open(path, "w") as f:

        f.write("# MHKP instance for submodel V0 of Deplano et al. (2019)\n")
        f.write("# BINS:  length width height priority\n")
        f.write("# ITEMS: length width height\n")
        f.write(f"# beta (gcd of item sides) = {grid_beta(items)}\n")

        f.write("BINS\n")
        for (L, W, H, p) in bins:
            f.write(f"{L} {W} {H} {p:.12g}\n")

        f.write("ITEMS\n")
        for (l, w, h) in items:
            f.write(f"{l} {w} {h}\n")


def read_instance(path):
    """
    Read an instance written by write_instance().

    Returns (bins, items) with bins as (L, W, H, priority) and items as
    (l, w, h). Raises ValueError with the file and line on malformed input,
    so that a bad instance fails loudly rather than being silently skipped.
    """

    bins, items = [], []
    section = None

    with open(path) as f:

        for lineno, raw in enumerate(f, 1):

            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            upper = line.upper()

            if upper == "BINS":
                section = "bins"
                continue

            if upper == "ITEMS":
                section = "items"
                continue

            parts = line.split()

            try:
                if section == "bins":
                    if len(parts) != 4:
                        raise ValueError(
                            f"expected 4 fields, got {len(parts)}")
                    bins.append((int(parts[0]), int(parts[1]),
                                 int(parts[2]), float(parts[3])))
                elif section == "items":
                    if len(parts) != 3:
                        raise ValueError(
                            f"expected 3 fields, got {len(parts)}")
                    items.append((int(parts[0]), int(parts[1]), int(parts[2])))
                else:
                    raise ValueError("data line outside a BINS/ITEMS section")
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}\n  {line!r}") from None

    if not bins:
        raise ValueError(f"{path}: no bins found")

    if not items:
        raise ValueError(f"{path}: no items found")

    return bins, items


def generate_all(output_directory="MHKP", sets=None, seed=20260917,
                 table8=False, table12=False):
    """
    Generate the whole corpus.

    Each group is seeded from its own name, so that regenerating one group -
    or adding a group later - cannot disturb the draws of any other. That
    would not hold with a single shared stream, where inserting a group shifts
    every subsequent draw and silently invalidates stored results.

    `sets` overrides the per-group instance count for every group; leave it as
    None to use the counts in GROUPS, which follow the paper's design.
    """

    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)

    total = 0

    groups = list(GROUPS)

    if table8:
        # One group per Table 8 size, so that a size can be run on its own.
        # This grid also serves Table 11, which compares WFBF against V2 on
        # the same 1-bin, n=18-90 sweep - see the note above TABLE8_SIZES.
        groups += [(f"t8_n{n}", n, 1, TABLE8_SETS) for n in TABLE8_SIZES]

    if table12:
        # Table 12: 1 bin, n = 100-200. Named t12_n* to keep it apart from
        # the t8_n*/Table 11 range, since both are 1-bin single-size groups
        # and could otherwise collide if the ranges ever overlapped.
        groups += [(f"t12_n{n}", n, 1, TABLE12_SETS) for n in TABLE12_SIZES]

    for (name, n_items, n_bins, default_count) in groups:

        count = default_count if sets is None else sets

        group_dir = out / name
        group_dir.mkdir(exist_ok=True)

        # A per-group stream, derived from the corpus seed and the group name.
        rng = random.Random(f"{seed}:{name}")

        # Instance files are zero-padded so that a lexical listing is also a
        # numeric one; with 1000 instances "instance10" would otherwise sort
        # between "instance1" and "instance2".
        width = len(str(count))

        for number in range(1, count + 1):

            bins, items = generate_instance(n_items, n_bins, rng)

            write_instance(
                group_dir / f"{name}_instance{number:0{width}d}.txt",
                bins, items)
            total += 1

        print(f"{name}: {count} instances, {n_items} items, {n_bins} bin(s)")

    print(f"\nGenerated {total} instances in: {out}")
    print(f"Reproducible from seed {seed}")

    return out


def main():

    parser = argparse.ArgumentParser(
        description="Generate MHKP instances for submodel V0 of "
                    "Deplano et al. (2019).")

    parser.add_argument("--output-directory", default="MHKP",
                        help="where to write the instances (default: MHKP)")
    parser.add_argument("--sets", type=int, default=None,
                        help="override the instances per group; the default "
                             "uses the per-group counts of GROUPS, which "
                             "follow the paper (1000 for the Table 7 group)")
    parser.add_argument("--seed", type=int, default=20260917,
                        help="seed for the whole corpus")
    parser.add_argument("--table8", action="store_true",
                        help="also generate the Table 8/11 size sweep: one "
                             f"bin, {len(TABLE8_SIZES)} sizes from "
                             f"{TABLE8_SIZES[0]} to {TABLE8_SIZES[-1]} items")
    parser.add_argument("--table12", action="store_true",
                        help="also generate the Table 12 size sweep: one "
                             f"bin, {len(TABLE12_SIZES)} sizes from "
                             f"{TABLE12_SIZES[0]} to {TABLE12_SIZES[-1]} "
                             f"items")

    args = parser.parse_args()

    generate_all(output_directory=args.output_directory,
                 sets=args.sets, seed=args.seed, table8=args.table8,
                 table12=args.table12)


if __name__ == "__main__":
    main()
