"""
Custom-sized randomly generated instances (Section 5.1.3)
=========================================================

Generator for the medium-sized instances of

    Xu, X., Wu, B., Ma, Z., Yu, Y. (2026)
    "The three-dimensional bin packing problem with variable box size"
    Transportation Research Part E 214, 105038.

Section 5.1.3 fills the gap between the Martello instances (10-15 boxes,
see martello.py) and the BR instances (81-146 boxes) with test cases of
10-50 boxes, "adapting the methodology from the Martello instances,
modifying only the box size ranges".

Bin size
--------
L = W = H = 1000 mm for every class.

Box types (Table 6)
-------------------
    Box type    li              wi              hi
    type10      [1, L]          [1, W]          [1, H]
    type20      ]L/3, 2L/3]     ]W/3, 2W/3]     [1, H/3]
    type30      ]2L/3, L]       [1, W/3]        [1, H/3]
    type40      [1, L/4]        ]W/4, W/2]      ]H/4, 3H/4]
    type50      [1, 2L/5]       [1, 2W/5]       ]H/5, 2H/5]

The "]a, b]" notation denotes a half-open interval that excludes the lower
bound, so those draws start at floor(a) + 1.

Classes
-------
Five classes, where the class number is the NUMBER OF BOXES in the
instance, not the box type:

    Class10 -> 10 boxes, Class20 -> 20 boxes, ... Class50 -> 50 boxes

Type mixture, following the same biased scheme as the Martello classes:

    "Class i: type i with probability 60%, other four types with
     probability 10% each, i in {10,20,30,40,50}"

so the class's own type dominates. This reuses the randomtype() logic of
martello.py: draw t in [1,10]; if t <= 5 the type becomes t, otherwise the
class's own type is kept. That yields 60% for the class's type (t in 6..10
plus the case t == own type) and 10% for each of the other four.

Section 5.4.3 evaluates "10 sets of instances following the generation
rules described in Section 5.1.3", so ten instances per class are produced
here as well.

Output
------
Written in the same text format as martello.py, so that both solvers read
them without changes:

    n W H D
    w_1 h_1 d_1
    ...

Usage
-----
    python3 custom_instances.py
    python3 custom_instances.py --output-directory Custom --sets 10
"""

import argparse
from pathlib import Path

from martello import MartelloRandom, write_instance


# Bin size for every class in Section 5.1.3.
BDIM = 1000

# The five classes; the number is the box count per instance.
CLASSES = [10, 20, 30, 40, 50]

# Position of each class in the type mixture: the biased draw works on
# type indices 1..5, which map to type10..type50.
CLASS_TO_TYPE = {10: 1, 20: 2, 30: 3, 40: 4, 50: 5}


def generate_item(rng, L, W, H, requested_type):
    """
    Draw one box of the given type, following Table 6.

    Mirrors randomtype() of martello.py: the requested type is kept with
    probability 60% and replaced by one of the five types otherwise, which
    gives the other four types 10% each.

    Returns (actual_type, w, h, d) with the same (width, height, depth)
    ordering that write_instance() expects.
    """

    actual_type = requested_type

    t = rng.rint(1, 10)

    if t <= 5:
        actual_type = t

    if actual_type == 1:
        # type10: [1, L] x [1, W] x [1, H]
        l = rng.rint(1, L)
        w = rng.rint(1, W)
        h = rng.rint(1, H)

    elif actual_type == 2:
        # type20: ]L/3, 2L/3] x ]W/3, 2W/3] x [1, H/3]
        l = rng.rint(L // 3 + 1, 2 * L // 3)
        w = rng.rint(W // 3 + 1, 2 * W // 3)
        h = rng.rint(1, H // 3)

    elif actual_type == 3:
        # type30: ]2L/3, L] x [1, W/3] x [1, H/3]
        l = rng.rint(2 * L // 3 + 1, L)
        w = rng.rint(1, W // 3)
        h = rng.rint(1, H // 3)

    elif actual_type == 4:
        # type40: [1, L/4] x ]W/4, W/2] x ]H/4, 3H/4]
        l = rng.rint(1, L // 4)
        w = rng.rint(W // 4 + 1, W // 2)
        h = rng.rint(H // 4 + 1, 3 * H // 4)

    elif actual_type == 5:
        # type50: [1, 2L/5] x [1, 2W/5] x ]H/5, 2H/5]
        l = rng.rint(1, 2 * L // 5)
        w = rng.rint(1, 2 * W // 5)
        h = rng.rint(H // 5 + 1, 2 * H // 5)

    # write_instance() writes "w h d", i.e. width, height, depth, with the
    # depth playing the role of the length.
    return actual_type, w, h, l


def generate_instance(n, bdim, instance_number):
    """
    Generate one instance of the class that holds n boxes.

    The seed follows martello.py's convention, srand(v + n) with
    v = 1..10, so instances are reproducible and differ per class.
    """

    if n not in CLASSES:
        raise ValueError(f"n must be one of {CLASSES}")

    if not 1 <= instance_number <= 10:
        raise ValueError("instance_number must be between 1 and 10")

    rng = MartelloRandom(n + instance_number)

    L = W = H = bdim

    requested_type = CLASS_TO_TYPE[n]

    items = []

    for item_id in range(1, n + 1):

        actual_type, w, h, d = generate_item(rng, L, W, H, requested_type)

        items.append({
            "id": item_id,
            "class": actual_type,
            "width": w,
            "height": h,
            "depth": d,
        })

    return W, H, L, items


def generate_all_classes(output_directory="Custom", sets=10):
    """
    Generate every class of Section 5.1.3.

    Section 5.4.3 uses ten sets of these instances, so `sets` defaults
    to 10, giving 5 classes x 10 = 50 instances.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in CLASSES:

        class_dir = output_dir / f"class_{n}"
        class_dir.mkdir(exist_ok=True)

        for instance_number in range(1, sets + 1):

            W, H, D, items = generate_instance(
                n=n,
                bdim=BDIM,
                instance_number=instance_number,
            )

            filename = (
                class_dir
                / f"class{n}_n{n}_instance{instance_number}.txt"
            )

            write_instance(filename, W, H, D, items)

        print(f"class_{n}: {sets} instances with {n} boxes each")

    print(f"\nGenerated instances in: {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description="Generate the custom-sized instances of Section 5.1.3."
    )

    parser.add_argument(
        "--output-directory", default="Custom",
        help="where to write the instances (default: Custom)",
    )
    parser.add_argument(
        "--sets", type=int, default=10,
        help="instances per class (default: 10, as used in Section 5.4.3)",
    )

    args = parser.parse_args()

    generate_all_classes(
        output_directory=args.output_directory,
        sets=args.sets,
    )


if __name__ == "__main__":
    main()
