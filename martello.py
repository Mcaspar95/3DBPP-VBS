from pathlib import Path


class MartelloRandom:
    """
    Exact reproduction of srand48x() and lrand48x()
    from the original C code.
    """

    def __init__(self, seed):
        self.srand(seed)

    def srand(self, seed):
        self._h48 = seed
        self._l48 = 0x330E

    def lrand(self):
        self._h48 = (
            self._h48 * 0xDEECE66D
            + self._l48 * 0x5DEEC
        )

        self._l48 = (
            self._l48 * 0xE66D
            + 0xB
        )

        self._h48 = self._h48 + (self._l48 >> 16)
        self._l48 = self._l48 & 0xFFFF

        return self._h48 >> 1

    def randm(self, x):
        return self.lrand() % x

    def rint(self, a, b):
        return self.randm(b - a + 1) + a


def generate_item(rng, W, H, D, requested_type):
    """
    Exact reproduction of randomtype() for Martello classes 1-5.
    """

    actual_type = requested_type

    # Original C code:
    # t = rint(1,10);
    # if (t <= 5) type = t;
    #
    # Therefore:
    # requested type occurs with probability 6/10
    # IF t > 5, because requested_type remains unchanged.
    #
    # The other four types occur with probability 1/10 each,
    # except when t == requested_type among 1..5.
    t = rng.rint(1, 10)

    if t <= 5:
        actual_type = t

    if actual_type == 1:
        w = rng.rint(1, W // 2)
        h = rng.rint(2 * H // 3, H)
        d = rng.rint(2 * D // 3, D)

    elif actual_type == 2:
        w = rng.rint(2 * W // 3, W)
        h = rng.rint(1, H // 2)
        d = rng.rint(2 * D // 3, D)

    elif actual_type == 3:
        w = rng.rint(2 * W // 3, W)
        h = rng.rint(2 * H // 3, H)
        d = rng.rint(1, D // 2)

    elif actual_type == 4:
        # Exact original code:
        # w = rint(W/2, H)
        w = rng.rint(W // 2, H)
        h = rng.rint(H // 2, H)
        d = rng.rint(D // 2, D)

    elif actual_type == 5:
        w = rng.rint(1, W // 2)
        h = rng.rint(1, H // 2)
        d = rng.rint(1, D // 2)

    return actual_type, w, h, d


CLASS_BDIM = {
    1: 100,
    2: 100,
    3: 100,
    4: 100,
    5: 100,
    6: 100,
    7: 400,
    8: 1000,
}

CLASS_MAXDIM = {
    6: 100,
    7: 350,
    8: 1000,
}


def generate_uniform_item(rng, bdim, item_class):
    """
    Classes 6-8: li, wi, hi uniformly random in [1, maxdim],
    independent of bin size (bin is L=W=H=bdim).
    """

    maxdim = CLASS_MAXDIM[item_class]

    w = rng.rint(1, maxdim)
    h = rng.rint(1, maxdim)
    d = rng.rint(1, maxdim)

    return w, h, d


def generate_instance(
    n,
    bdim,
    requested_type,
    instance_number
):
    """
    Generate one Martello instance.

    Original seed:
        srand(v + n)

    where v = 1,...,10.
    """

    if requested_type not in range(1, 9):
        raise ValueError("requested_type must be between 1 and 8")

    if not 1 <= instance_number <= 10:
        raise ValueError("instance_number must be between 1 and 10")

    seed = n + instance_number

    rng = MartelloRandom(seed)

    W = H = D = bdim

    items = []

    if requested_type in (1, 2, 3, 4):

        for item_id in range(1, n + 1):

            actual_type, w, h, d = generate_item(
                rng,
                W,
                H,
                D,
                requested_type
            )

            items.append({
                "id": item_id,
                "class": actual_type,
                "width": w,
                "height": h,
                "depth": d
            })

    elif requested_type == 5:

        # 11 boxes of type 5, drawn with the same biased randomtype()
        # logic as classes 1-4, plus 1 box of each of the other
        # 4 types (1-4) so that some boxes are guaranteed not to load.
        for item_id in range(1, 12):

            actual_type, w, h, d = generate_item(
                rng,
                W,
                H,
                D,
                requested_type
            )

            items.append({
                "id": item_id,
                "class": actual_type,
                "width": w,
                "height": h,
                "depth": d
            })

        item_id = 12

        for other_type in (1, 2, 3, 4):

            _, w, h, d = generate_item(
                rng,
                W,
                H,
                D,
                other_type
            )

            items.append({
                "id": item_id,
                "class": other_type,
                "width": w,
                "height": h,
                "depth": d
            })

            item_id += 1

    else:
        # Classes 6, 7, 8
        for item_id in range(1, n + 1):

            w, h, d = generate_uniform_item(rng, bdim, requested_type)

            items.append({
                "id": item_id,
                "class": requested_type,
                "width": w,
                "height": h,
                "depth": d
            })

    return W, H, D, items


def write_instance(filename, W, H, D, items):
    """
    Write in the original simple format:

    n W H D
    w_1 h_1 d_1
    ...
    w_n h_n d_n
    """

    with open(filename, "w") as f:

        f.write(f"{len(items)} {W} {H} {D}\n")

        for item in items:
            f.write(
                f"{item['width']} "
                f"{item['height']} "
                f"{item['depth']}\n"
            )


CLASS_N = {
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 15,
    6: 10,
    7: 10,
    8: 10,
}


def generate_all_classes(
    output_directory="martello_instances"
):
    """
    Generate:
        Classes 1, 2, 3, 4, 5, 6, 7, 8
        10 instances per class
        (10 boxes/instance, except Class 5 which has 15)

    Total: 80 instances.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance_class in (1, 2, 3, 4, 5, 6, 7, 8):

        class_dir = output_dir / f"class_{instance_class}"
        class_dir.mkdir(exist_ok=True)

        n = CLASS_N[instance_class]
        bdim = CLASS_BDIM[instance_class]

        for instance_number in range(1, 11):

            W, H, D, items = generate_instance(
                n=n,
                bdim=bdim,
                requested_type=instance_class,
                instance_number=instance_number
            )

            filename = (
                class_dir
                / f"class{instance_class}_"
                  f"n{n}_"
                  f"instance{instance_number}.txt"
            )

            write_instance(
                filename,
                W,
                H,
                D,
                items
            )

    print(f"Generated instances in: {output_dir}")

# Guarded so that this module can be imported (e.g. by custom_instances.py,
# which reuses MartelloRandom and write_instance) without regenerating the
# instances as a side effect.
if __name__ == "__main__":
    generate_all_classes(
        output_directory="Martello"
    )