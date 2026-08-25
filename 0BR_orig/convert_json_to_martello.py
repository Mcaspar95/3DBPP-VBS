#!/usr/bin/python3
"""
Convert the BR*/*.json instances into the same simple text format
used by martello.py:

n W H D
w_1 h_1 d_1
...
w_n h_n d_n

Each JSON "Items" entry is an item TYPE with a "Demand" count; it is
expanded into that many individual boxes (Stock, Cost, Demand, Value,
C1_* fields are dropped).
"""

import json
import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent
BR_FOLDERS = [f"BR{i}" for i in range(1, 16)]


def convert_instance(json_path):

    with open(json_path) as f:
        data = json.load(f)

    obj = data["Objects"][0]
    W, H, D = obj["Length"], obj["Height"], obj["Depth"]

    boxes = []

    for item in data["Items"]:
        w, h, d = item["Length"], item["Height"], item["Depth"]
        demand = item["Demand"]

        for _ in range(demand):
            boxes.append((w, h, d))

    return W, H, D, boxes


def write_instance(filename, W, H, D, boxes):

    with open(filename, "w") as f:

        f.write(f"{len(boxes)} {W} {H} {D}\n")

        for w, h, d in boxes:
            f.write(f"{w} {h} {d}\n")


def convert():

    for br_folder in BR_FOLDERS:

        src_dir = SOURCE_ROOT / br_folder
        if not src_dir.exists():
            continue

        out_dir = SOURCE_ROOT / "martello_format" / br_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(
            src_dir.glob("*.json"),
            key=lambda p: int(p.stem)
        )

        for json_path in json_files:

            W, H, D, boxes = convert_instance(json_path)

            out_path = out_dir / f"{json_path.stem}.txt"

            write_instance(out_path, W, H, D, boxes)

        print(f"{br_folder}: converted {len(json_files)} instances -> {out_dir}")


if __name__ == "__main__":
    convert()
