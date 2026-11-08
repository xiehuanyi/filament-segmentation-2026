"""Submission CSV: one row per predicted filament, ``filament_id,segmentation_rle`` with filament_id =
``<frame stem>_<k>`` (k = 1, 2, ... in decreasing score order) and segmentation_rle = compressed COCO RLE counts
of the 2048 x 2048 mask. Masks of one frame are pixel-disjoint.

    python -m filseg.submission --check sub.csv [--images DIR]
"""
import argparse
import re

import numpy as np
from pycocotools import mask as mu

from .common import H, W, list_stems
from .paths import P, add_path_args, apply_path_args


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, cnt in rows:
            assert "," not in cnt and '"' not in cnt
            f.write(f"{fid},{cnt}\n")
    print(f"wrote {path}: {len(rows)} filaments on {len({r[0].rsplit('_', 1)[0] for r in rows})} frames", flush=True)


def read_csv(path):
    rows = []
    with open(path) as f:
        assert f.readline().strip() == "filament_id,segmentation_rle"
        for line in f:
            fid, cnt = line.rstrip("\n").split(",", 1)
            rows.append((fid, cnt))
    return rows


def by_frame(rows):
    out = {}
    for fid, cnt in rows:
        out.setdefault(fid.rsplit("_", 1)[0], []).append({"size": [H, W], "counts": cnt.encode("ascii")})
    return out


def validate(rows, stems):
    """Unique ids, ids name known frames, every mask is non-empty and masks of a frame do not overlap."""
    ids = [r[0] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for fid in ids:
        assert re.fullmatch(r"\d{14}[A-Z][a-z]_\d+", fid), fid
        assert fid.rsplit("_", 1)[0] in stems, fid
    for st, rles in by_frame(rows).items():
        acc = np.zeros((H, W), np.uint8)
        for r in rles:
            m = mu.decode(r)
            assert m.sum() > 0
            acc += m
        assert acc.max() <= 1, f"overlap in {st}"
    return True


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--check", required=True)
    a = apply_path_args(ap.parse_args())
    rows = read_csv(a.check)
    validate(rows, set(list_stems(P.images)))
    print(f"{a.check}: {len(rows)} rows, {len(by_frame(rows))} frames, valid", flush=True)


if __name__ == "__main__":
    main()
