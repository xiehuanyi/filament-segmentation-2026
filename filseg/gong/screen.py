"""Quality screen of unlabelled frames: per-frame disk statistics, kept when they fall inside the range of the
competition training frames (removes cloudy, saturated, defocused or odd frames).

    python -m filseg.gong.screen --dir PSEUDO_DIR          # writes PSEUDO_DIR/stats.csv and PSEUDO_DIR/keep.txt
    python -m filseg.gong.screen --train --out stats_train.csv   # the same statistics for the training frames

Statistics on the disk (r < 880 px): mean, std, std of the Laplacian of the 1-px-blurred frame (sharpness),
std of the flat-fielded frame, fraction of |flat| > 4, 1st and 99th percentiles.
Keep rule (bounds read off the training-set distribution): 117 <= mean <= 134, 12 <= std <= 20.5,
1.3 <= lapstd <= 3.9, flatstd >= 0.70, p1 >= 70, p99 <= 168. On our 1500 fetched frames it keeps 1314.
"""
import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .. import common
from ..common import normalize_flat
from ..paths import P, add_path_args, apply_path_args

DISK = common._R < 880
RULE = dict(mean=(117, 134), std=(12, 20.5), lapstd=(1.3, 3.9), flatstd=(0.70, None), p1=(70, None), p99=(None, 168))


def stats(path):
    im8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = im8.astype(np.float32)
    v = im[DISK]
    lap = cv2.Laplacian(cv2.GaussianBlur(im, (0, 0), 1.0), cv2.CV_32F)[DISK]
    x = normalize_flat(im8)[DISK]
    q = np.percentile(v, [1, 50, 99])
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem, v.mean(), v.std(), lap.std(), x.std(), (np.abs(x) > 4).mean(), q[0], q[2]


def keep(row):
    for k, (lo, hi) in RULE.items():
        v = float(row[k])
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            return False
    return True


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--dir", default=None, help="folder with images/<stem>.jpg (fetched frames)")
    ap.add_argument("--train", action="store_true", help="compute the statistics of the training frames instead")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = apply_path_args(ap.parse_args())
    fs = sorted(glob.glob(f"{P.train_images}/*.jpeg")) if a.train else sorted(glob.glob(f"{a.dir}/images/*.jp*g"))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(stats, fs, chunksize=8))
    out = a.out or f"{a.dir}/stats.csv"
    cols = ["stem", "mean", "std", "lapstd", "flatstd", "out4", "p1", "p99"]
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join([r[0]] + [f"{x:.4f}" for x in r[1:]]) + "\n")
    if not a.train:
        kept = [r["stem"] for r in csv.DictReader(open(out)) if keep(r)]
        open(f"{a.dir}/keep.txt", "w").write("\n".join(kept) + "\n")
        print(f"{len(rows)} frames, kept {len(kept)} -> {a.dir}/keep.txt", flush=True)
    else:
        print(len(rows), "training frames ->", out, flush=True)


if __name__ == "__main__":
    main()
