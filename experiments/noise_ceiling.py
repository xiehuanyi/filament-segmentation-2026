"""How predictable is a candidate's q? On frames with >= 2 annotator records, the correlation between a
candidate's matched IoU (0 if IoU <= 0.5) against record A and against record B (level-0.3 candidates that have a
refiner crop). A perfect predictor of the expected q reaches about sqrt(corr) correlation with one record's q.

    python experiments/noise_ceiling.py --tag ensB --work WORK
"""
import argparse
import os
import pickle
import sys

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filseg.common import load_gt  # noqa: E402
from filseg.paths import P, add_path_args, apply_path_args  # noqa: E402


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tag", default="ensB")
    a = apply_path_args(ap.parse_args())
    gt = load_gt()
    C = pickle.load(open(f"{P.cands}/{a.tag}_oof.pkl", "rb"))
    d = np.load(f"{P.refine}/{a.tag}_oof.npz")
    keep = {(str(s), int(j)) for s, j in zip(d["stem"], d["j"])}
    A, B = [], []
    for s, cs in C.items():
        if len(gt[s]) < 2:
            continue
        idx = [j for j, c in enumerate(cs) if abs(c["t"] - 0.3) < 1e-6 and (s, j) in keep]
        if not idx:
            continue
        rl = [cs[j]["rle"] for j in idx]
        per = []
        for g in gt[s]:
            if not g:
                per.append(np.zeros(len(idx)))
                continue
            iou = np.asarray(mu.iou(rl, g, [0] * len(g)))
            per.append(np.where(iou.max(1) > 0.5, iou.max(1), 0.0))
        A += list(per[0])
        B += list(per[1])
    A, B = np.array(A), np.array(B)
    r = np.corrcoef(A, B)[0, 1]
    print(f"candidates {len(A)}  corr(record A, record B) = {r:.3f}  P(match A) = {(A > 0).mean():.3f}  "
          f"agreement on match = {((A > 0) == (B > 0)).mean():.3f}  -> ceiling corr ~{np.sqrt(r):.3f}")


if __name__ == "__main__":
    main()
