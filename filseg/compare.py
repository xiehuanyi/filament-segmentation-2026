"""Reproduction checks: compare two submission CSVs, or two intermediate files of the pipeline.

    python -m filseg.compare csv  A.csv B.csv          # rows, identical masks, per-filament IoU, PQ(A vs B)
    python -m filseg.compare cands A.pkl B.pkl         # candidate tables (features and masks)
    python -m filseg.compare npz  A.npz B.npz          # arrays (crops, refined crops, rescorer outputs)
    python -m filseg.compare maps DIR_A DIR_B          # uint8 probability maps (*.png) with the same names

csv: masks of a frame are pixel-disjoint in both files, so a filament of A matches at most one of B with
IoU > 0.5. PQ(A vs B) treats B as a single annotator record (PQ = 1 iff the files are identical up to the order
of the rows).
"""
import argparse
import glob
import os
import pickle

import cv2
import numpy as np
from pycocotools import mask as mu

from .metric import counts, summary
from .submission import by_frame, read_csv


def compare_csv(a_path, b_path, verbose=True):
    A, B = by_frame(read_csv(a_path)), by_frame(read_csv(b_path))
    frames = sorted(set(A) | set(B))
    tot = np.zeros(4)
    ious, same_frames, ident = [], 0, 0
    na = sum(len(v) for v in A.values())
    nb = sum(len(v) for v in B.values())
    for f in frames:
        a, b = A.get(f, []), B.get(f, [])
        tot += counts(b, a)  # B as ground truth, A as prediction
        if a and b:
            iou = np.asarray(mu.iou(a, b, [0] * len(b)))
            best = iou.max(1)
            ious += best[best > 0.5].tolist()
            ident += int((best == 1.0).sum())
        if sorted(r["counts"] for r in a) == sorted(r["counts"] for r in b):
            same_frames += 1
    s = summary(tot)
    res = dict(rows_a=na, rows_b=nb, frames=len(frames), identical_frames=same_frames, identical_filaments=ident,
               matched=int(tot[1]), unmatched_a=int(tot[2]), unmatched_b=int(tot[3]),
               mean_iou_matched=float(np.mean(ious)) if ious else 0.0,
               min_iou_matched=float(np.min(ious)) if ious else 0.0, pq_a_vs_b=s["PQ"],
               byte_identical=open(a_path, "rb").read() == open(b_path, "rb").read())
    if verbose:
        for k, v in res.items():
            print(f"{k:22s} {v}")
    return res


def compare_cands(a_path, b_path, verbose=True):
    A, B = pickle.load(open(a_path, "rb")), pickle.load(open(b_path, "rb"))
    keys = sorted(set(A) | set(B))
    diff_frames, n, same = [], 0, 0
    max_feat = 0.0
    for s in keys:
        ca, cb = A.get(s, []), B.get(s, [])
        eq = len(ca) == len(cb)
        if eq:
            for x, y in zip(ca, cb):
                n += 1
                if x["rle"]["counts"] != y["rle"]["counts"]:
                    eq = False
                    continue
                d = max((abs(float(x[k]) - float(y[k])) for k in x if k != "rle" and k in y), default=0.0)
                max_feat = max(max_feat, d)
                same += d == 0
        if not eq:
            diff_frames.append(s)
    res = dict(frames=len(keys), frames_with_different_candidates=len(diff_frames), candidates_compared=n,
               candidates_identical=same, max_feature_abs_diff=max_feat, example_frames=diff_frames[:5])
    if verbose:
        for k, v in res.items():
            print(f"{k:34s} {v}")
    return res


def compare_npz(a_path, b_path, verbose=True):
    A, B = np.load(a_path), np.load(b_path)
    res = {}
    for k in sorted(set(A.files) | set(B.files)):
        if k not in A.files or k not in B.files:
            res[k] = "missing in one file"
            continue
        x, y = A[k], B[k]
        if x.shape != y.shape:
            res[k] = f"shape {x.shape} vs {y.shape}"
        elif x.dtype.kind in "fiu":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            res[k] = dict(identical=bool((d == 0).all()), max_abs_diff=float(d.max()) if d.size else 0.0,
                          frac_diff=float((d > 0).mean()) if d.size else 0.0)
        else:
            res[k] = dict(identical=bool((x == y).all()))
    if verbose:
        for k, v in res.items():
            print(f"{k:8s} {v}")
    return res


def compare_maps(a_dir, b_dir, verbose=True):
    fa = sorted(os.path.basename(f) for f in glob.glob(f"{a_dir}/*.png"))
    n = same = 0
    maxd = 0
    frac = []
    for f in fa:
        if not os.path.exists(f"{b_dir}/{f}"):
            continue
        x = cv2.imread(f"{a_dir}/{f}", 0).astype(np.int16)
        y = cv2.imread(f"{b_dir}/{f}", 0).astype(np.int16)
        d = np.abs(x - y)
        n += 1
        same += int(d.max() == 0)
        maxd = max(maxd, int(d.max()))
        frac.append(float((d > 0).mean()))
    res = dict(maps=n, identical=same, max_abs_diff_levels=maxd, mean_frac_pixels_diff=float(np.mean(frac)) if frac else 0.0)
    if verbose:
        print(res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["csv", "cands", "npz", "maps"])
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args()
    {"csv": compare_csv, "cands": compare_cands, "npz": compare_npz, "maps": compare_maps}[a.kind](a.a, a.b)


if __name__ == "__main__":
    main()
