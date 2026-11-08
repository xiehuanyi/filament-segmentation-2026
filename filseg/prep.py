"""Stage 0: parse the MAGFiLO COCO json into per-image ground truth, soft consensus targets and day-grouped folds.

    python -m filseg.prep [--data DATA] [--work WORK]

Writes under $FIL_WORK/cache:
  gt.pkl               {stem: [[rle, ...] for each annotator record]}  (pycocotools RLE dicts)
  meta.json            {stems, test_stems, folds {stem: fold}, n_sets {stem: k}, categories}
  targets/<stem>.png   uint8 mean over annotator records of the record's union mask, x255 (the soft training target)
  sep/<stem>.png       uint8 255 where >= 2 instances of one record touch (optional auxiliary target, unused by s15/s16)

Folds: the 707 training frames are grouped by observation day (frames of one day from different GONG sites
show the same filaments), days are shuffled with seed 2026 and dealt round-robin into 5 folds.
"""
import argparse
import collections
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

from .common import H, W, list_stems
from .paths import P, add_path_args, apply_path_args


def load(path):
    coco = json.load(open(path))
    stem_of = {im["id"]: os.path.splitext(im["file_name"])[0] for im in coco["images"]}
    sets_of = collections.defaultdict(list)
    for im in coco["images"]:
        sets_of[stem_of[im["id"]]].append(im["id"])
    anns = collections.defaultdict(list)
    for a in coco["annotations"]:
        rle = mu.merge(mu.frPyObjects(a["segmentation"], H, W))
        anns[a["image_id"]].append(rle)
    gt = {s: [anns[a] for a in sets] for s, sets in sets_of.items()}
    return gt, coco


def make_target(args):
    stem, sets, out = args
    acc = np.zeros((H, W), np.float32)
    sep = np.zeros((H, W), np.uint8)
    k = np.ones((3, 3), np.uint8)
    for rles in sets:
        if not rles:
            continue
        m = mu.decode(rles)  # H,W,n
        u = (m.sum(2) > 0)
        acc += u
        if m.shape[2] > 1:
            cnt = np.zeros((H, W), np.uint8)
            for j in range(m.shape[2]):
                cnt += cv2.dilate(m[:, :, j], k) > 0
            sep |= (cnt >= 2).astype(np.uint8)
    acc /= max(len(sets), 1)
    cv2.imwrite(os.path.join(out, "targets", stem + ".png"), np.round(acc * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out, "sep", stem + ".png"), sep * 255)
    return stem


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    a = apply_path_args(ap.parse_args())
    out = P.cache
    os.makedirs(os.path.join(out, "targets"), exist_ok=True)
    os.makedirs(os.path.join(out, "sep"), exist_ok=True)
    gt, coco = load(P.train_json)
    stems = sorted(gt)
    test_stems = list_stems(P.images)
    # folds grouped by calendar day (same-day frames from different sites see the same filaments)
    days = sorted({s[:8] for s in stems})
    rng = np.random.RandomState(2026)
    perm = rng.permutation(len(days))
    day_fold = {days[i]: int(j % a.folds) for j, i in enumerate(perm)}
    folds = {s: day_fold[s[:8]] for s in stems}
    with open(os.path.join(out, "gt.pkl"), "wb") as f:
        pickle.dump(gt, f)
    json.dump({"stems": stems, "test_stems": test_stems, "folds": folds,
               "n_sets": {s: len(gt[s]) for s in stems},
               "categories": coco["categories"]}, open(os.path.join(out, "meta.json"), "w"))
    print("images", len(stems), "test", len(test_stems), "fold sizes", collections.Counter(folds.values()))
    with ProcessPoolExecutor(a.workers) as ex:
        for _ in ex.map(make_target, [(s, gt[s], out) for s in stems]):
            pass
    n_inst = [len(r) for s in stems for r in gt[s]]
    areas = [int(mu.area(r)) for s in stems for rs in gt[s] for r in rs]
    print("annotator records", len(n_inst), "instances/record mean", np.mean(n_inst), "median", np.median(n_inst))
    print("instance area percentiles 1/5/10/25/50/75/90/99:", np.percentile(areas, [1, 5, 10, 25, 50, 75, 90, 99]))


if __name__ == "__main__":
    main()
