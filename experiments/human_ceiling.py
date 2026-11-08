"""Human baselines on frames with 3 annotator records, scored like the metric (pooled PQ) against each held-out
record: another single annotator, the intersection and the union of the other two (connected components >= 30 px),
and our OOF predictions against the same records. Also our pooled PQ by number of records per frame.

    python experiments/human_ceiling.py --pred WORK/oof_s15.csv --work WORK
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filseg.common import load_gt, load_meta  # noqa: E402
from filseg.metric import counts, image_counts, summary  # noqa: E402
from filseg.paths import add_path_args, apply_path_args  # noqa: E402
from filseg.submission import by_frame, read_csv  # noqa: E402

GT = PRED = None


def instances(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [mu.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in range(1, n) if (lab == i).sum() >= 30]


def work(s):
    sets = GT[s]
    res = {"n": len(sets), "ours": image_counts(sets, PRED.get(s, []))}
    if len(sets) == 3 and all(sets):
        U = [mu.decode(g).max(2) > 0 for g in sets]
        loo = {"single": np.zeros(4), "inter2": np.zeros(4), "union2": np.zeros(4)}
        for k in range(3):
            o = [U[i] for i in range(3) if i != k]
            loo["inter2"] += counts(sets[k], instances(o[0] & o[1]))
            loo["union2"] += counts(sets[k], instances(o[0] | o[1]))
            loo["single"] += counts(sets[k], list(sets[(k + 1) % 3]))
        res.update(loo)
    return res


def main():
    global GT, PRED
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--pred", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = apply_path_args(ap.parse_args())
    GT = load_gt()
    PRED = by_frame(read_csv(a.pred))
    with ProcessPoolExecutor(a.workers) as ex:
        R = list(ex.map(work, load_meta()["stems"], chunksize=2))
    for n in (1, 2, 3):
        c = sum(r["ours"] for r in R if r["n"] == n)
        print(f"ours on frames with {n} records ({sum(r['n'] == n for r in R)} frames): {summary(c)}")
    three = [r for r in R if "inter2" in r]
    print(f"frames with 3 non-empty records: {len(three)}")
    print("ours vs each record          ", summary(sum(r["ours"] for r in three)))
    for k in ("single", "inter2", "union2"):
        print(f"human {k:7s} vs held-out   ", summary(sum(r[k] for r in three)))


if __name__ == "__main__":
    main()
