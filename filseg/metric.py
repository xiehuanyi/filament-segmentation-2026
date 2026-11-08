"""The host's pooled Panoptic Quality (self-evaluation notebook), computed on COCO RLEs.

Every (annotator instance, prediction) pair with IoU > 0.5 is a TP; predictions hitting nothing are FP;
annotator instances hit by nothing are FN. Pooled over every annotator record of every image:
    PQ = sum IoU(TP) / (TP + 0.5 FP + 0.5 FN)
Counts are kept as a 4-vector [sum IoU of TPs, TP, FP, FN] so they can be summed over images.
"""
import numpy as np
from pycocotools import mask as mu


def counts(gt_rles, pred_rles):
    """[sum IoU(TP), TP, FP, FN] of one annotator record against one image's predictions."""
    if not gt_rles:
        return np.array([0.0, 0, len(pred_rles), 0])
    if not pred_rles:
        return np.array([0.0, 0, 0, len(gt_rles)])
    iou = np.asarray(mu.iou(pred_rles, gt_rles, [0] * len(gt_rles))).T  # (n_gt, n_pred)
    hit = iou > 0.5
    return np.array([float(iou[hit].sum()), int(hit.sum()), int((hit.sum(0) == 0).sum()),
                     int((hit.sum(1) == 0).sum())])


def image_counts(gt_sets, pred_rles):
    """Sum of counts over all annotator records of one image."""
    c = np.zeros(4)
    for g in gt_sets:
        c += counts(g, pred_rles)
    return c


def pq(c):
    c = np.asarray(c, dtype=float)
    den = c[1] + 0.5 * c[2] + 0.5 * c[3]
    return c[0] / den if den > 0 else 0.0


def summary(c):
    c = np.asarray(c, dtype=float)
    return dict(PQ=pq(c), SQ=c[0] / max(c[1], 1), TP=int(c[1]), FP=int(c[2]), FN=int(c[3]))
