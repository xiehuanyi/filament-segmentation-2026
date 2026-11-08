"""Evaluation in the form of the host's self-evaluation notebook (azimahmadzadeh/self-evaluation-notebook):
pooled PQ, the distributions of IoU and Dice over overlapping (annotator instance, prediction) pairs, and the
one-to-many / many-to-one matching counts.

    python -m filseg.evaluate --pred oof_s15.csv [--out_dir figures --name s15]

Ground truth = every annotator record of every training frame in the CSV's frames (cache/gt.pkl); as in the
notebook, each (annotator record, frame) is one "annotator-image" and records without any instance do not appear.
For each annotator-image the IoU matrix (n_gt x n_pred) is computed on the masks (pycocotools; Dice = 2 IoU / (1 + IoU)).
  PQ          pairs with IoU > 0.5 are TP; unmatched predictions FP, unmatched instances FN (pooled)
  IoU / Dice  all pairs with IoU > 0 (as plotted by the notebook), and the TP pairs only
  1:n         for each annotator instance, the number of predictions it overlaps (IoU > 0); 1:0 = missed
  n:1         for each prediction, the number of instances of that record it overlaps; 0:1 = spurious
Writes <out_dir>/<name>_iou.png, _dice.png, _m2n.png and <name>_eval.json.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from pycocotools import mask as mu

from .common import load_gt, load_meta
from .paths import P, add_path_args, apply_path_args
from .submission import by_frame, read_csv


def overlap_stats(pred, gt, stems):
    """pred: {stem: [rle]}; gt: {stem: [[rle] per record]}. Returns a dict of flat arrays / counters."""
    ious, tp_ious = [], []
    gt_deg, pred_deg = [], []
    tp = fp = fn = 0
    for s in stems:
        p = pred.get(s, [])
        for g in gt[s]:
            if not g:  # the notebook builds annotator-images from GT rows: empty records do not exist there
                continue
            if not p:
                gt_deg += [0] * len(g)
                fn += len(g)
                continue
            iou = np.asarray(mu.iou(g, p, [0] * len(p))).reshape(len(g), len(p))  # (n_gt, n_pred)
            ious += iou[iou > 0].tolist()
            hit = iou > 0.5
            tp_ious += iou[hit].tolist()
            tp += int(hit.sum())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
            ov = iou > 0
            gt_deg += ov.sum(1).tolist()
            pred_deg += ov.sum(0).tolist()
    ious, tp_ious = np.array(ious), np.array(tp_ious)
    pq = tp_ious.sum() / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    return dict(pq=float(pq), sq=float(tp_ious.mean()) if tp else 0.0, tp=tp, fp=fp, fn=fn,
                iou=ious, dice=2 * ious / (1 + ious), tp_iou=tp_ious, tp_dice=2 * tp_ious / (1 + tp_ious),
                gt_degree=Counter(gt_deg), pred_degree=Counter(pred_deg))


def summarise(st):
    def dist(x):
        x = np.asarray(x)
        if not len(x):
            return {}
        return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                    p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))
    g, p = st["gt_degree"], st["pred_degree"]
    return dict(PQ=st["pq"], SQ=st["sq"], TP=st["tp"], FP=st["fp"], FN=st["fn"],
                iou_overlapping_pairs=dist(st["iou"]), dice_overlapping_pairs=dist(st["dice"]),
                iou_tp_pairs=dist(st["tp_iou"]), dice_tp_pairs=dist(st["tp_dice"]),
                one_to_n={f"1:{k}": int(v) for k, v in sorted(g.items())},
                n_to_one={f"{k}:1": int(v) for k, v in sorted(p.items())},
                gt_hits=int(sum(v for k, v in g.items() if k > 0)), gt_misses=int(g.get(0, 0)),
                pred_hits=int(sum(v for k, v in p.items() if k > 0)), pred_misses=int(p.get(0, 0)),
                gt_one_to_many=int(sum(v for k, v in g.items() if k > 1)),
                pred_many_to_one=int(sum(v for k, v in p.items() if k > 1)))


def plot_distribution(ax, scores, title, xlabel):
    """Histogram of overlap scores > 0 as in the notebook (50 bins, normalised, mean line)."""
    scores = np.asarray(scores)
    scores = scores[scores > 0]
    counts, edges = np.histogram(scores, bins=np.linspace(0.0, 1.0, 51))
    prob = counts / max(counts.sum(), 1)
    ax.bar((edges[:-1] + edges[1:]) / 2, prob, width=0.02, edgecolor="black", alpha=0.7, color="navy")
    ax.grid(axis="y", linestyle="--")
    if scores.size:
        ax.axvline(scores.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {scores.mean():.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title(title)


def plot_m2n(ax, gdeg, pdeg, title):
    """Split bar chart as in the notebook: GT instances (1:n) on the right, predictions (n:1) on the left."""
    off, bw = 0.5, 0.8
    for d, c in gdeg.items():
        ax.bar(d + off, c, width=bw, color="crimson" if d == 0 else "forestgreen", edgecolor="black", alpha=0.7)
    for d, c in pdeg.items():
        ax.bar(-(d + off), c, width=bw, color="darkorange" if d == 0 else "steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--")
    pos = sorted([d + off for d in gdeg] + [-(d + off) for d in pdeg])
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{int(round(-v - off))}:1" if v < 0 else f"1:{int(round(v - off))}" for v in pos], rotation=45, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Matching case (IoU > 0)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="forestgreen", edgecolor="black", alpha=0.7, label=f"GT hits (1:n, n>0): {sum(v for k, v in gdeg.items() if k > 0)}"),
        Patch(facecolor="crimson", edgecolor="black", alpha=0.7, label=f"GT misses (1:0): {gdeg.get(0, 0)}"),
        Patch(facecolor="darkorange", edgecolor="black", alpha=0.7, label=f"Prediction misses (0:1): {pdeg.get(0, 0)}"),
        Patch(facecolor="steelblue", edgecolor="black", alpha=0.7, label=f"Prediction hits (n:1, n>0): {sum(v for k, v in pdeg.items() if k > 0)}"),
    ], fontsize=7)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--pred", required=True, help="OOF predictions in the submission format")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--name", default="oof")
    a = apply_path_args(ap.parse_args())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = load_gt()
    stems = load_meta()["stems"]
    pred = {s: [r for r in v] for s, v in by_frame(read_csv(a.pred)).items()}
    st = overlap_stats(pred, gt, stems)
    res = summarise(st)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(f"{a.out_dir}/{a.name}_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    for key, label in (("iou", "IoU"), ("dice", "Dice")):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        plot_distribution(ax, st[key], f"{label} distribution (OOF, {a.name})", label)
        fig.tight_layout()
        fig.savefig(f"{a.out_dir}/{a.name}_{key}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    plot_m2n(ax, st["gt_degree"], st["pred_degree"], f"One-to-many and many-to-one (OOF, {a.name})")
    fig.tight_layout()
    fig.savefig(f"{a.out_dir}/{a.name}_m2n.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
