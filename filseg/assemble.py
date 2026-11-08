"""Stage 6: selection and assembly. Keeps the crops whose score exceeds tau, pastes the refined masks back into
the frame and writes pixel-disjoint instances (OOF: pooled PQ; test: the submission CSV).

score = (1 - w_cnn) * u_hat (LightGBM) + w_cnn * rescorer output (--rescore <name>, refine/<tag>_<name>_*.npz)
Kept candidates are pasted in decreasing score order: the refined crop is warped back to the frame, thresholded
at --thrs, restricted to the connected parts that touch the seed candidate and to pixels not yet claimed by a
higher-scored filament (fallback to the seed if fewer than 10 pixels remain).

    python -m filseg.assemble --tag ensB --split oof  --rescore rsAll --w_cnn 0.5 --tau 0.22 --thrs 0.5 [--out oof.csv]
    python -m filseg.assemble --tag ensB --split test --rescore rsAll --w_cnn 0.5 --tau 0.22 --thrs 0.5 --out sub.csv
"""
import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

from .common import H, W, load_gt, load_meta
from .metric import image_counts, summary
from .paths import P, add_path_args, apply_path_args
from .refiner import S
from .submission import write_csv


def paste(pred, cx, cy, L):
    """Warps a refined crop back into frame coordinates; returns the sub-image and its box (y0, y1, x0, x1)."""
    sc = S / L
    M = np.array([[sc, 0, S / 2 - sc * cx], [0, sc, S / 2 - sc * cy]], np.float32)
    x0 = int(max(cx - L / 2 - 2, 0))
    x1 = int(min(cx + L / 2 + 2, W))
    y0 = int(max(cy - L / 2 - 2, 0))
    y1 = int(min(cy + L / 2 + 2, H))
    M2 = M.copy()
    M2[0, 2] += sc * x0
    M2[1, 2] += sc * y0
    sub = cv2.warpAffine(pred, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sub, (y0, y1, x0, x1)


def build(items, cands, thrs, mode="replace"):
    """items: list of (j, cx, cy, L, score, refined crop). Returns {thr: [rle]} (+ 'seed': the unrefined masks)."""
    items = sorted(items, key=lambda t: -t[4])
    out = {"seed": [cands[j]["rle"] for j, *_ in items]}
    for thr in thrs:
        claim = np.zeros((H, W), np.uint8)
        rl = []
        for j, cx, cy, L, q, pred in items:
            seed = mu.decode(cands[j]["rle"])
            sub, (y0, y1, x0, x1) = paste(pred, cx, cy, L)
            m = np.zeros((H, W), np.uint8)
            m[y0:y1, x0:x1] = sub > int(thr * 255)
            if mode == "union":
                m |= seed
            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            if n > 1:
                keep = np.unique(lab[(seed > 0) & (lab > 0)])
                m = np.isin(lab, keep).astype(np.uint8)
            m &= (1 - claim)
            if m.sum() < 10:
                m = seed & (1 - claim)
            if m.sum() == 0:
                continue
            claim |= m
            rl.append(mu.encode(np.asfortranarray(m)))
        out[thr] = rl
    return out


def scores(tag, split, rescore=None, w_cnn=0.5):
    """Crop table of refine/<tag>_<split>.npz with the blended score in 'qhat'."""
    z = np.load(f"{P.refine}/{tag}_{split}.npz")
    d = {k: z[k] for k in z.files if k not in ("X", "Y")}
    if rescore:
        cq = np.full(len(d["qhat"]), np.nan)
        files = [f"{P.refine}/{tag}_{rescore}_f{k}.npz" for k in range(5)] if split == "oof" else [f"{P.refine}/{tag}_{rescore}_test.npz"]
        for f in files:
            if os.path.exists(f):
                p = np.load(f)
                cq[p["idx"]] = p["q"]
        if split == "test":
            assert not np.isnan(cq).any(), "rescorer predictions missing"
        d["has_cnn"] = ~np.isnan(cq)
        d["qhat"] = (1 - w_cnn) * d["qhat"] + w_cnn * cq
    return d


def refined_crops(tag, split, n):
    pred = np.zeros((n, S, S), np.uint8)
    have = np.zeros(n, bool)
    files = [f"{P.refine}/{tag}_pred_f{k}.npz" for k in range(5)] if split == "oof" else [f"{P.refine}/{tag}_pred_test.npz"]
    for f in files:
        if os.path.exists(f):
            p = np.load(f)
            pred[p["idx"]] = p["pred"]
            have[p["idx"]] = True
    return pred, have


_A = _GT = _C = None


def _init(a):
    global _A, _GT, _C
    _A = a
    _GT = load_gt() if a.split == "oof" else None
    _C = pickle.load(open(f"{P.cands}/{a.tag}_{a.split}.pkl", "rb"))


def _work(args):
    stem, items = args
    res = build(items, _C[stem], _A.thrs, _A.mode)
    if _A.split == "oof":
        out = {k: image_counts(_GT[stem], v) for k, v in res.items()}
        if getattr(_A, "keep_rles", False):
            out["rles"] = [r["counts"].decode("ascii") for r in res[_A.thrs[0]]]
        return stem, out
    return stem, {k: [r["counts"].decode("ascii") for r in v] for k, v in res.items()}


def selected(tag, split, rescore=None, w_cnn=0.5, tau=0.22):
    """{stem: [(j, cx, cy, L, score, refined crop), ...]} of the kept candidates, and the list of stems."""
    d = scores(tag, split, rescore, w_cnn)
    pred, have = refined_crops(tag, split, len(d["stem"]))
    if rescore and split == "oof":
        have &= d["has_cnn"]
    by = {}
    for i in range(len(d["stem"])):
        if have[i] and d["qhat"][i] > tau:
            by.setdefault(str(d["stem"][i]), []).append((int(d["j"][i]), float(d["cx"][i]), float(d["cy"][i]),
                                                        float(d["L"][i]), float(d["qhat"][i]), pred[i]))
    stems = sorted({str(s) for s, h in zip(d["stem"], have) if h})
    return by, stems


def run(tag, split, rescore=None, w_cnn=0.5, tau=0.22, thrs=(0.5,), mode="replace", workers=12, out=None, quiet=False,
        per_frame=None):
    """OOF: returns {thr or 'seed': pooled count vector} over the evaluated folds (and fills `per_frame`
    {stem: {thr: counts}} when a dict is given; frames without any crop are added with all their instances as FN).
    With `out` the OOF predictions (first threshold) are also written as a CSV in the submission format.
    Test: writes the CSV `out` and returns its rows."""
    by, stems = selected(tag, split, rescore, w_cnn, tau)
    a = argparse.Namespace(tag=tag, split=split, thrs=list(thrs), mode=mode, keep_rles=bool(out))
    tot, rows = {}, []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(a,)) as ex:
        for stem, res in ex.map(_work, [(s, by.get(s, [])) for s in stems], chunksize=2):
            if split == "oof":
                for k, cnt in enumerate(res.pop("rles", [])):
                    rows.append((f"{stem}_{k + 1}", cnt))
                for k, v in res.items():
                    tot[k] = tot.get(k, 0) + v
                if per_frame is not None:
                    per_frame[stem] = res
            else:
                for k, cnt in enumerate(res[a.thrs[0]]):
                    rows.append((f"{stem}_{k + 1}", cnt))
    if split == "oof":
        # frames of the evaluated folds without any crop still count their ground-truth instances as FN
        meta = load_meta()
        gt = load_gt()
        ev = {meta["folds"][x] for x in stems}
        sset = set(stems)
        missing = [s for s in meta["stems"] if meta["folds"][s] in ev and s not in sset]
        extra = sum(len(g) for s in missing for g in gt[s])
        tot = {k: v + np.array([0, 0, 0, extra]) for k, v in tot.items()}
        if per_frame is not None:
            for s in missing:
                n = sum(len(g) for g in gt[s])
                per_frame[s] = {k: np.array([0.0, 0, 0, n]) for k in tot}
        if not quiet:
            for k, v in tot.items():
                print(k, summary(v), flush=True)
        if out:
            write_csv(rows, out)
        return tot
    write_csv(rows, out)
    return rows


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="oof")
    ap.add_argument("--tau", type=float, default=0.22)
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.5])
    ap.add_argument("--mode", default="replace")
    ap.add_argument("--rescore", default=None)
    ap.add_argument("--w_cnn", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = apply_path_args(ap.parse_args())
    run(a.tag, a.split, a.rescore, a.w_cnn, a.tau, a.thrs, a.mode, a.workers, a.out)


if __name__ == "__main__":
    main()
