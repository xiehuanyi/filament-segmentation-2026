"""Stage 2: candidate filaments = connected components of the ensemble probability map at nested thresholds.

    python -m filseg.candidates --tag ensB --split oof|test --members e6_effv2s_c1536:2 ... e1_tta8 \
        [--thresholds 0.2 0.3 0.4 0.5 0.6 0.7 0.8]

Ensemble map: for each member map, the mean over its fold models (test) or its OOF map, then the weighted mean
over members, rounded to uint8. Components (8-connectivity, area >= 30 px, at most 300 per threshold) at
thresholds t1 < t2 < ... are nested; each candidate records its parent at the next lower threshold and shape /
probability / intensity descriptors (FEATS) used by the scorer.

Why candidates are scored one at a time: for pixel-disjoint predictions the pooled PQ decomposes. A candidate j
on an image with m annotator records adds s_j = sum_g IoU_jg [IoU_jg > 0.5] to the numerator and m/2 to the
denominator (the FN half is the constant 0.5 N_gt), so
    PQ(K) = sum_K s_j / (0.5 N_gt + 0.5 sum_K m_j)
and the optimal rule keeps candidate j iff q_j = s_j / m_j > PQ*/2. q_j (the annotator-averaged matched IoU) is
the target of the scorers. OOF tables get s, m, q, hits from the ground truth.

Writes $FIL_WORK/cands/<tag>_<split>.pkl: {stem: [candidate dict with 'rle' and features]}.
"""
import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

from .common import H, W, list_stems, load_gt, load_img, load_meta, normalize
from .paths import P, add_path_args, apply_path_args

S8 = np.ones((3, 3), bool)

FEATS = ["t", "area", "larea", "meanp", "maxp", "p50", "p90", "stdp", "core5", "core7", "core9",
         "bh", "bw", "extent", "elong", "rdist", "imean", "ring_contrast", "n_comp_img", "rank_meanp",
         "child_n", "parent_area_ratio", "solidity", "d1_ar", "d2_ar", "d3_ar", "d1_n", "d2_n", "persist",
         "par_meanp", "par_n_children"]


def enc(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def extract(p_u8, img_norm, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7), min_area=30, max_per_t=300):
    """Candidate dicts (with 'rle' and descriptors) of a uint8 probability map; img_norm = z-normalised frame."""
    pf = p_u8.astype(np.float32) / 255.0
    cands = []
    prev_lab = None
    prev_ids = None
    cy0 = cx0 = (H - 1) / 2.0
    for ti, t in enumerate(thresholds):
        lab, n = ndi.label(p_u8 > int(round(t * 255)), structure=S8)
        if n == 0:
            prev_lab, prev_ids = lab, {}
            continue
        objs = ndi.find_objects(lab)
        lf = lab.ravel()
        nz = np.flatnonzero(lf)
        lv = lf[nz]
        area = np.bincount(lv, minlength=n + 1)
        order = [c for c in range(1, n + 1) if area[c] >= min_area]
        if len(order) > max_per_t:
            order = sorted(order, key=lambda c: -area[c])[:max_per_t]
        ids = {}
        rows_t = []
        for c in order:
            sl = objs[c - 1]
            y0, y1 = max(sl[0].start - 6, 0), min(sl[0].stop + 6, H)  # padded box for the ring
            x0, x1 = max(sl[1].start - 6, 0), min(sl[1].stop + 6, W)
            lsub = lab[y0:y1, x0:x1]
            m = (lsub == c)
            ps = pf[y0:y1, x0:x1][m]
            isub = img_norm[y0:y1, x0:x1]
            ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
            ys, xs = np.nonzero(m)
            yc, xc = ys.mean(), xs.mean()
            if len(ys) > 2:
                cov = np.cov(np.stack([ys, xs]).astype(np.float64))
                ev = np.sort(np.linalg.eigvalsh(cov))
                elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
            else:
                elong = 1.0
            hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
            harea = max(cv2.contourArea(hull), 1.0)
            full = np.zeros((H, W), np.uint8)
            full[y0:y1, x0:x1] = m
            parent = -1
            par_ratio = 1.0
            if prev_lab is not None:
                pl = prev_lab[ys[0] + y0, xs[0] + x0]
                parent = prev_ids.get(int(pl), -1)
                if parent >= 0:
                    par_ratio = len(ys) / cands[parent]["area"]
            d = dict(t=t, area=int(len(ys)), larea=float(np.log(len(ys))), meanp=float(ps.mean()),
                     maxp=float(ps.max()), p50=float(np.median(ps)), p90=float(np.percentile(ps, 90)),
                     stdp=float(ps.std()), core5=float((ps > 0.5).mean()), core7=float((ps > 0.7).mean()),
                     core9=float((ps > 0.9).mean()), bh=int(sl[0].stop - sl[0].start),
                     bw=int(sl[1].stop - sl[1].start),
                     extent=float(len(ys) / ((sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start))),
                     elong=elong, rdist=float(np.hypot(yc + y0 - cy0, xc + x0 - cx0) / 1024.0),
                     imean=float(isub[m].mean()),
                     ring_contrast=float(isub[ring].mean() - isub[m].mean()) if ring.any() else 0.0,
                     solidity=float(len(ys) / harea),
                     parent=parent, parent_area_ratio=par_ratio, rle=enc(full))
            ids[c] = len(cands)
            cands.append(d)
            rows_t.append(len(cands) - 1)
        if rows_t:  # per-threshold image-level features
            mp = np.array([cands[j]["meanp"] for j in rows_t])
            rk = np.argsort(np.argsort(-mp))
            for r, j in zip(rk, rows_t):
                cands[j]["n_comp_img"] = len(rows_t)
                cands[j]["rank_meanp"] = int(r)
        prev_lab, prev_ids = lab, ids
    for c in cands:
        c["child_n"] = 0
    for c in cands:
        if c["parent"] >= 0:
            cands[c["parent"]]["child_n"] += 1
    # descendant aggregates by relative level (1, 2, 3 levels above) and persistence
    tl = {t: i for i, t in enumerate(thresholds)}
    agg = [dict() for _ in cands]  # level index -> [area, count]
    for j in reversed(range(len(cands))):
        c = cands[j]
        lj = tl[c["t"]]
        c["persist"] = float(max([thresholds[l] for l in agg[j]] + [c["t"]]))
        for d in (1, 2, 3):
            ar, n = agg[j].get(lj + d, [0, 0])
            c[f"d{d}_ar"] = ar / c["area"]
            if d <= 2:
                c[f"d{d}_n"] = n
        p = c["parent"]
        if p >= 0:
            e = agg[p].setdefault(lj, [0, 0])
            e[0] += c["area"]
            e[1] += 1
            for l, (ar, n) in agg[j].items():
                e = agg[p].setdefault(l, [0, 0])
                e[0] += ar
                e[1] += n
    for c in cands:
        p = c["parent"]
        c["par_meanp"] = cands[p]["meanp"] if p >= 0 else 0.0
        c["par_n_children"] = cands[p]["child_n"] if p >= 0 else 0
    return cands


def score_cands(cands, gt_sets):
    """Adds s (sum of hit IoUs over annotator records), m (number of records), q = s/m and hits to each candidate."""
    if not cands:
        return
    rles = [c["rle"] for c in cands]
    s = np.zeros(len(cands))
    hits = np.zeros(len(cands))
    for g in gt_sets:
        if not g:
            continue
        iou = np.asarray(mu.iou(rles, g, [0] * len(g)))  # (n_c, n_g)
        h = iou > 0.5
        s += (iou * h).sum(1)
        hits += h.sum(1)
    for j, c in enumerate(cands):
        c["s"] = float(s[j])
        c["m"] = len(gt_sets)
        c["q"] = float(s[j] / len(gt_sets))
        c["hits"] = int(hits[j])


def feature_matrix(cs):
    return np.array([[c.get(f, 0.0) for f in FEATS] for c in cs], np.float32)


def member_files(name, stem, split):
    """Map files of one member for one frame: its OOF map, or one test map per finished fold model."""
    root = os.path.join(P.maps, name)
    if split == "oof":
        return [f"{root}/oof/{stem}.png"]
    files = [f for f in sorted(glob.glob(f"{root}/test_f*/{stem}.png"))
             if os.path.exists(f"{root}/{f.split('/')[-2]}.done")]
    return files or sorted(glob.glob(f"{root}/test_full/{stem}.png"))


def parse_member(spec):
    name, w = (spec.split(":") + ["1"])[:2]
    return name, float(w)


def ensemble_map(stem, members, split):
    """Weighted mean of the members' maps (each member = mean of its fold-model maps), rounded to uint8."""
    acc = None
    wsum = 0
    for spec in members:
        name, w = parse_member(spec)
        files = member_files(name, stem, split)
        assert files, (name, stem)
        for f in files:
            m = cv2.imread(f, 0)
            assert m is not None, f
            acc = m.astype(np.float32) * (w / len(files)) if acc is None else acc + m.astype(np.float32) * (w / len(files))
        wsum += w
    return np.clip(np.round(acc / wsum), 0, 255).astype(np.uint8)


_A = None
_GT = None


def _init(a):
    global _A, _GT
    _A = a
    _GT = load_gt() if a.split == "oof" else None


def _work(stem):
    p = ensemble_map(stem, _A.members, _A.split)
    img = normalize(load_img(stem, test=(_A.split == "test")))
    c = extract(p, img, thresholds=tuple(_A.thresholds))
    if _A.split == "oof":
        score_cands(c, _GT[stem])
    return stem, c


def build(tag, members, split, thresholds, workers=8, stems=None):
    a = argparse.Namespace(tag=tag, members=members, split=split, thresholds=thresholds)
    if stems is None:
        stems = load_meta()["stems"] if split == "oof" else list_stems(P.images)
    os.makedirs(P.cands, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(a,)) as ex:
        for stem, c in ex.map(_work, stems, chunksize=2):
            out[stem] = c
    pickle.dump(out, open(f"{P.cands}/{tag}_{split}.pkl", "wb"))
    print(f"candidates {tag} {split}: {len(out)} images, {sum(len(v) for v in out.values())} candidates", flush=True)
    return out


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--members", nargs="+", required=True, help="map names, optionally name:weight")
    ap.add_argument("--split", default="oof", choices=["oof", "test"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--workers", type=int, default=8)
    a = apply_path_args(ap.parse_args())
    build(a.tag, a.members, a.split, a.thresholds, a.workers)


if __name__ == "__main__":
    main()
