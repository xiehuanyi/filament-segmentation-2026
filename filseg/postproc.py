"""Simple probability map -> pixel-disjoint instance RLEs (connected components); used for quick validation
during dense training. The final pipelines use candidates.py + scorer.py + refiner.py instead."""
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

H = W = 2048
S8 = np.ones((3, 3), bool)


def enc(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def label(p, t_lo):
    """p: float [0,1] or uint8 map. Returns (label image, number of components), 8-connectivity."""
    if p.dtype == np.uint8:
        return ndi.label(p > int(round(t_lo * 255)), structure=S8)
    return ndi.label(p > t_lo, structure=S8)


def comp_stats(p, lab, n):
    pf = p.astype(np.float32) / (255.0 if p.dtype == np.uint8 else 1.0)
    lf = lab.ravel()
    nz = np.flatnonzero(lf)
    lv = lf[nz]
    pv = pf.ravel()[nz]
    area = np.bincount(lv, minlength=n + 1)[1:].astype(np.int64)
    psum = np.bincount(lv, weights=pv, minlength=n + 1)[1:]
    pmax = np.zeros(n + 1, np.float32)
    np.maximum.at(pmax, lv, pv)
    return area, psum / np.maximum(area, 1), pmax[1:]


def components_rles(p, t_lo=0.5, min_area=0, t_hi=0.0, min_meanp=0.0, max_n=500):
    lab, n = label(p, t_lo)
    if n == 0:
        return []
    area, meanp, pmax = comp_stats(p, lab, n)
    keep = np.flatnonzero((area >= min_area) & (pmax > t_hi) & (meanp >= min_meanp)) + 1
    if len(keep) > max_n:
        keep = keep[np.argsort(-area[keep - 1])[:max_n]]
    objs = ndi.find_objects(lab)
    out = []
    for c in keep:
        sl = objs[c - 1]
        m = np.zeros((H, W), np.uint8)
        m[sl] = lab[sl] == c
        out.append(enc(m))
    return out
