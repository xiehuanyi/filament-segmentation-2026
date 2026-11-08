"""Stage 3: LightGBM hurdle scorer u_hat = P(an annotator record matches the candidate) x E[IoU | match].

    python -m filseg.scorer --tag ensB          # needs cands/<tag>_oof.pkl (with ground-truth s, m, q, hits)

Two boosters on the candidate descriptors (candidates.FEATS):
  lgb0  cross-entropy on the fraction of annotator records that match the candidate (hits / m), weight m
  lgb1  regression of the mean matched IoU on candidates with >= 1 match, weight hits
u_hat = lgb0 * lgb1 estimates q = s / m. Cross-fitted by fold (fold k is scored by boosters trained on the
other folds), so the OOF scores are honest inputs to the later stages. LightGBM is deterministic here (fixed
seed and 8 threads): refitting on the same table reproduces the released boosters bit for bit.

Writes $FIL_WORK/pp/<tag>_oofq.npz (cross-fitted qhat per candidate row, in stem-sorted order),
pp/<tag>_lgb0.txt + _lgb1.txt (boosters fit on all folds, used for test) and pp/<tag>.json
(PQ of the single-level keep rule 'level sel_t, u_hat > tau' on OOF, for information).
"""
import argparse
import json
import os
import pickle

import lightgbm as lgb
import numpy as np

from .candidates import FEATS, feature_matrix
from .common import load_gt, load_meta
from .paths import P, add_path_args, apply_path_args

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
ROUNDS = 500


def pq_from(sel_s, sel_m, n_gt):
    return sel_s / (0.5 * n_gt + 0.5 * sel_m)


def table(C, folds):
    """Flattens a candidate table (stems sorted, candidates in order) into features and targets."""
    rows = [(s, j, folds[s], c) for s in sorted(C) for j, c in enumerate(C[s])]
    F = feature_matrix([c for *_, c in rows])
    T = {k: np.array([c[k] for *_, c in rows], float) for k in ("q", "s", "m", "hits")}
    return rows, F, T


def fit(F, T, idx, seed=0):
    """Hurdle boosters on rows idx. Returns [lgb0 (match fraction), lgb1 (IoU given a match)]."""
    hits, mv, sv = T["hits"], T["m"], T["s"]
    hfrac = np.clip(hits / mv, 0, 1)
    miou = np.where(hits > 0, sv / np.maximum(hits, 1), 0.0)
    pos = idx[hits[idx] > 0]
    mh = lgb.train(dict(PARAMS, objective="cross_entropy", seed=seed), lgb.Dataset(F[idx], hfrac[idx], weight=mv[idx]),
                   num_boost_round=ROUNDS)
    mm = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(F[pos], miou[pos], weight=hits[pos]), num_boost_round=ROUNDS)
    return [mh, mm]


def predict(models, F):
    """u_hat for a feature matrix (product of the two boosters)."""
    return np.prod([m.predict(F) for m in models], axis=0)


def load(scorer_dir, tag):
    cfg = json.load(open(os.path.join(scorer_dir, f"{tag}.json")))
    return [lgb.Booster(model_file=os.path.join(scorer_dir, f"{tag}_lgb{i}.txt")) for i in range(cfg["n_models"])]


def fit_tag(tag, sel_t=0.3):
    """Cross-fitted OOF scores + full-data boosters for cands/<tag>_oof.pkl (see the module docstring)."""
    a = argparse.Namespace(tag=tag, sel_t=sel_t)
    meta = load_meta()
    gt = load_gt()
    folds = meta["folds"]
    C = pickle.load(open(f"{P.cands}/{a.tag}_oof.pkl", "rb"))
    rows, F, T = table(C, folds)
    fold_arr = np.array([r[2] for r in rows])
    n_gt = sum(len(g) for s in C for g in gt[s])
    print(f"{a.tag}: N_gt {n_gt}, candidates {len(rows)}", flush=True)
    qhat = np.zeros(len(rows))
    for f in sorted(set(fold_arr.tolist())):
        tr = np.flatnonzero(fold_arr != f)
        qhat[fold_arr == f] = predict(fit(F, T, tr), F[fold_arr == f])
    t = F[:, FEATS.index("t")]
    os.makedirs(P.pp, exist_ok=True)
    np.savez(f"{P.pp}/{a.tag}_oofq.npz", qhat=qhat, q=T["q"], m=T["m"], t=t, area=F[:, FEATS.index("area")],
             meanp=F[:, FEATS.index("meanp")], fold=fold_arr, img=np.array([r[0] for r in rows]))
    res = []
    for tau in np.arange(0.10, 0.36, 0.02):
        k = (np.abs(t - a.sel_t) < 1e-6) & (qhat > tau)
        res.append((pq_from(T["s"][k].sum(), T["m"][k].sum(), n_gt), float(tau)))
    best = max(res)
    print(f"keep rule 'level {a.sel_t}, u_hat > tau' on OOF: best PQ {best[0]:.4f} at tau {best[1]:.2f}", flush=True)
    models = fit(F, T, np.arange(len(rows)))
    for i, m in enumerate(models):
        m.save_model(f"{P.pp}/{a.tag}_lgb{i}.txt")
    json.dump({"model": "hurdle", "n_models": 2, "n_seeds": 1, "sel_t": a.sel_t, "sel_t_tau": best[1], "sel_t_pq": best[0]},
              open(f"{P.pp}/{a.tag}.json", "w"), indent=1)
    print("saved", f"{P.pp}/{a.tag}_lgb{{0,1}}.txt", flush=True)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sel_t", type=float, default=0.3)
    a = apply_path_args(ap.parse_args())
    fit_tag(a.tag, a.sel_t)


if __name__ == "__main__":
    main()
