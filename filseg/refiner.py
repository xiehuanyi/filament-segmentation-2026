"""Stage 4: crop refiner. A small U-Net (ResNet-34 encoder, 4 input channels) re-draws the mask of each kept
candidate in a 192 x 192 crop around it.

Crops (one per level-sel_t candidate with u_hat > min_q, square box 1.5 x the longer bbox side + 32 px, >= 96 px):
  ch0 frame (z-normalised, x32 + 128)   ch1 ensemble probability map   ch2 the candidate (seed) mask
  ch3 the other candidates with u_hat > sel_tau
Target (OOF only): mean over annotator records of the record's best-matching instance mask (IoU > 0.2).

    python -m filseg.refiner crops   --tag ensB --members ... --split oof          # uses pp/<tag>_oofq.npz
    python -m filseg.refiner crops   --tag ensB --members ... --split test --scorer_dir DIR
    python -m filseg.refiner train   --tag ensB --fold k   (k = 0..4: trained without fold k, predicts fold k; -1: all)
    python -m filseg.refiner predict --tag ensB --split test --ckpt DIR/ensB_refiner.pt [--amp fp16]

Training: BCE + soft Dice, AdamW + one-cycle, 20 epochs, D4 and brightness augmentation, cross-fitted by fold.
Prediction: mean over the 4 rotations, stored as uint8 (p * 255).
Files in $FIL_WORK/refine: <tag>_<split>.npz (crops, geometry, qhat), <tag>_pred_f<k>.npz / <tag>_pred_test.npz,
<tag>_refiner_f<k>.pt / <tag>_refiner.pt.
"""
import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mu

from .candidates import feature_matrix
from .common import H, W, list_stems, load_gt, load_img, load_meta, normalize
from .paths import P, add_path_args, apply_path_args

S = 192


# ----------------------------------------------------------------------------------------------- crops
def crop_params(rle):
    x, y, w, h = mu.toBbox(rle)
    L = max(w, h) * 1.5 + 32
    L = max(L, 96.0)
    return x + w / 2, y + h / 2, L


def crop(arr, cx, cy, L, interp):
    sc = S / L
    M = np.array([[sc, 0, S / 2 - sc * cx], [0, sc, S / 2 - sc * cy]], np.float32)
    return cv2.warpAffine(arr, M, (S, S), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_prob_map(stem, members, split):
    """Ensemble map as used for the crops: per member the float mean of its fold maps, weighted mean, uint8."""
    acc = None
    wsum = 0.0
    for spec in members:
        r, w = (spec.split(":") + ["1"])[:2]
        w = float(w)
        root = os.path.join(P.maps, r)
        if split == "test":
            files = [f for f in sorted(glob.glob(f"{root}/test_f*/{stem}.png"))
                     if os.path.exists(f"{root}/{f.split('/')[-2]}.done")] or sorted(glob.glob(f"{root}/test_full/{stem}.png"))
        else:
            files = [f"{root}/oof/{stem}.png"]
        assert files, (r, stem)
        m = np.mean([cv2.imread(f, 0).astype(np.float32) for f in files], 0) * w
        acc = m if acc is None else acc + m
        wsum += w
    return np.clip(np.round(acc / wsum), 0, 255).astype(np.uint8)


_A = _GT = None


def _init(a):
    global _A, _GT
    _A = a
    _GT = load_gt() if a.split == "oof" else None


def _crops_work(args):
    stem, cands, keep_idx, sel_idx, qh = args
    test = _A.split == "test"
    img = normalize(load_img(stem, test=test))
    img8 = np.clip(img * 32 + 128, 0, 255).astype(np.uint8)
    prob = crop_prob_map(stem, _A.members, _A.split)
    masks = {j: mu.decode(cands[j]["rle"]) for j in keep_idx}
    others_all = np.zeros((H, W), np.uint8)
    for j in sel_idx:
        others_all |= masks[j] if j in masks else mu.decode(cands[j]["rle"])
    gts = [mu.decode(g) if g else None for g in (_GT[stem] if not test else [])]  # (H, W, n) per record
    X, Y, info = [], [], []
    for j in keep_idx:
        m = masks[j]
        cx, cy, L = crop_params(cands[j]["rle"])
        oth = others_all & (1 - m)
        x = np.stack([crop(img8, cx, cy, L, cv2.INTER_LINEAR), crop(prob, cx, cy, L, cv2.INTER_LINEAR),
                      crop(m * 255, cx, cy, L, cv2.INTER_LINEAR), crop(oth * 255, cx, cy, L, cv2.INTER_LINEAR)])
        y = np.zeros((S, S), np.float32)
        cnt = 0
        if not test:
            for g, gm in zip(_GT[stem], gts):
                if gm is None:
                    continue
                iou = np.asarray(mu.iou([cands[j]["rle"]], g, [0] * len(g)))[0]
                k = int(iou.argmax())
                if iou[k] > 0.2:
                    y += crop(gm[:, :, k] * 255, cx, cy, L, cv2.INTER_LINEAR).astype(np.float32)
                    cnt += 1
        if cnt:
            y /= cnt
        X.append(x)
        Y.append(np.round(y).astype(np.uint8))
        info.append((j, cx, cy, L, cnt, len(_GT[stem]) if not test else 0, float(qh[j])))
    return stem, np.array(X, np.uint8).reshape(-1, 4, S, S), np.array(Y, np.uint8).reshape(-1, S, S), info


def make_crops(tag, members, split, scorer_dir=None, sel_t=0.3, min_q=0.08, sel_tau=0.18, workers=12, out=None):
    """Crops of the level-sel_t candidates of cands/<tag>_<split>.pkl. u_hat: cross-fitted OOF scores
    (pp/<tag>_oofq.npz) or, for test, the boosters in scorer_dir."""
    from . import scorer
    C = pickle.load(open(f"{P.cands}/{tag}_{split}.pkl", "rb"))
    if split == "oof":
        d = np.load(f"{P.pp}/{tag}_oofq.npz")
        stems = sorted(C)
        off, i = {}, 0
        for s in stems:
            off[s] = i
            i += len(C[s])
        qall = {s: d["qhat"][off[s]:off[s] + len(C[s])] for s in stems}
    else:
        ms = scorer.load(scorer_dir or P.pp, tag)
        stems = list_stems(P.images)
        qall = {}
        for s in stems:
            cs = C.get(s, [])
            qall[s] = scorer.predict(ms, feature_matrix(cs)) if cs else np.zeros(0)
    jobs = []
    for s in stems:
        cs = C.get(s, [])
        qh = qall[s]
        lev = [j for j, c in enumerate(cs) if abs(c["t"] - sel_t) < 1e-6]
        keep = [j for j in lev if qh[j] > min_q]
        sel = [j for j in lev if qh[j] > sel_tau]
        if keep:
            jobs.append((s, cs, keep, sel, qh))
    a = argparse.Namespace(split=split, members=members)
    Xs, Ys, infos = [], [], []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(a,)) as ex:
        for stem, X, Y, info in ex.map(_crops_work, jobs, chunksize=2):
            Xs.append(X)
            Ys.append(Y)
            infos += [(stem,) + t for t in info]
    os.makedirs(P.refine, exist_ok=True)
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    col = lambda k: np.array([t[k] for t in infos])  # noqa: E731
    np.savez(f"{P.refine}/{out or tag}_{split}.npz", X=X, Y=Y, stem=col(0), j=col(1), cx=col(2), cy=col(3), L=col(4),
             cnt=col(5), nsets=col(6), qhat=col(7))
    print(f"crops {out or tag} {split}: {X.shape}, with target {int((col(5) > 0).sum())}", flush=True)


# ----------------------------------------------------------------------------------------------- network
def _net(encoder="resnet34", weights=None):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=4, classes=1)


def _prep(xb, dev):
    import torch
    x = xb.to(dev, non_blocking=True).float() / 255.0
    return ((x - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)


def predict(net, Xa, dt):
    """Refined probability crops (uint8) for crops Xa (n, 4, S, S); mean over the 4 rotations."""
    import torch
    dev = next(net.parameters()).device
    net.eval()
    out = np.zeros((len(Xa), Xa.shape[2], Xa.shape[3]), np.uint8)
    with torch.no_grad():
        for b in range(0, len(Xa), 128):
            x = _prep(torch.from_numpy(Xa[b:b + 128]), dev)
            acc = 0
            for k in range(4):
                v = torch.rot90(x, k, dims=[2, 3])
                with torch.autocast("cuda", dtype=dt):
                    p = torch.sigmoid(net(v.contiguous(memory_format=torch.channels_last)).float())
                acc = acc + torch.rot90(p, -k, dims=[2, 3])
            out[b:b + 128] = (acc[:, 0] / 4 * 255).round().clamp(0, 255).byte().cpu().numpy()
    return out


def load_net(ckpt, encoder="resnet34"):
    import torch
    net = _net(encoder)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return net.cuda().to(memory_format=torch.channels_last).eval()


def train(tag, fold, epochs=20, bs=64, lr=1e-3, encoder="resnet34", amp="bf16", test_tag=None):
    import torch
    import torch.nn.functional as F
    dt = torch.float16 if amp == "fp16" else torch.bfloat16
    torch.manual_seed(fold + 10)
    d = np.load(f"{P.refine}/{tag}_oof.npz")
    folds = load_meta()["folds"]
    fa = np.array([folds[s] for s in d["stem"]])
    X, Y, cnt = d["X"], d["Y"], d["cnt"]
    tr = np.flatnonzero((fa != fold) & (cnt > 0))
    dev = torch.device("cuda")
    net = _net(encoder, "imagenet").to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    def aug(x, y):
        k = int(np.random.randint(4))
        fl = np.random.rand() < 0.5
        x = torch.rot90(x, k, dims=[2, 3])
        y = torch.rot90(y, k, dims=[1, 2])
        if fl:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2])
        return x, y

    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        tl = 0
        for b in range(len(tr) // bs):
            idx = torch.from_numpy(np.sort(perm[b * bs:(b + 1) * bs]))
            x, y = aug(Xt[idx], Yt[idx])
            x = _prep(x, dev)
            y = y.to(dev).float()[:, None] / 255.0
            x[:, 0] = x[:, 0] * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-0.2, 0.2))
            with torch.autocast("cuda", dtype=dt):
                o = net(x)
            o = o.float()
            p = torch.sigmoid(o)
            loss = F.binary_cross_entropy_with_logits(o, y) + \
                (1 - (2 * (p * y).sum((1, 2, 3)) + 1) / ((p + y).sum((1, 2, 3)) + 1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        print(f"ep {ep} loss {tl / max(1, len(tr) // bs):.4f} {time.time() - t0:.0f}s", flush=True)
    if fold >= 0:
        torch.save(net.state_dict(), f"{P.refine}/{tag}_refiner_f{fold}.pt")
        te = np.flatnonzero(fa == fold)
        np.savez(f"{P.refine}/{tag}_pred_f{fold}.npz", idx=te, pred=predict(net, X[te], dt))
    else:
        torch.save(net.state_dict(), f"{P.refine}/{tag}_refiner.pt")
        dt_ = np.load(f"{P.refine}/{test_tag or tag}_test.npz")
        np.savez(f"{P.refine}/{test_tag or tag}_pred_test.npz", idx=np.arange(len(dt_["X"])), pred=predict(net, dt_["X"], dt))
    print("done", time.time() - t0, flush=True)


def predict_split(tag, split, ckpt, amp="bf16", fold=None, encoder="resnet34"):
    """Refined crops of <tag>_<split>.npz from a saved refiner (test: all crops; oof: the crops of `fold`)."""
    import torch
    dt = torch.float16 if amp == "fp16" else torch.bfloat16
    d = np.load(f"{P.refine}/{tag}_{split}.npz")
    net = load_net(ckpt, encoder)
    if split == "test":
        idx = np.arange(len(d["X"]))
        name = f"{tag}_pred_test.npz"
    else:
        folds = load_meta()["folds"]
        idx = np.flatnonzero(np.array([folds[str(s)] for s in d["stem"]]) == fold)
        name = f"{tag}_pred_f{fold}.npz"
    np.savez(f"{P.refine}/{name}", idx=idx, pred=predict(net, d["X"][idx], dt))
    print("refined", len(idx), "crops ->", name, flush=True)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("cmd", choices=["crops", "train", "predict"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--split", default="oof")
    ap.add_argument("--scorer_dir", default=None)
    ap.add_argument("--sel_t", type=float, default=0.3)
    ap.add_argument("--min_q", type=float, default=0.08)
    ap.add_argument("--sel_tau", type=float, default=0.18)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = apply_path_args(ap.parse_args())
    if a.cmd == "crops":
        make_crops(a.tag, a.members, a.split, a.scorer_dir, a.sel_t, a.min_q, a.sel_tau, a.workers)
    elif a.cmd == "train":
        train(a.tag, a.fold, a.epochs, amp=a.amp)
    else:
        predict_split(a.tag, a.split, a.ckpt, a.amp, a.fold)


if __name__ == "__main__":
    sys.exit(main())
