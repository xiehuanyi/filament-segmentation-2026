"""Stage 5: CNN rescorer. Predicts a candidate's annotator-averaged matched IoU q from its refiner crop; the final
keep score is the mean of the LightGBM u_hat and the rescorer output.

Network: ResNet-34 encoder (4 crop channels, or 6 with --xc) -> global average + max pooling, concatenated with
log(crop size) -> MLP (256, dropout 0.2) -> logit. BCE against q, AdamW + one-cycle, D4 augmentation; prediction
is the mean over the 8 dihedral transforms. --xc adds two channels from extra_channels(): the per-pixel std of the
ensemble members' maps (x4) and the mean map of the distilled students.

Training pool: the OOF crops of several candidate sets (--train_tags, e.g. 7 ensembles built from the same dense
runs), targets q looked up in cands/<tag>_oof.pkl. Cross-fitted: the model for fold k never sees fold-k frames;
--fold -1 trains on everything (used for test).

    python -m filseg.rescorer xc      --tag ensB --split oof|test --members M1 ... --students S1 ...
    python -m filseg.rescorer train   --tag ensB --fold k --seed s --epochs 6 --name rsq_s1 [--train_tags ...] [--xc]
    python -m filseg.rescorer predict --tag ensB --split test --ckpt DIR/ensB_rsq_s1_full.pt --name rsq_s1 [--xc]
    python -m filseg.rescorer avg     --tag ensB --names rsq_s0 rsq_s1 rsq_s2 rsq_s3 --out rsqE
Outputs in $FIL_WORK/refine: <tag>_<name>_f<k>.npz / <tag>_<name>_test.npz (idx, q) and checkpoints
<tag>_<name>_f<k>.pt / <tag>_<name>_full.pt. --pool_refine / --pool_cands point at another folder holding the
training pool (default: $FIL_WORK/refine and $FIL_WORK/cands).
"""
import argparse
import glob
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .common import load_meta
from .paths import P, add_path_args, apply_path_args
from .refiner import crop

DEFAULT_TRAIN_TAGS = ["ensB", "ensA", "ens4w", "e6t8", "e137t8", "e13", "e1"]


# ----------------------------------------------------------------------------------------------- extra channels
_XA = None


def _xc_init(a):
    global _XA
    _XA = a


def _run_map(name, stem, split):
    root = os.path.join(P.maps, name)
    if split == "oof":
        files = [f"{root}/oof/{stem}.png"]
    else:
        files = [f for f in sorted(glob.glob(f"{root}/test_f*/{stem}.png")) if os.path.exists(f"{root}/{f.split('/')[-2]}.done")]
    return np.mean([cv2.imread(f, 0).astype(np.float32) for f in files], 0)


def _xc_work(args):
    stem, rows = args
    mem = np.stack([_run_map(r, stem, _XA.split) for r in _XA.members])
    sd = np.clip(mem.std(0) * 4, 0, 255).astype(np.uint8)
    stu = np.clip(np.mean([_run_map(r, stem, _XA.split) for r in _XA.students], 0), 0, 255).astype(np.uint8)
    return [(i, np.stack([crop(sd, cx, cy, L, cv2.INTER_LINEAR), crop(stu, cx, cy, L, cv2.INTER_LINEAR)]))
            for i, cx, cy, L in rows]


def extra_channels(tag, split, members, students, workers=12):
    """Writes refine/<tag>_<split>_xc.npz: Xc (n, 2, 192, 192) uint8 = member-map std x4 and student mean map,
    cropped with the geometry of refine/<tag>_<split>.npz."""
    z = np.load(f"{P.refine}/{tag}_{split}.npz")
    by = {}
    for i, (s, cx, cy, L) in enumerate(zip(z["stem"], z["cx"], z["cy"], z["L"])):
        by.setdefault(str(s), []).append((i, float(cx), float(cy), float(L)))
    Xc = np.zeros((len(z["stem"]), 2, 192, 192), np.uint8)
    a = argparse.Namespace(split=split, members=members, students=students)
    with ProcessPoolExecutor(workers, initializer=_xc_init, initargs=(a,)) as ex:
        for res in ex.map(_xc_work, sorted(by.items()), chunksize=2):
            for i, x in res:
                Xc[i] = x
    np.savez(f"{P.refine}/{tag}_{split}_xc.npz", Xc=Xc)
    print("extra channels", tag, split, Xc.shape, flush=True)


# ----------------------------------------------------------------------------------------------- network
def make_net(xc=False, encoder="resnet34", weights="imagenet"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = smp.encoders.get_encoder(encoder, in_channels=6 if xc else 4, depth=5, weights=weights)
            ch = self.enc.out_channels[-1]
            self.head = nn.Sequential(nn.Linear(ch * 2 + 1, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, x, logL):
            f = self.enc(x)[-1]
            g = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1), F.adaptive_max_pool2d(f, 1).flatten(1), logL], 1)
            return self.head(g)[:, 0]

    return Net()


def _aux(d, idx):
    return np.log(d["L"][idx])[:, None].astype(np.float32)


def _crops(d, refine_dir, tg, split, idx, xc):
    x = d["X"][idx]
    if xc:
        x = np.concatenate([x, np.load(f"{refine_dir}/{tg}_{split}_xc.npz")["Xc"][idx]], 1)
    return x


def _prep(xb, dev):
    import torch
    x = xb.to(dev, non_blocking=True).float() / 255.0
    return ((x - 0.5) / 0.25).contiguous(memory_format=torch.channels_last)


def predict(net, Xp, Lp):
    """Mean sigmoid output over the 8 dihedral transforms; Xp (n, C, 192, 192) uint8, Lp (n, 1) log crop size."""
    import torch
    dev = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Xp), 256):
            x = torch.from_numpy(Xp[b:b + 256])
            l = torch.from_numpy(Lp[b:b + 256]).to(dev)
            acc = 0
            for k in range(4):
                for fl in (False, True):
                    v = torch.rot90(x, k, dims=[2, 3])
                    v = torch.flip(v, dims=[3]) if fl else v
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        acc = acc + torch.sigmoid(net(_prep(v, dev), l).float())
            out.append((acc / 8).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(tag, fold, seed, epochs, name, train_tags, xc=False, bs=128, lr=5e-4, pool_refine=None, pool_cands=None):
    import torch
    import torch.nn.functional as F
    pool_refine = pool_refine or P.refine
    pool_cands = pool_cands or P.cands
    folds = load_meta()["folds"]
    torch.manual_seed(100 + fold + 1000 * seed)
    np.random.seed(100 + fold + 1000 * seed)
    Xs, Qs, Ls = [], [], []
    for tg in train_tags:
        if not os.path.exists(f"{pool_refine}/{tg}_oof.npz") or (xc and not os.path.exists(f"{pool_refine}/{tg}_oof_xc.npz")):
            print("skip (no crops)", tg, flush=True)
            continue
        d = np.load(f"{pool_refine}/{tg}_oof.npz")
        C = pickle.load(open(f"{pool_cands}/{tg.split('_t0')[0]}_oof.pkl", "rb"))
        q = np.array([C[str(s)][int(j)]["q"] for s, j in zip(d["stem"], d["j"])], np.float32)
        fa = np.array([folds[str(s)] for s in d["stem"]])
        keep = np.flatnonzero(fa != fold)
        Xs.append(_crops(d, pool_refine, tg, "oof", keep, xc))
        Qs.append(q[keep])
        Ls.append(_aux(d, keep))
        print(tg, len(keep), flush=True)
    X = np.concatenate(Xs)
    Q = np.concatenate(Qs)
    LL = np.concatenate(Ls)
    del Xs
    print("train crops", X.shape, "mean q", Q.mean(), flush=True)
    dev = torch.device("cuda")
    net = make_net(xc).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    nb = len(X) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb, pct_start=0.1)
    Xt = torch.from_numpy(X)
    Qt = torch.from_numpy(Q)
    Lt = torch.from_numpy(LL)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(X))
        tl = 0
        for b in range(nb):
            idx = torch.from_numpy(np.sort(perm[b * bs:(b + 1) * bs]))
            x = Xt[idx]
            k = int(np.random.randint(4))
            x = torch.rot90(x, k, dims=[2, 3])
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[3])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(_prep(x, dev), Lt[idx].to(dev))
            loss = F.binary_cross_entropy_with_logits(o.float(), Qt[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        print(f"ep {ep} loss {tl / nb:.4f} {time.time() - t0:.0f}s", flush=True)
    part = f"f{fold}" if fold >= 0 else "full"
    os.makedirs(P.refine, exist_ok=True)
    torch.save(net.state_dict(), f"{P.refine}/{tag}_{name}_{part}.pt")
    if fold >= 0:
        predict_split(tag, "oof", net=net, name=name, fold=fold, xc=xc)
    elif os.path.exists(f"{P.refine}/{tag}_test.npz"):
        predict_split(tag, "test", net=net, name=name, xc=xc)
    print("done", time.time() - t0, flush=True)


def predict_split(tag, split, ckpt=None, net=None, name="rs", fold=None, xc=False):
    """Rescorer outputs for the crops of refine/<tag>_<split>.npz (oof: the crops of `fold`)."""
    import torch
    if net is None:
        net = make_net(xc, weights=None)
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        net = net.cuda().to(memory_format=torch.channels_last)
    d = np.load(f"{P.refine}/{tag}_{split}.npz")
    if split == "oof":
        folds = load_meta()["folds"]
        idx = np.flatnonzero(np.array([folds[str(s)] for s in d["stem"]]) == fold)
        out = f"{P.refine}/{tag}_{name}_f{fold}.npz"
    else:
        idx = np.arange(len(d["L"]))
        out = f"{P.refine}/{tag}_{name}_test.npz"
    q = predict(net, _crops(d, P.refine, tag, split, idx, xc), _aux(d, idx))
    np.savez(out, idx=idx, q=q)
    print("rescored", len(idx), "crops ->", os.path.basename(out), flush=True)


def average(tag, names, out, parts=None):
    """Mean of several rescorer outputs (e.g. seeds) -> refine/<tag>_<out>_<part>.npz for every part present."""
    for part in parts or [f"f{k}" for k in range(5)] + ["test"]:
        fs = [f"{P.refine}/{tag}_{t}_{part}.npz" for t in names]
        fs = [f for f in fs if os.path.exists(f)]
        if fs:
            assert len(fs) == len(names), (part, fs)
            ps = [np.load(f) for f in fs]
            assert all((p["idx"] == ps[0]["idx"]).all() for p in ps)
            np.savez(f"{P.refine}/{tag}_{out}_{part}.npz", idx=ps[0]["idx"], q=np.mean([p["q"] for p in ps], 0))


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("cmd", choices=["xc", "train", "predict", "avg"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="oof")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--name", default="rs")
    ap.add_argument("--train_tags", nargs="+", default=DEFAULT_TRAIN_TAGS)
    ap.add_argument("--xc", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--students", nargs="+", default=None)
    ap.add_argument("--pool_refine", default=None)
    ap.add_argument("--pool_cands", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = apply_path_args(ap.parse_args())
    if a.cmd == "xc":
        extra_channels(a.tag, a.split, a.members, a.students, a.workers)
    elif a.cmd == "train":
        train(a.tag, a.fold, a.seed, a.epochs, a.name, a.train_tags, a.xc, pool_refine=a.pool_refine, pool_cands=a.pool_cands)
    elif a.cmd == "predict":
        predict_split(a.tag, a.split, ckpt=a.ckpt, name=a.name, fold=a.fold, xc=a.xc)
    else:
        average(a.tag, a.names, a.out)


if __name__ == "__main__":
    main()
