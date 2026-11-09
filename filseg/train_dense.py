"""Stage 1: train one fold of a dense model, then write its OOF and test probability maps.

    python -m filseg.train_dense --name e6_effv2s_c1536 --fold 0 --encoder tu-tf_efficientnetv2_s.in21k_ft_in1k \
        --crop 1536 --bs 4 --crops_per_img 4 --tta 8

Training: random crops of the native 2048 px frames (foreground-biased, +-15% scale, D4, brightness/contrast,
blur and noise), soft target = mean of the annotators' masks (cache/targets), loss BCE + soft Dice, AdamW with
warm-up + cosine decay, EMA of the weights (the EMA model is the one saved and used). ``--fold -1`` trains on
all training frames. ``--pseudo DIR`` adds unlabelled frames DIR/images/<stem>.jpg with teacher maps
DIR/targets_<f k>/<stem>.png (see teacher.py); the step count grows with the number of added frames.

Outputs:
  $FIL_WORK/runs/<name>/model_<tag>.pt, args_<tag>.json, <tag>.done      (<tag> = f<k> or full)
  $FIL_WORK/maps/<name>/oof/<stem>.png       maps of the held-out fold (TTA --tta), + oof_<tag>.done
  $FIL_WORK/maps/<name>/test_<tag>/<stem>.png  test maps, + test_<tag>.done
Checkpoints every 5 epochs (resumable after preemption); a per-fold lock file avoids duplicate jobs.
"""
import argparse
import copy
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import H, W, load_gt, load_img, load_meta, normalize
from .dense import amp_dtype, build, predict_full, to_u8
from .metric import image_counts, summary
from .paths import P, add_path_args, apply_path_args
from .postproc import components_rles


def get_args():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, required=True)  # -1 = train on all
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="tu-efficientnetv2_s")
    ap.add_argument("--weights_init", dest="weights_init", default="imagenet")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--crops_per_img", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--fg_prob", type=float, default=0.7)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--sep_w", type=float, default=0.0)  # weight of a 2nd output channel (aux target); 0 = 1 channel
    ap.add_argument("--aux", default="sep")
    ap.add_argument("--hard_target", type=float, default=0.0)  # >0: binarise the soft target at this level
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--scale_aug", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", default="z")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0)  # keep this fraction of training days (same step count)
    ap.add_argument("--pseudo", default="")  # dir with images/<stem>.jpg and targets_<tag>/<stem>.png
    ap.add_argument("--pseudo_max", type=int, default=100000)
    return apply_path_args(ap.parse_args())


class TrainDS(Dataset):
    def __init__(self, imgs, tgts, seps, a, n):
        self.imgs, self.tgts, self.seps, self.a, self.n = imgs, tgts, seps, a, n
        self.fg = [np.argwhere(t[::8, ::8] > 64) * 8 for t in tgts]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = self.a
        rng = np.random
        k = rng.randint(len(self.imgs))
        im, tg, sp = self.imgs[k], self.tgts[k], self.seps[k]
        c = a.crop
        s = 1.0 + rng.uniform(-a.scale_aug, a.scale_aug) if a.scale_aug > 0 else 1.0
        src = min(int(round(c * s)), H)
        if rng.rand() < a.fg_prob and len(self.fg[k]):
            cy, cx = self.fg[k][rng.randint(len(self.fg[k]))]
            cy += rng.randint(-src // 3, src // 3 + 1)
            cx += rng.randint(-src // 3, src // 3 + 1)
        else:
            cy, cx = rng.randint(200, H - 200), rng.randint(200, W - 200)
        y0 = int(np.clip(cy - src // 2, 0, H - src))
        x0 = int(np.clip(cx - src // 2, 0, W - src))
        x = im[y0:y0 + src, x0:x0 + src].astype(np.float32)
        t = tg[y0:y0 + src, x0:x0 + src]
        sp_ = sp[y0:y0 + src, x0:x0 + src] if sp is not None else None
        if src != c:
            x = cv2.resize(x, (c, c), interpolation=cv2.INTER_LINEAR)
            t = cv2.resize(t, (c, c), interpolation=cv2.INTER_LINEAR)
            if sp_ is not None:
                sp_ = cv2.resize(sp_, (c, c), interpolation=cv2.INTER_NEAREST)
        r = rng.randint(4)  # D4
        fl = rng.rand() < 0.5
        x = np.rot90(x, r)
        t = np.rot90(t, r)
        if sp_ is not None:
            sp_ = np.rot90(sp_, r)
        if fl:
            x = x[:, ::-1]
            t = t[:, ::-1]
            if sp_ is not None:
                sp_ = sp_[:, ::-1]
        x = np.ascontiguousarray(x, dtype=np.float32)
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.2, 0.2)  # intensity aug on the normalised image
        if rng.rand() < 0.3:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
        if rng.rand() < 0.3:
            x = x + rng.normal(0, rng.uniform(0.02, 0.1), x.shape).astype(np.float32)
        t = np.ascontiguousarray(t, dtype=np.float32) / 255.0
        if a.hard_target > 0:
            t = (t >= a.hard_target).astype(np.float32)
        out = [torch.from_numpy(x)[None], torch.from_numpy(t)[None]]
        if sp_ is not None:
            out.append(torch.from_numpy(np.ascontiguousarray(sp_, dtype=np.float32) / 255.0)[None])
        return out


def soft_dice_loss(logit, t, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = (p + t).sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def main():
    a = get_args()
    dt = amp_dtype(a.amp)
    torch.manual_seed(a.seed * 100 + a.fold + 1)
    np.random.seed(a.seed * 100 + a.fold + 1)
    out = f"{P.runs}/{a.name}"
    mdir = f"{P.maps}/{a.name}"
    os.makedirs(out, exist_ok=True)
    os.makedirs(f"{mdir}/oof", exist_ok=True)
    tag = f"f{a.fold}" if a.fold >= 0 else "full"
    done_flag = f"{out}/{tag}.done"
    if os.path.exists(done_flag):
        print("already done", done_flag)
        return
    lock = f"{out}/running_{tag}.lock"
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 900 and \
            open(lock).read().strip() != os.environ.get("SLURM_JOB_ID", "local"):
        # a requeued (preempted) Slurm job keeps its job id, so its own fresh lock does not block it
        print("another job is running this fold (fresh lock); skipping", lock)
        return
    open(lock, "w").write(os.environ.get("SLURM_JOB_ID", "local"))
    targs = {k: v for k, v in vars(a).items() if k not in ("data", "work", "weights", "images")}
    targs["weights"] = targs.pop("weights_init")  # key name used in args_<tag>.json
    json.dump(targs, open(f"{out}/args_{tag}.json", "w"), indent=1)
    meta = load_meta()
    gt = load_gt()
    stems = meta["stems"]
    folds = meta["folds"]
    tr = [s for s in stems if folds[s] != a.fold]
    va = [s for s in stems if folds[s] == a.fold] if a.fold >= 0 else []
    n_steps_imgs = len(tr)
    if a.frac < 1:
        days = sorted({s[:8] for s in tr})
        rs = np.random.RandomState(7)
        keep = set(rs.choice(days, int(round(len(days) * a.frac)), replace=False).tolist())
        tr = [s for s in tr if s[:8] in keep]
    print(f"train {len(tr)} val {len(va)}", flush=True)
    t0 = time.time()
    imgs = [normalize(load_img(s), a.norm).astype(np.float16) for s in tr]
    tgts = [cv2.imread(f"{P.cache}/targets/{s}.png", 0) for s in tr]
    nch = 2 if a.sep_w > 0 else 1
    seps = [cv2.imread(f"{P.cache}/{a.aux}/{s}.png", 0) if nch == 2 else None for s in tr]
    if a.pseudo:
        pt = f"{a.pseudo}/targets_{tag}"
        ps = sorted(f[:-4] for f in os.listdir(pt) if f.endswith(".png"))[:a.pseudo_max]
        for s in ps:
            im = cv2.imread(f"{a.pseudo}/images/{s}.jpg", cv2.IMREAD_GRAYSCALE)
            imgs.append(normalize(im, a.norm).astype(np.float16))
            tgts.append(cv2.imread(f"{pt}/{s}.png", 0))
            seps.append(None)
        n_steps_imgs += len(ps)
        print(f"pseudo images {len(ps)} from {pt}", flush=True)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    ds = TrainDS(imgs, tgts, seps, a, n=n_steps_imgs * a.crops_per_img)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True,
                    worker_init_fn=lambda w: np.random.seed((a.seed * 1000 + a.fold * 100 + w + int(time.time())) % 2**31))
    dev = torch.device("cuda")
    model = build(a.arch, a.encoder, a.weights_init if a.weights_init != "none" else None, nch)
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16"))
    steps = a.epochs * len(dl)
    warm = min(500, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    step = 0
    start_ep = 0
    ck = f"{out}/ckpt_{tag}.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_ep, step = st["ep"] + 1, st["step"]
        print(f"resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, a.epochs):
        model.train()
        tl = 0
        t1 = time.time()
        for batch in dl:
            x = batch[0].to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            t = batch[1].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=dt):
                o = model(x)
            o = o.float()
            loss = F.binary_cross_entropy_with_logits(o[:, :1], t) + a.dice_w * soft_dice_loss(o[:, :1], t)
            if nch == 2:
                sp = batch[2].to(dev, non_blocking=True)
                pw = torch.tensor(5.0 if a.aux == "sep" else 3.0, device=dev)
                loss = loss + a.sep_w * F.binary_cross_entropy_with_logits(o[:, 1:2], sp, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            d = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tl += loss.item()
        print(f"ep {ep} loss {tl / len(dl):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t1:.0f}s", flush=True)
        os.utime(lock, None)
        if (ep + 1) % 5 == 0 and ep < a.epochs - 1:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(), "ep": ep, "step": step}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        if va and ((ep + 1) % a.eval_every == 0 or ep == a.epochs - 1):  # quick validation: CC at 0.5, area >= 300
            c = np.zeros(4)
            for s in va[:60] if ep < a.epochs - 1 else va:
                p = predict_full(ema, normalize(load_img(s), a.norm), tta=1, nch=nch, dtype=dt)[0]
                c += image_counts(gt[s], components_rles(p, t_lo=0.5, min_area=300))
            print(f"  val ep {ep} quick PQ {summary(c)}", flush=True)
    torch.save(ema.state_dict(), f"{out}/model_{tag}.pt")
    for i_, s in enumerate(va):  # OOF maps of the held-out fold
        if i_ % 20 == 0:
            os.utime(lock, None)
        p = predict_full(ema, normalize(load_img(s), a.norm), tta=a.tta, nch=nch, dtype=dt)
        cv2.imwrite(f"{mdir}/oof/{s}.png", to_u8(p[0]))
    if va:
        open(f"{mdir}/oof_{tag}.done", "w").write("ok")
    if not a.no_test:
        td = f"{mdir}/test_{tag}"
        os.makedirs(td, exist_ok=True)
        for i_, s in enumerate(meta["test_stems"]):
            if i_ % 20 == 0:
                os.utime(lock, None)
            p = predict_full(ema, normalize(load_img(s, test=True), a.norm), tta=a.tta, nch=nch, dtype=dt)
            cv2.imwrite(f"{td}/{s}.png", to_u8(p[0]))
        open(f"{mdir}/test_{tag}.done", "w").write("ok")
    open(done_flag, "w").write("ok")
    if os.path.exists(ck):
        os.remove(ck)
    if os.path.exists(lock):
        os.remove(lock)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
