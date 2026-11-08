"""Teacher maps on unlabelled frames = distillation targets for the students (noisy-student style, one round).

1) maps of one teacher model:
     python -m filseg.teacher maps --dir PSEUDO_DIR --run e6_effv2s_c1536 --tag f0 [--tta 4] [--from_runs]
   writes PSEUDO_DIR/maps/<run>_<tag>/<stem>.png for every frame listed in PSEUDO_DIR/keep.txt (restartable).
2) average the teachers of one student fold:
     python -m filseg.teacher avg --dir PSEUDO_DIR --tag f0 --runs e6_effv2s_c1536 e10_effv2m_c1536 e12_effv2s_flat_c1536
   writes PSEUDO_DIR/targets_<tag>/<stem>.png = mean of the teacher maps (uint8).

Fold hygiene: the targets of student fold k (targets_f<k>) come only from the teachers' fold-k models, which never
saw fold-k frames, so the student's OOF maps stay honest. Teacher maps use bf16 autocast when the GPU supports it.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from .common import normalize
from .dense import load_model, predict_full, to_u8
from .infer_dense import model_dir
from .paths import add_path_args, apply_path_args


def keep_list(d):
    stems = sorted(f[:-4] for f in os.listdir(f"{d}/images") if f.endswith(".jpg"))
    if os.path.exists(f"{d}/keep.txt"):  # quality screen (gong.screen)
        keep = set(open(f"{d}/keep.txt").read().split())
        stems = [s for s in stems if s in keep]
    return stems


def model_maps(d, run, tag, tta=4, from_runs=False):
    model, targs = load_model(model_dir(run, from_runs), tag)
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = f"{d}/maps/{run}_{tag}"
    os.makedirs(out, exist_ok=True)
    n = 0
    for s in keep_list(d):
        if os.path.exists(f"{out}/{s}.png"):
            continue
        im = cv2.imread(f"{d}/images/{s}.jpg", cv2.IMREAD_GRAYSCALE)
        p = predict_full(model, normalize(im, targs.get("norm", "z")), tta=tta, dtype=dt)[0]
        cv2.imwrite(f"{out}/{s}.png.tmp.png", to_u8(p))
        os.replace(f"{out}/{s}.png.tmp.png", f"{out}/{s}.png")
        n += 1
    print("wrote", n, "maps", out, flush=True)


def average(d, tag, runs):
    out = f"{d}/targets_{tag}"
    os.makedirs(out, exist_ok=True)
    specs = []
    for r in runs:
        run, w = (r.split(":") + ["1"])[:2]
        mtag = "full" if run.endswith("@full") else tag
        specs.append((f"{d}/maps/{run.replace('@full', '')}_{mtag}", float(w)))
    stems = sorted(set.intersection(*[{f[:-4] for f in os.listdir(p) if f.endswith(".png")} for p, _ in specs]))
    sel = set(keep_list(d))
    stems = [s for s in stems if s in sel]
    for f in os.listdir(out):  # drop targets of frames that are no longer selected
        if f.endswith(".png") and f[:-4] not in set(stems):
            os.remove(f"{out}/{f}")
    for s in stems:
        acc = sum(cv2.imread(f"{p}/{s}.png", 0).astype(np.float32) * w for p, w in specs) / sum(w for _, w in specs)
        cv2.imwrite(f"{out}/{s}.png", np.clip(np.round(acc), 0, 255).astype(np.uint8))
    print("averaged", len(stems), "->", out, flush=True)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("cmd", choices=["maps", "avg"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--tag", default="f0")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--from_runs", action="store_true")
    a = apply_path_args(ap.parse_args())
    d = os.path.abspath(a.dir)
    if a.cmd == "maps":
        model_maps(d, a.run, a.tag, a.tta, a.from_runs)
    else:
        average(d, a.tag, a.runs)


if __name__ == "__main__":
    main()
