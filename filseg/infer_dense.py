"""Dense inference: probability maps of trained fold models for the held-out fold (OOF) or for the test frames.

    python -m filseg.infer_dense --run e6_effv2s_c1536 --split test --tta 8            # all 5 fold models
    python -m filseg.infer_dense --run e1_effv2s --map e1_tta8 --split oof --folds 2 --tta 8

Weights are read from $FIL_WEIGHTS/dense/<run>/{model,args}_f<k>.pt|json (released weights) or, with
--from_runs, from $FIL_WORK/runs/<run>/ (models trained by train_dense.py). Each map is the D4-TTA mean of the
sigmoid output, computed under autocast with the model's training dtype (args "amp": bf16 or fp16), stored as
uint8 round(p * 255) in $FIL_WORK/maps/<map>/oof/<stem>.png or maps/<map>/test_f<k>/<stem>.png.
Existing maps are kept (restartable); a <split>_f<k>.done marker is written when a fold model is finished.
"""
import argparse
import os
import time

import cv2
import torch

from .common import list_stems, load_img, load_meta, normalize
from .dense import amp_dtype, load_model, predict_full, to_u8
from .paths import P, add_path_args, apply_path_args


def model_dir(run, from_runs=False):
    return os.path.join(P.runs if from_runs else os.path.join(P.weights, "dense"), run)


def run_maps(run, map_name, split, folds, tta, from_runs=False, stems=None):
    mdir = model_dir(run, from_runs)
    out_root = os.path.join(P.maps, map_name)
    meta = load_meta() if split == "oof" else None
    test = list_stems(P.images) if split == "test" else None
    for k in folds:
        tag = f"f{k}"
        done = os.path.join(out_root, f"{split}_{tag}.done")
        if split == "oof":
            todo = [s for s in meta["stems"] if meta["folds"][s] == k]
            out = os.path.join(out_root, "oof")
        else:
            todo = test
            out = os.path.join(out_root, f"test_{tag}")
        if stems is not None:
            todo = [s for s in todo if s in stems]
        if os.path.exists(done) and stems is None:
            print("done already", done, flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        model, targs = load_model(mdir, tag)
        dt = amp_dtype(targs.get("amp", "bf16"))
        norm = targs.get("norm", "z")
        t0, n = time.time(), 0
        for s in todo:
            f = os.path.join(out, f"{s}.png")
            if os.path.exists(f):
                continue
            p = predict_full(model, normalize(load_img(s, test=(split == "test")), norm), tta=tta, dtype=dt)[0]
            cv2.imwrite(f + ".tmp.png", to_u8(p))
            os.replace(f + ".tmp.png", f)
            n += 1
        if stems is None:
            open(done, "w").write("ok")
        print(f"{map_name} {split} {tag}: {n} maps in {time.time() - t0:.0f}s ({targs['encoder']}, {targs.get('amp')}, "
              f"norm {norm}, tta {tta})", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--run", required=True, help="model run name (folder with model_f<k>.pt)")
    ap.add_argument("--map", default=None, help="output map name (default: run)")
    ap.add_argument("--split", choices=["oof", "test"], required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--from_runs", action="store_true", help="read models from $FIL_WORK/runs instead of $FIL_WEIGHTS")
    a = apply_path_args(ap.parse_args())
    run_maps(a.run, a.map or a.run, a.split, a.folds, a.tta, a.from_runs)


if __name__ == "__main__":
    main()
