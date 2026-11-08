"""Run one final configuration end to end (final/s15.json or final/s16.json).

Test (inference from released weights, writes the submission CSV):
    python -m filseg.pipeline --config final/s15.json --split test --out submission_s15.csv \
        --data DATA --work WORK --weights WEIGHTS [--images DIR] [--gpus 0 1 2 3]
OOF (all 5 folds; reproduces the cross-validated PQ):
    python -m filseg.pipeline --config final/s15.json --split oof --data DATA --work WORK --weights WEIGHTS \
        [--pool_refine DIR --pool_cands DIR] [--gpus ...]

Stages (each writes its outputs under $FIL_WORK and is skipped when they already exist, so a run can resume):
  maps        every fold model of every map in config["maps"] predicts every frame (D4 TTA)     -> maps/
  candidates  ensemble map -> nested components + descriptors                                  -> cands/
  scorer      test: released boosters; oof: cross-fitted boosters refit on the OOF candidates  -> pp/
  crops       refiner / rescorer crops of the level-0.3 candidates (+ extra channels if needed) -> refine/
  refiner     refined crops from the released refiner (oof: the fold-k refiner for fold k)    -> refine/
  pool        oof only: the other candidate sets of the rescorer training pool (final/pool.json): candidates,
              cross-fitted scorer and crops, built from the released dense weights (skipped with --pool_refine)
  rescorer    CNN rescorer outputs; test: released full-data checkpoints; oof: fold checkpoints from
              $FIL_WEIGHTS/rescorer when present, otherwise retrained on the pool (about a minute each on one H200)
  assemble    score = (1 - w_cnn) * u_hat + w_cnn * rescorer, keep score > tau, paste refined masks
Stage wall times are appended to $FIL_WORK/timings.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import assemble, candidates, refiner, rescorer, scorer
from .metric import summary
from .paths import P, add_path_args, apply_path_args

FOLDS = [0, 1, 2, 3, 4]


def log_time(stage, t0, cfg, split):
    os.makedirs(P.work, exist_ok=True)
    rec = dict(config=cfg["name"], split=split, stage=stage, seconds=round(time.time() - t0, 1),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    open(os.path.join(P.work, "timings.jsonl"), "a").write(json.dumps(rec) + "\n")
    print(f"[{cfg['name']} {split}] {stage}: {rec['seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------- dense maps
def all_maps(cfg, split):
    """Maps of the configuration; for OOF also the maps needed to rebuild the rescorer training pool."""
    maps = dict(cfg["maps"])
    if split == "oof" and cfg.get("_pool"):
        need = pool_tags(cfg)
        for name, m in cfg["_pool"]["maps"].items():
            if any(name in [x.split(":")[0] for x in cfg["_pool"]["tags"][t]["members"]] for t in need):
                maps.setdefault(name, m)
        for t in need:
            for x in cfg["_pool"]["tags"][t]["members"]:
                assert x.split(":")[0] in maps, (t, x)
    return maps


def pool_tags(cfg):
    tags = []
    for g in cfg["rescorer"]["groups"].values():
        tags += [t for t in g["train_tags"] if t != cfg["tag"] and t not in tags]
    return tags


def dense_tasks(cfg, split, folds):
    tasks = []
    for name, m in all_maps(cfg, split).items():
        for k in folds:
            done = os.path.join(P.maps, name, f"{split}_f{k}.done")
            if not os.path.exists(done):
                tasks.append(dict(run=m["run"], map=name, tta=m["tta"], fold=k))
    return tasks


def dense_stage(cfg, split, folds, gpus):
    from .infer_dense import run_maps
    tasks = dense_tasks(cfg, split, folds)
    if not tasks:
        return
    if len(gpus) <= 1:
        for t in tasks:
            run_maps(t["run"], t["map"], split, [t["fold"]], t["tta"])
        return
    tasks.sort(key=lambda t: "effv2m" not in t["run"])  # heavier models first
    procs = []
    for i, g in enumerate(gpus):
        mine = tasks[i::len(gpus)]
        if not mine:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-m", "filseg.pipeline", "--dense_worker", json.dumps(mine), "--split", split,
               "--config", "-"]
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    assert all(r == 0 for r in rc), f"dense workers failed: {rc}"


def dense_worker(tasks, split):
    from .infer_dense import run_maps
    for t in json.loads(tasks):
        run_maps(t["run"], t["map"], split, [t["fold"]], t["tta"])


# ------------------------------------------------------------------------------------------- stages
def wpath(*parts):
    return os.path.join(P.weights, *parts)


def exists(path):
    return os.path.exists(path)


def run(cfg, split, out=None, gpus=(0,), pool_refine=None, pool_cands=None, workers=12, stages=None):
    tag = cfg["tag"]
    rs = cfg["rescorer"]
    cr = cfg["crops"]
    want = (lambda s: True) if not stages else (lambda s: s in stages)

    t0 = time.time()
    if want("maps"):
        dense_stage(cfg, split, FOLDS, list(gpus))
        log_time("maps", t0, cfg, split)

    t0 = time.time()
    if want("candidates") and not exists(f"{P.cands}/{tag}_{split}.pkl"):
        candidates.build(tag, cfg["ensemble"], split, cfg["thresholds"], workers)
        log_time("candidates", t0, cfg, split)

    t0 = time.time()
    if split == "oof" and want("scorer") and not exists(f"{P.pp}/{tag}_oofq.npz"):
        scorer.fit_tag(tag, cr["sel_t"])
        log_time("scorer", t0, cfg, split)

    t0 = time.time()
    if want("crops") and not exists(f"{P.refine}/{tag}_{split}.npz"):
        refiner.make_crops(tag, cfg["ensemble"], split, scorer_dir=wpath("scorer") if split == "test" else None,
                           sel_t=cr["sel_t"], min_q=cr["min_q"], sel_tau=cr["sel_tau"], workers=workers)
        log_time("crops", t0, cfg, split)
    t0 = time.time()
    if want("crops") and cfg.get("xc") and not exists(f"{P.refine}/{tag}_{split}_xc.npz"):
        rescorer.extra_channels(tag, split, cfg["xc"]["members"], cfg["xc"]["students"], workers)
        log_time("extra_channels", t0, cfg, split)

    t0 = time.time()
    ref = cfg["refiner"]
    if want("refiner"):
        if split == "test" and not exists(f"{P.refine}/{tag}_pred_test.npz"):
            refiner.predict_split(tag, "test", wpath("refiner", ref["ckpt"]), ref["amp"])
        if split == "oof":
            for k in FOLDS:
                if not exists(f"{P.refine}/{tag}_pred_f{k}.npz"):
                    refiner.predict_split(tag, "oof", wpath("refiner", ref["fold_ckpt"].format(k=k)), ref["amp"], fold=k)
        log_time("refiner", t0, cfg, split)

    t0 = time.time()
    if split == "oof" and want("pool") and not pool_refine and cfg.get("_pool"):
        # rescorer training pool: OOF crops of the other candidate sets in the groups' train_tags (final/pool.json)
        for t in pool_tags(cfg):
            spec = cfg["_pool"]["tags"][t]
            if not exists(f"{P.cands}/{t}_oof.pkl"):
                candidates.build(t, spec["members"], "oof", spec["thresholds"], workers)
            if not exists(f"{P.pp}/{t}_oofq.npz"):
                scorer.fit_tag(t, cr["sel_t"])
            if not exists(f"{P.refine}/{t}_oof.npz"):
                refiner.make_crops(t, spec["members"], "oof", sel_t=cr["sel_t"], min_q=cr["min_q"], sel_tau=cr["sel_tau"],
                                   workers=workers)
        log_time("pool", t0, cfg, split)

    t0 = time.time()
    if want("rescorer"):
        for gname, g in rs["groups"].items():
            for m in g["members"]:
                name, seed = m["name"], m["seed"]
                if split == "test":
                    if not exists(f"{P.refine}/{tag}_{name}_test.npz"):
                        rescorer.predict_split(tag, "test", ckpt=wpath("rescorer", f"{tag}_{name}_full.pt"), name=name, xc=g["xc"])
                else:
                    for k in FOLDS:
                        if exists(f"{P.refine}/{tag}_{name}_f{k}.npz"):
                            continue
                        ck = wpath("rescorer", f"{tag}_{name}_f{k}.pt")
                        if exists(ck):
                            rescorer.predict_split(tag, "oof", ckpt=ck, name=name, fold=k, xc=g["xc"])
                        else:  # fold checkpoints are not part of the inference bundle: retrain (cheap)
                            subprocess.check_call([sys.executable, "-m", "filseg.rescorer", "train", "--tag", tag,
                                                   "--fold", str(k), "--seed", str(seed), "--epochs", str(g["epochs"]),
                                                   "--name", name, "--train_tags", *g["train_tags"]]
                                                  + (["--xc"] if g["xc"] else [])
                                                  + (["--pool_refine", pool_refine] if pool_refine else [])
                                                  + (["--pool_cands", pool_cands] if pool_cands else []))
            rescorer.average(tag, [m["name"] for m in g["members"]], gname)
        rescorer.average(tag, list(rs["groups"]), rs["combine"])
        log_time("rescorer", t0, cfg, split)

    t0 = time.time()
    if want("assemble"):
        per_frame = {} if split == "oof" else None
        if split == "oof" and not out:
            out = os.path.join(P.work, f"oof_{cfg['name']}.csv")
        res = assemble.run(tag, split, rescore=rs["combine"], w_cnn=rs["w_cnn"], tau=cfg["tau"], thrs=[ref["thr"]],
                           workers=workers, out=out, per_frame=per_frame)
        log_time("assemble", t0, cfg, split)
        if split == "oof":
            report = {"pooled": summary(res[ref["thr"]]), "seed_masks": summary(res["seed"]),
                      "per_fold": per_fold(per_frame, ref["thr"])}
            json.dump(report, open(os.path.join(P.work, f"oof_{cfg['name']}.json"), "w"), indent=1)
            print(json.dumps(report, indent=1), flush=True)


def per_fold(per_frame, thr):
    from .common import load_meta
    folds = load_meta()["folds"]
    out = {}
    for k in FOLDS:
        c = [v[thr] for s, v in per_frame.items() if folds[s] == k]
        if c:
            out[f"f{k}"] = summary(np.sum(c, 0))
    return out


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["test", "oof"], default="test")
    ap.add_argument("--out", default=None, help="submission CSV (test)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=None,
                    help="subset of: maps candidates scorer crops refiner pool rescorer assemble")
    ap.add_argument("--pool_refine", default=None, help="rescorer training pool crops (oof retraining)")
    ap.add_argument("--pool_cands", default=None, help="rescorer training pool candidate tables")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dense_worker", default=None, help=argparse.SUPPRESS)
    a = apply_path_args(ap.parse_args())
    if a.dense_worker:
        return dense_worker(a.dense_worker, a.split)
    cfg = json.load(open(a.config))
    pool_file = os.path.join(os.path.dirname(os.path.abspath(a.config)), "pool.json")
    cfg["_pool"] = json.load(open(pool_file)) if os.path.exists(pool_file) else None
    if a.split == "test" and not a.out:
        a.out = os.path.join(P.work, f"submission_{cfg['name']}.csv")
    run(cfg, a.split, a.out, a.gpus, a.pool_refine, a.pool_cands, a.workers, a.stages)


if __name__ == "__main__":
    main()
