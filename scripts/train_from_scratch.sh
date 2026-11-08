#!/bin/bash
# Full training recipe of the two final pipelines (s15, s16), stage by stage, as run during the competition.
# Every command is one job (one GPU unless noted); the loops are sequential here but independent across folds,
# so in practice they were spread over many GPUs. Rough cost: ~70 H200-hours in total (see README).
#
# usage: FIL_DATA=... FIL_WORK=... PSEUDO=... bash scripts/train_from_scratch.sh [stage ...]
#   stages: prep teachers tta8 gong kd students pool refiner rescorer final   (default: all, in this order)
# GPUs without bf16 (e.g. V100): add --amp fp16 to train_dense (as done for some folds; see each args_f<k>.json).
set -euo pipefail
: "${FIL_DATA:?set FIL_DATA}" "${FIL_WORK:?set FIL_WORK}"
PSEUDO=${PSEUDO:-$FIL_WORK/pseudo}
PY="python -m"
FOLDS="0 1 2 3 4"
EFS=tu-tf_efficientnetv2_s.in21k_ft_in1k
EFM=tu-tf_efficientnetv2_m.in21k_ft_in1k
C1536="--crop 1536 --bs 4 --crops_per_img 4 --tta 8"
STAGES=${*:-prep teachers tta8 gong kd students pool refiner rescorer final}
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---- 0. ground truth, consensus targets, day-grouped folds
has prep && $PY filseg.prep

# ---- 1. dense teachers: U-Nets on the soft consensus target, 60 epochs, 5 folds each
#         (fold k is trained without fold k and writes the OOF maps of fold k and its test maps)
if has teachers; then
  for k in $FOLDS; do
    $PY filseg.train_dense --name e1_effv2s      --fold $k --encoder $EFS                     # 1024 px crops, TTA4 maps
    $PY filseg.train_dense --name e3_effv2s_flat --fold $k --encoder $EFS --norm flat
    $PY filseg.train_dense --name e7_effv2m      --fold $k --encoder $EFM
    $PY filseg.train_dense --name e6_effv2s_c1536       --fold $k --encoder $EFS $C1536       # 1536 px crops, TTA8 maps
    $PY filseg.train_dense --name e10_effv2m_c1536      --fold $k --encoder $EFM $C1536
    $PY filseg.train_dense --name e12_effv2s_flat_c1536 --fold $k --encoder $EFS $C1536 --norm flat
  done
fi

# ---- 1b. TTA8 maps of the 1024-px models (ensemble members e1_tta8, e3_tta8, e7_tta8)
if has tta8; then
  for split in oof test; do
    $PY filseg.infer_dense --from_runs --run e1_effv2s      --map e1_tta8 --split $split --tta 8
    $PY filseg.infer_dense --from_runs --run e3_effv2s_flat --map e3_tta8 --split $split --tta 8
    $PY filseg.infer_dense --from_runs --run e7_effv2m      --map e7_tta8 --split $split --tta 8
  done
fi

# ---- 2. unlabelled GONG frames (other days, >= 6 h from any test frame) and quality screen
if has gong; then
  $PY filseg.gong.fetch --n 1500 --out $PSEUDO --seed 0       # ~1 request/s, a few hours
  $PY filseg.gong.screen --dir $PSEUDO                        # -> keep.txt (1314 of 1500 frames)
fi

# ---- 3. distillation targets: for student fold k, the mean map of the three 1536-px teachers' fold-k models
if has kd; then
  for k in $FOLDS; do
    for run in e6_effv2s_c1536 e10_effv2m_c1536 e12_effv2s_flat_c1536; do
      $PY filseg.teacher maps --dir $PSEUDO --run $run --tag f$k --tta 4 --from_runs
    done
    $PY filseg.teacher avg --dir $PSEUDO --tag f$k --runs e6_effv2s_c1536 e10_effv2m_c1536 e12_effv2s_flat_c1536
  done
fi

# ---- 4. students: same recipe on train frames + 1314 pseudo-labelled frames, 30 epochs
if has students; then
  for k in $FOLDS; do
    $PY filseg.train_dense --name e31_kd_effv2s_c1536      --fold $k --encoder $EFS $C1536 --epochs 30 --pseudo $PSEUDO
    $PY filseg.train_dense --name e33_kd_effv2m_c1536      --fold $k --encoder $EFM $C1536 --epochs 30 --pseudo $PSEUDO
    $PY filseg.train_dense --name e34_kd_effv2s_flat_c1536 --fold $k --encoder $EFS $C1536 --epochs 30 --pseudo $PSEUDO --norm flat
  done
fi

# ---- 5. candidate sets, cross-fitted LightGBM scorers and refiner/rescorer crops
#         ensB / eS are the final pipelines; the others only feed the rescorer training pool (final/pool.json)
if has pool; then
  python - <<'PY'
import json
from filseg import candidates, refiner, rescorer, scorer
pool = json.load(open("final/pool.json"))
for tag, spec in pool["tags"].items():
    candidates.build(tag, spec["members"], "oof", spec["thresholds"], workers=16)
    scorer.fit_tag(tag)
    refiner.make_crops(tag, spec["members"], "oof", workers=16)
for tag in ("ensB", "eS"):
    spec = pool["tags"][tag]
    candidates.build(tag, spec["members"], "test", spec["thresholds"], workers=16)
    refiner.make_crops(tag, spec["members"], "test", workers=16)          # boosters from $FIL_WORK/pp
s15 = json.load(open("final/s15.json"))
for split in ("oof", "test"):                                             # extra rescorer channels of s15
    rescorer.extra_channels("ensB", split, s15["xc"]["members"], s15["xc"]["students"], workers=16)
PY
fi

# ---- 6. crop refiners (cross-fitted; -1 = all folds, predicts the test crops)
if has refiner; then
  for k in $FOLDS -1; do
    $PY filseg.refiner train --tag ensB --fold $k --amp fp16      # ensB refiner was trained in fp16 (V100)
    $PY filseg.refiner train --tag eS   --fold $k
  done
fi

# ---- 7. CNN rescorers: every group and seed of final/s15.json and final/s16.json, folds 0-4 and -1
if has rescorer; then
  python - <<'PY'
import json, subprocess, sys
from filseg import rescorer
for name in ("s15", "s16"):
    cfg = json.load(open(f"final/{name}.json"))
    tag, rs = cfg["tag"], cfg["rescorer"]
    for gname, g in rs["groups"].items():
        for m in g["members"]:
            for k in (0, 1, 2, 3, 4, -1):
                subprocess.check_call([sys.executable, "-m", "filseg.rescorer", "train", "--tag", tag, "--fold", str(k),
                                       "--seed", str(m["seed"]), "--epochs", str(g["epochs"]), "--name", m["name"],
                                       "--train_tags", *g["train_tags"]] + (["--xc"] if g["xc"] else []))
        rescorer.average(tag, [m["name"] for m in g["members"]], gname)
    rescorer.average(tag, list(rs["groups"]), rs["combine"])
PY
fi

# ---- 8. OOF PQ and test CSVs
if has final; then
  for name in s15 s16; do
    python - "$name" <<'PY'
import json, sys
from filseg import assemble
cfg = json.load(open(f"final/{sys.argv[1]}.json")); rs = cfg["rescorer"]
for split in ("oof", "test"):
    assemble.run(cfg["tag"], split, rescore=rs["combine"], w_cnn=rs["w_cnn"], tau=cfg["tau"], thrs=[cfg["refiner"]["thr"]],
                 out=None if split == "oof" else f"submission_{cfg['name']}.csv")
PY
  done
fi
