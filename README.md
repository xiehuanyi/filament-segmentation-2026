# Solar filament instance segmentation on GONG H-alpha frames

Code and weights of our two final submissions to the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(Kaggle, MAGFiLO data). Input: 2048 x 2048 grayscale GONG H-alpha frames. Output: one pixel-disjoint mask
per filament, written as COCO RLE (`filament_id,segmentation_rle`), scored with pooled Panoptic Quality
against every annotator record of a frame.

| pick | Kaggle submission | pipeline | 5-fold OOF PQ |
|---|---|---|---|
| **s15** | 57951397 (`s15_ensB_ref050_rsAll.csv`) | 6-model teacher ensemble (ensB) -> scorer -> refiner -> rescorer, tau 0.22 | **0.4678** |
| **s16** | 57951808 (`s16_eS_ref050_rsAll.csv`) | 3 distilled student models (eS) -> scorer -> refiner -> rescorer, tau 0.26 | **0.4669** |

Each pick is fully described by one configuration file: `final/s15.json`, `final/s16.json`.

## Pipeline

1. **Dense models** (`filseg/train_dense.py`, `filseg/infer_dense.py`). U-Nets (segmentation_models_pytorch,
   EfficientNetV2-S/M encoders) predict for every pixel the probability that an annotator marks it, trained on the
   mean of the annotators' masks (BCE + soft Dice, EMA weights, D4 / scale / photometric augmentation) on random
   1024 or 1536 px crops of the native frames. Inference on the full frame with 8-fold dihedral TTA; a model's
   test map is the mean of its 5 fold models.
2. **Distillation** (`filseg/gong/`, `filseg/teacher.py`), s16 only. 1500 unlabelled GONG frames from the public NSO
   archive (other days than any train/test frame and >= 6 h from every test timestamp; allowed by the hosts for
   training, never used at inference), 1314 after a quality screen. Students are trained on the training frames
   plus these frames, whose targets are the mean maps of the three 1536 px teachers of the same fold.
3. **Candidates** (`filseg/candidates.py`). Connected components of the ensemble map at nested thresholds
   (s15: 0.2 ... 0.8; s16: 0.3) with shape / probability / intensity descriptors. Pooled PQ over pixel-disjoint
   predictions decomposes into one term per candidate, so a candidate should be kept when its annotator-averaged
   matched IoU q exceeds PQ*/2; the next stages estimate q.
4. **Scorer** (`filseg/scorer.py`). LightGBM hurdle model u = P(matched by an annotator) x E[IoU | matched] on the
   descriptors, cross-fitted by fold.
5. **Refiner** (`filseg/refiner.py`). For each level-0.3 candidate, a 192 x 192 crop (frame, ensemble map, candidate
   mask, other candidates) goes through a ResNet-34 U-Net that re-draws the filament mask (target: mean of the
   matched annotator masks).
6. **Rescorer** (`filseg/rescorer.py`). A ResNet-34 regressor on the same crops predicts q; several seeds / training
   pools are averaged (s15: 3 groups, 8 networks; s16: 2 groups, 5 networks).
7. **Selection and assembly** (`filseg/assemble.py`). score = mean(u, rescorer); keep score > tau; paste the refined
   masks (threshold 0.5) in decreasing score order, pixel-disjoint.

## Repository layout

| path | content |
|---|---|
| `filseg/paths.py` | where data, work files and weights live (`FIL_DATA`, `FIL_WORK`, `FIL_WEIGHTS`, `FIL_IMAGES` or `--data/--work/--weights/--images`) |
| `filseg/common.py` | frame I/O, disk z-score and flat-field normalisation |
| `filseg/metric.py` | the host's pooled PQ on RLEs |
| `filseg/prep.py` | COCO json -> ground truth, soft consensus targets, day-grouped folds |
| `filseg/dense.py`, `train_dense.py`, `infer_dense.py` | U-Net definition, training of one fold, probability maps from saved models |
| `filseg/gong/fetch.py`, `gong/screen.py`, `teacher.py` | unlabelled frames, quality screen, teacher maps / distillation targets |
| `filseg/candidates.py`, `scorer.py`, `refiner.py`, `rescorer.py`, `assemble.py` | stages 3-7 above |
| `filseg/pipeline.py` | runs a final configuration end to end (test or OOF) |
| `filseg/evaluate.py` | PQ, IoU / Dice distributions, 1:n and n:1 matching (host self-evaluation notebook) |
| `filseg/compare.py` | compares two submissions or two intermediate files (reproduction checks) |
| `filseg/submission.py` | CSV writing and validation |
| `final/s15.json`, `final/s16.json` | the two final configurations |
| `final/pool.json` | candidate sets that form the rescorer training pool |
| `final/weights_manifest.csv` | every weight file with size and sha256 |
| `scripts/train_from_scratch.sh` | the full training recipe, stage by stage |
| `scripts/get_weights.py` | downloads the weights (optional) and checks them against the manifest |
| `scripts/make_manifest.py`, `scripts/report_figures.py` | weights manifest; per-stage PQ, example overlays and the ablation chart |
| `experiments/` | analysis scripts (selection oracle, annotator agreement, human baselines) |
| `demo.ipynb` | end-to-end demo on the test frames with overlays |
| `figures/` | OOF evaluation of the submitted pipelines (IoU / Dice distributions, 1:n / n:1 matching, per-stage PQ, example overlays, ablation chart); `figures/from_weights/`: the same statistics recomputed from the released weights |

## Setup

Python 3.11 and an NVIDIA GPU. Inference was run on H200 and RTX 3090 GPUs (teacher maps also on an 11 GB RTX
2080 Ti); training used H200 (most folds), V100 32 GB and RTX 3090 GPUs.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # or: pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.8 PyTorch wheels; for another CUDA version change the index URL and the `+cu128`
suffixes. The ImageNet encoder weights are only needed for training (downloaded by timm / smp on first use).

### Data

```bash
kaggle competitions download -c filament-segmentation-2026 && unzip filament-segmentation-2026.zip -d DATA
```

`DATA` must contain `train/train_images/*.jpeg`, `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` and
`test/test_images/*.jpeg`. Every command takes `--data DATA --work WORK --weights WEIGHTS` (or the environment
variables `FIL_DATA`, `FIL_WORK`, `FIL_WEIGHTS`); `--images DIR` (`FIL_IMAGES`) predicts another folder of frames.
All outputs go below `WORK`.

### Weights

Inference bundle 7.33 GB (s15 alone 6.80 GB, s16 alone 2.54 GB), plus 0.98 GB of fold refiners for the OOF check;
`final/weights_manifest.csv` lists every file with size, sha256 and the picks that use it:

```
WEIGHTS/
  dense/<run>/model_f{0..4}.pt, args_f{0..4}.json   9 runs x 5 folds (EMA state dicts + training arguments)
  scorer/{ensB,eS}_lgb{0,1}.txt, {ensB,eS}.json      LightGBM boosters
  refiner/{ensB,eS}_refiner.pt                       full-data refiners (+ _f{0..4}.pt fold refiners, OOF only)
  rescorer/{ensB,eS}_<name>_full.pt                  13 full-data rescorers
```

Download: TODO (link to the hosted weights). Check a weights folder with
`python scripts/get_weights.py --out WEIGHTS` (all files and sha256; `--pick s16` for one pick only).

## Reproduce the submissions (inference from the released weights)

```bash
python -m filseg.pipeline --config final/s15.json --split test --data DATA --work WORK --weights WEIGHTS \
    --out submission_s15.csv --gpus 0          # several GPUs: --gpus 0 1 2 3 (dense stage only)
python -m filseg.pipeline --config final/s16.json --split test --data DATA --work WORK --weights WEIGHTS \
    --out submission_s16.csv
python -m filseg.compare csv submission_s15.csv s15_ensB_ref050_rsAll.csv     # against the submitted file
```

s15 needs the maps of all 9 dense runs (the 3 students feed two extra rescorer channels); s16 reuses the 3 student
maps when run in the same `WORK`. Stage wall times are appended to `WORK/timings.jsonl`. For the 180 test frames:
s15 21 min on two H200s (dense maps 18.6 min, everything else 2.2 min; about 40 min on one H200); s16 about 12 min on
one H200 and 32 min on one RTX 3090. Almost all of it is the dense stage (45 or 15 fold models x 8 TTA passes).

What reproduces exactly: see [Reproducibility](#reproducibility).

## Reproduce the cross-validation score

```bash
python -m filseg.prep --data DATA --work WORK
python -m filseg.pipeline --config final/s15.json --split oof --data DATA --work WORK --weights WEIGHTS --gpus 0 1 2 3
python -m filseg.pipeline --config final/s16.json --split oof --data DATA --work WORK --weights WEIGHTS
python -m filseg.evaluate --pred WORK/oof_s15.csv --work WORK --out_dir figures --name s15
```

Needs the fold refiners (`refiner/*_f{0..4}.pt`, bundle `oof` in the manifest). The OOF run predicts each
training frame with the fold model that did not see it, refits the cross-fitted
LightGBM scorer (on identical candidate tables the refit is bit-identical to the released boosters), rebuilds the rescorer training pool (`final/pool.json`,
the OOF crops of 12 candidate sets made from the same dense runs) and retrains the fold rescorers (about one
minute each on an H200; their checkpoints are not part of the weights bundle). The pooled and per-fold PQ are
written to `WORK/oof_s15.json`, the OOF predictions to `WORK/oof_s15.csv`.

## Train from scratch

`scripts/train_from_scratch.sh` lists every training command, stage by stage (each stage can be run alone:
`bash scripts/train_from_scratch.sh teachers`). Per-run arguments are also stored with every checkpoint
(`dense/<run>/args_f<k>.json`).

| stage | command (per fold k) | hardware | wall time |
|---|---|---|---|
| prep | `filseg.prep` | CPU | 1 min |
| 1024 px teachers e1 / e3 / e7 | `train_dense --encoder effv2-s / effv2-s --norm flat / effv2-m` (60 ep) | 1 GPU | 25-30 min per fold (H200) |
| 1536 px teachers e6 / e10 / e12 | `train_dense ... --crop 1536 --bs 4 --crops_per_img 4 --tta 8` (60 ep) | 1 GPU | 1.4-1.9 h per fold |
| TTA8 maps of e1 / e3 / e7 | `infer_dense --from_runs --tta 8` | 1 GPU | ~1 min per fold and split |
| unlabelled frames | `gong.fetch --n 1500`, `gong.screen` | CPU, network | a few hours (1 request/s) |
| distillation targets | `teacher maps` (3 teachers x 5 folds, TTA4), `teacher avg` | 1 GPU | ~5 min per model |
| students e31 / e33 / e34 | `train_dense ... --epochs 30 --pseudo PSEUDO` | 1 GPU | 2-3 h per fold |
| candidates, scorers, crops | `candidates`, `scorer`, `refiner crops` for the 12 sets of `final/pool.json` | 16-32 CPU cores | 2-5 min per set |
| refiners | `refiner train --fold k` (k = 0..4, -1) | 1 GPU | ~30 s per fold (H200), ~90 s (V100) |
| rescorers | `rescorer train --fold k --seed s` per group of the configs | 1 GPU | 15-60 s per run + pool loading |
| OOF PQ and test CSV | `assemble` | CPU | 1 min |

In total about 70 H200-hours, almost all of it in the 30 dense runs (15 teacher and 15 student fold models).
During the competition some folds ran on V100s (`--amp fp16`) or an RTX 3090; the `amp` field of each
`args_f<k>.json` records the precision used.

## Results (5-fold OOF PQ, host metric)

Numbers from our development log; "shared scorer" rows compare dense models with one
common scorer, the others are full pipelines with their own cross-fitted scorer.

| step | OOF PQ |
|---|---|
| single EfficientNetV2-S U-Net, 1024 px crops, simple connected-component rule (2 folds) | 0.440 |
| + learned LightGBM selection | 0.449 |
| single model, 1536 px instead of 1024 px crops (TTA8, shared scorer) | 0.4481 -> 0.4539 |
| teacher ensemble ensB (6 models, own scorer) | 0.4607 |
| + crop refiner | 0.4637 |
| + CNN rescorer blended with LightGBM (4 seeds, rsqE) | 0.4668 |
| + average of 3 rescorer groups (rsAll) = **s15** | **0.4678** |
| 3 distilled students eS (own scorer) | 0.4608 |
| + refiner | 0.4619 |
| + rescorer (3 seeds, 12-set pool) | 0.4657 |
| + average of 2 rescorer groups = **s16** | **0.4669** |
| oracle selection with the true q at threshold 0.3 (upper bound for the selector) | 0.528 |

Public leaderboard (about 90 test frames, rounded to 0.01): 0.40 for both picks.

## Validation protocol

* 5 folds grouped by observation day (`prep.py`: days shuffled with seed 2026 and dealt round-robin): frames of one
  day from different GONG sites show the same filaments, so they never straddle folds. All annotator records of a
  frame stay in its fold. Fold sizes 143 / 145 / 137 / 144 / 138 frames.
* Every learned stage is cross-fitted: the fold-k OOF prediction of the dense models, the scorer, the refiner and
  the rescorer comes from models trained without fold k. Student fold k is distilled only from the teachers' fold-k
  models, so its OOF maps stay honest.
* The score is the host's pooled PQ over all annotator records (`filseg/metric.py`), on all 707 training frames.
* The hyper-parameters chosen on OOF are the candidate level (0.3), tau, the refiner threshold (0.5) and the
  LightGBM / rescorer blend weight (0.5), each from a small grid.
* Test predictions use the mean map of the 5 fold models of each dense run and full-data scorer, refiner and
  rescorer models.

## Reproducibility

Checks run on 2026-11-08 (`filseg.compare` reproduces them):

| check | result |
|---|---|
| release code vs development code on identical inputs (the development probability maps, refined crops and rescorer outputs) | every intermediate file is identical: folds, ground truth and targets; candidate tables (s15: 12003 test / 46805 OOF candidates, s16: 1859 / 7222); LightGBM boosters and cross-fitted scores; crops and extra channels. Both submitted CSVs are reproduced **byte for byte**; OOF PQ 0.4678 (s15) and 0.4669 (s16) |
| rescorer checkpoints (not kept during development) retrained with the release code, same seeds and training pool, H200 | the test outputs of all 13 networks are bit-identical to the development runs; these retrained networks are the released checkpoints |
| full inference from the released weights in a fresh environment (`uv pip install -r requirements.txt`), H200 | s15: 1157 / 1157 filaments, 986 identical masks, the rest matched with IoU >= 0.95 (mean IoU 0.9998), PQ of our CSV against the submitted one 0.9998. s16: 1089 / 1089, 721 identical, IoU >= 0.98, PQ 0.9997 |
| full inference, RTX 3090 (CUDA 12.6 wheels) | s16: 1089 / 1089 filaments, 1088 matched (632 identical masks, mean IoU 0.9995, min 0.978), PQ against the submitted file 0.9986 |
| OOF from the released weights (5 folds; scorer refit, rescorer pool rebuilt, fold rescorers retrained) | s15 0.4672 (development 0.4678; folds 0.4579 / 0.4645 / 0.4746 / 0.4920 / 0.4477 vs 0.4600 / 0.4657 / 0.4741 / 0.4904 / 0.4497). s16 0.4664 (development 0.4669; folds 0.4561 / 0.4633 / 0.4741 / 0.4894 / 0.4502 vs 0.4542 / 0.4636 / 0.4719 / 0.4910 / 0.4543) |

Why the inference is not byte-identical: recomputed on the same GPU type and precision, the dense models are
bit-identical (the 30 test fold-maps and 4243 of the 6363 OOF frame maps of s15 that were computed on H200s during
development). The other 15 test fold-maps (and the matching OOF maps) had been computed on V100s (fp16) or an
RTX 3090; recomputing them on an H200 changes about 1e-4 of the pixels of a map (isolated
pixels, by a few uint8 levels). This moves some component boundaries by a pixel and changes the refiner inputs
slightly; scores change by 3e-4 (LightGBM) and 1e-4 (rescorer) on average and no keep decision changes. Every later
stage is deterministic given its inputs (LightGBM refits are bit-identical, and so are the rescorer and refiner
outputs on the same GPU type). In the OOF run the small map differences also change the candidate tables that the
scorer is refit on and from which the rescorer training pool is rebuilt, and the fold rescorers are retrained on that
pool; the resulting OOF PQ differs from the development value by -0.0006 (s15) and -0.0004 (s16).

## License

The code in this repository and the released weights are under the MIT License (see `LICENSE`).
The pretrained encoder weights we start from (timm / segmentation_models_pytorch), the MAGFiLO annotations and the
GONG H-alpha frames keep their own terms.
