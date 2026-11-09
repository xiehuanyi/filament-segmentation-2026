---
license: mit
tags:
  - image-segmentation
  - instance-segmentation
  - solar-physics
  - h-alpha
  - pytorch
---

# Solar filament instance segmentation: weights (Filament Segmentation Challenge 2026)

Weights of our two final submissions to the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The code, configs and full documentation are in the GitHub repository
[xiehuanyi/filament-segmentation-2026](https://github.com/xiehuanyi/filament-segmentation-2026).

Input: 2048 x 2048 GONG H-alpha frames. Output: one pixel-disjoint mask per filament.

| pick | pipeline | 5-fold OOF PQ |
|---|---|---|
| s15 | 6-model U-Net teacher ensemble -> LightGBM scorer -> crop refiner -> CNN rescorers | 0.4678 |
| s16 | 3 distilled student U-Nets -> LightGBM scorer -> crop refiner -> CNN rescorers | 0.4669 |

## Files

```
dense/<run>/model_f{0..4}.pt, args_f{0..4}.json   9 runs x 5 folds (EMA state dicts + training arguments)
scorer/{ensB,eS}_lgb{0,1}.txt, {ensB,eS}.json      LightGBM boosters
refiner/{ensB,eS}_refiner.pt                       full-data refiners (+ _f{0..4}.pt fold refiners, OOF check only)
rescorer/{ensB,eS}_<name>_full.pt                  13 full-data rescorers
```

The inference bundle is 7.33 GB (s15 alone 6.80 GB, s16 alone 2.54 GB); the fold refiners add 0.98 GB.
`final/weights_manifest.csv` in the GitHub repository lists every file with its size and sha256.

## Usage

```bash
git clone https://github.com/xiehuanyi/filament-segmentation-2026 && cd filament-segmentation-2026
pip install -r requirements.txt
python scripts/get_weights.py --out WEIGHTS --hf_repo NightNightUp/filament-segmentation-2026               # download + sha256 check
python -m filseg.pipeline --config final/s15.json --split test --data DATA --work WORK --weights WEIGHTS \
    --out submission_s15.csv --gpus 0
```

Add `--bundle inference --pick s16` to `get_weights.py` to fetch only what one pick needs.

## Training data

The models were trained on the MAGFiLO annotations of the competition training frames. The students of s16 were
also trained on 1314 unlabelled GONG H-alpha frames from the public NSO archive: other days than any train or test
frame, and at least 6 h from every test timestamp. Those frames were used for training only, never at inference.

## License

MIT. The pretrained encoder weights we started from (timm / segmentation_models_pytorch), the MAGFiLO
annotations and the GONG frames keep their own terms.
