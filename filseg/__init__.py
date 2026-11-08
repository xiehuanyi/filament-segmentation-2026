"""filseg: solar filament instance segmentation on GONG H-alpha frames (Solar Filament Segmentation Challenge 2026).

Stages (one module each; run as ``python -m filseg.<module>``):
  prep          COCO annotations -> ground truth, soft consensus targets, day-grouped folds
  gong.fetch    unlabelled GONG frames from the NSO archive (>= 6 h from any test frame, other days)
  gong.screen   quality screen of those frames against the training-set statistics
  teacher       teacher probability maps on the unlabelled frames -> distillation targets
  train_dense   U-Net training (one fold), OOF and test probability maps
  infer_dense   probability maps from saved fold models
  candidates    nested connected components of the ensemble map + descriptors
  scorer        LightGBM hurdle scorer of each candidate's expected matched IoU
  refiner       crop U-Net that re-draws each kept candidate
  rescorer      CNN scorer on the refiner crops
  assemble      keep rule + pixel-disjoint paste -> OOF PQ or submission CSV
  pipeline      everything above for one final configuration (final/s15.json, final/s16.json)
  evaluate      PQ, IoU / Dice distributions and 1:n / n:1 matching counts (host self-evaluation notebook)
"""
