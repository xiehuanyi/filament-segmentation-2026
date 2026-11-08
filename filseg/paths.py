"""Filesystem layout of a run. Nothing is hard-coded: every location comes from an environment variable
or from the common CLI flags (``--data``, ``--work``, ``--weights``, ``--images``) that each entry point accepts.

  FIL_DATA     competition data root, i.e. the folder that holds ``train/`` and ``test/`` as downloaded from Kaggle
               (``train/train_images/*.jpeg``, ``train/MAGFiLO_1.0_Annotations_kaggle2026_train.json``,
               ``test/test_images/*.jpeg``)
  FIL_WORK     working directory; every stage writes below it:
                 cache/      meta.json, gt.pkl, consensus targets (prep.py)
                 runs/<run>/ dense-model checkpoints and args (train_dense.py)
                 maps/<map>/ probability maps: oof/<stem>.png and test_f<k>/<stem>.png (uint8, p * 255)
                 cands/      candidate tables <tag>_<split>.pkl
                 pp/         LightGBM scorer <tag>_lgb{0,1}.txt, <tag>.json, cross-fitted OOF scores <tag>_oofq.npz
                 refine/     refiner / rescorer crops, predictions and checkpoints
  FIL_WEIGHTS  released weights (layout: final/weights_manifest.csv); used by the inference pipeline
  FIL_IMAGES   folder with the frames to predict (default: $FIL_DATA/test/test_images)
"""
import argparse
import os


class _Paths:
    @property
    def data(self):
        return os.environ.get("FIL_DATA", "data")

    @property
    def work(self):
        return os.environ.get("FIL_WORK", "work")

    @property
    def weights(self):
        return os.environ.get("FIL_WEIGHTS", "weights")

    @property
    def images(self):
        return os.environ.get("FIL_IMAGES", os.path.join(self.data, "test", "test_images"))

    @property
    def train_images(self):
        return os.path.join(self.data, "train", "train_images")

    @property
    def train_json(self):
        return os.path.join(self.data, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    @property
    def cache(self):
        return os.path.join(self.work, "cache")

    @property
    def runs(self):
        return os.path.join(self.work, "runs")

    @property
    def maps(self):
        return os.path.join(self.work, "maps")

    @property
    def cands(self):
        return os.path.join(self.work, "cands")

    @property
    def pp(self):
        return os.path.join(self.work, "pp")

    @property
    def refine(self):
        return os.path.join(self.work, "refine")


P = _Paths()


def add_path_args(ap: argparse.ArgumentParser):
    """Adds --data/--work/--weights/--images; values override the environment variables."""
    g = ap.add_argument_group("paths (default: $FIL_DATA, $FIL_WORK, $FIL_WEIGHTS, $FIL_IMAGES)")
    g.add_argument("--data", default=None)
    g.add_argument("--work", default=None)
    g.add_argument("--weights", default=None)
    g.add_argument("--images", default=None)
    return ap


def apply_path_args(a):
    """Exports the path flags as environment variables so that worker processes see the same layout."""
    for flag, env in (("data", "FIL_DATA"), ("work", "FIL_WORK"), ("weights", "FIL_WEIGHTS"), ("images", "FIL_IMAGES")):
        v = getattr(a, flag, None)
        if v:
            os.environ[env] = os.path.abspath(v)
    return a
