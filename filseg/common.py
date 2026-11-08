"""Image I/O and input normalisation shared by every stage.

GONG H-alpha frames are 2048 x 2048 grayscale JPEGs with the solar disk centred at ~(1024, 1023) and a radius of
~900 px. Two input normalisations are used by the dense models:
  z     z-score over the disk, constant -4 off-disk                 (default)
  flat  flat-field: divide out the centre-to-limb profile and a smooth large-scale background, unit variance
"""
import json
import os
import pickle

import cv2
import numpy as np

from .paths import P

H = W = 2048


def img_path(stem, test=False):
    """Path of a frame. Test frames live in $FIL_IMAGES (default $FIL_DATA/test/test_images)."""
    return os.path.join(P.images if test else P.train_images, f"{stem}.jpeg")


def load_img(stem, test=False):
    im = cv2.imread(img_path(stem, test), cv2.IMREAD_GRAYSCALE)
    assert im is not None and im.shape == (H, W), stem
    return im


def list_stems(folder):
    """Sorted frame names (without extension) of the .jpeg/.jpg files in a folder."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg")))


def test_stems():
    return list_stems(P.images)


def load_meta():
    return json.load(open(f"{P.cache}/meta.json"))


def load_gt():
    """{stem: [[rle, ...] for each annotator record]} written by prep.py."""
    return pickle.load(open(f"{P.cache}/gt.pkl", "rb"))


_YY, _XX = np.mgrid[:H, :W].astype(np.float32)
_R = np.hypot(_YY - 1024.1, _XX - 1023.1)
_RBIN = np.minimum((_R / 2).astype(np.int32), 600)
DISK_R = 899.0


def normalize(im, mode="z"):
    """uint8 frame -> float32 network input."""
    if mode == "flat":
        return normalize_flat(im)
    disk = _R < DISK_R
    v = im[disk].astype(np.float32)
    m, s = float(v.mean()), max(float(v.std()), 1.0)
    x = (im.astype(np.float32) - m) / s
    x[~disk] = -4.0  # off-disk constant
    return x


def normalize_flat(im):
    """Flat-field: divide out the centre-to-limb profile (per-image radial mean) and a smooth large-scale
    background (clouds, gradients); returns a contrast map scaled to ~unit variance, 0 off-disk."""
    f = im.astype(np.float32)
    disk = _R < DISK_R - 3
    rb = _RBIN[disk]
    v = f[disk]
    cnt = np.bincount(rb, minlength=601)
    s = np.bincount(rb, weights=v, minlength=601)
    prof = s / np.maximum(cnt, 1)
    last = int(rb.max())
    prof[last + 1:] = prof[last]
    prof = np.maximum(prof, 1.0)
    c1 = f / prof[_RBIN]
    c1[~disk] = 1.0
    small = cv2.resize(c1, (256, 256), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(np.clip(small * 100, 0, 255).astype(np.uint8), 15).astype(np.float32) / 100.0
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
    c2 = c1 / np.maximum(bg, 0.3) - 1.0
    sd = max(float(c2[disk].std()), 1e-3)
    x = c2 / sd
    x[~disk] = 0.0
    return np.clip(x, -8, 8).astype(np.float32)
