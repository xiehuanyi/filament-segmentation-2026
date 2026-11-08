"""Dense models: a U-Net (segmentation_models_pytorch) that predicts, for every pixel of a full 2048 x 2048
frame, the probability that an annotator marks it as filament.

A trained fold model is stored as ``model_<tag>.pt`` (EMA weights, state_dict) next to ``args_<tag>.json``
(the train_dense.py arguments: architecture, encoder, input normalisation, mixed-precision dtype, ...),
with ``<tag>`` = ``f0`` ... ``f4`` for fold models (trained without that fold) or ``full``.
"""
import argparse
import json
import os

import numpy as np
import torch
import segmentation_models_pytorch as smp

from .common import H, W

# dihedral test-time augmentation: (number of 90-degree rotations, horizontal flip); the first `tta` are used
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]


def amp_dtype(name):
    return torch.float16 if name == "fp16" else torch.bfloat16


def build(arch, encoder, weights=None, nch=1):
    return getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=nch)


@torch.no_grad()
def predict_full(model, x_np, tta=4, nch=1, dtype=torch.bfloat16):
    """x_np: normalised float32 (H, W) frame. Returns the TTA-averaged probability map (nch, H, W) float32 on cpu."""
    dev = next(model.parameters()).device
    z = torch.from_numpy(x_np).to(dev)[None, None]
    acc = torch.zeros((nch, H, W), device=dev)
    ops = D4[:tta]
    for k, fl in ops:
        v = z
        if fl:
            v = torch.flip(v, dims=[3])
        if k:
            v = torch.rot90(v, k, dims=[2, 3])
        with torch.autocast("cuda", dtype=dtype):
            o = model(v.contiguous(memory_format=torch.channels_last)).float()
        p = torch.sigmoid(o)
        if k:
            p = torch.rot90(p, -k, dims=[2, 3])
        if fl:
            p = torch.flip(p, dims=[3])
        acc += p[0]
    return (acc / len(ops)).cpu().numpy()


def to_u8(p):
    """Probability map -> stored uint8 map (round(p * 255)). All maps on disk are in this format."""
    return np.round(p * 255).astype(np.uint8)


def load_model(model_dir, tag, device="cuda"):
    """Loads model_<tag>.pt with the architecture from args_<tag>.json. Returns (model, args dict)."""
    targs = json.load(open(os.path.join(model_dir, f"args_{tag}.json")))
    nch = 2 if targs.get("sep_w", 0) > 0 else 1
    model = build(targs.get("arch", "Unet"), targs["encoder"], None, nch)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"model_{tag}.pt"), map_location="cpu", weights_only=True))
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    return model, targs


def args_namespace(targs):
    return argparse.Namespace(**targs)
