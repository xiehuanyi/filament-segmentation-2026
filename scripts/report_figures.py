"""Figures and numbers for the write-up, from the OOF outputs of a pipeline run (python -m filseg.pipeline --split oof).

    python scripts/report_figures.py stages --config final/s15.json --work WORK --data DATA --out figures/
        PQ of the intermediate stages recomputed from WORK (scorer only / + refiner / + rescorers), -> <name>_stages.json
    python scripts/report_figures.py overlays --pred WORK/oof_s15.csv --work WORK --data DATA --out figures/ [--n 3]
        best and worst OOF frames by per-frame PQ (frames with >= --min_inst instances per annotator record):
        frame, our masks and the annotators' masks
    python scripts/report_figures.py ablation --numbers numbers.json --out figures/ablation.png
        bar chart of a {label: PQ} list
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filseg import assemble  # noqa: E402
from filseg.common import load_gt, load_img, load_meta  # noqa: E402
from filseg.metric import image_counts, pq, summary  # noqa: E402
from filseg.paths import add_path_args, apply_path_args  # noqa: E402
from filseg.submission import by_frame, read_csv  # noqa: E402


def stages(a):
    cfg = json.load(open(a.config))
    tag, rs, tau = cfg["tag"], cfg["rescorer"], cfg["tau"]
    out = {}

    def ev(label, rescore, t, thr=0.5):
        res = assemble.run(tag, "oof", rescore=rescore, w_cnn=rs["w_cnn"] if rescore else 0.0, tau=t, thrs=[thr],
                           workers=a.workers, quiet=True)
        out[label] = {"refined": summary(res[thr])["PQ"], "seed_masks": summary(res["seed"])["PQ"], "tau": t}
        print(f"{label:40s} tau {t:.2f}  seed masks {out[label]['seed_masks']:.4f}  refined {out[label]['refined']:.4f}", flush=True)

    for t in a.taus:
        ev(f"scorer only (LightGBM), tau {t:.2f}", None, t)
    for g in rs["groups"]:
        ev(f"mean(LightGBM, {g})", g, tau)
    ev(f"mean(LightGBM, {rs['combine']}) = final", rs["combine"], tau)
    json.dump(out, open(os.path.join(a.out, f"{cfg['name']}_stages.json"), "w"), indent=1)


def overlays(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pycocotools import mask as mu
    gt = load_gt()
    meta = load_meta()
    pred = by_frame(read_csv(a.pred))
    scores = []
    for s in meta["stems"]:
        if sum(len(g) for g in gt[s]) < a.min_inst * len(gt[s]):  # frames with at least min_inst instances per record
            continue
        scores.append((pq(image_counts(gt[s], pred.get(s, []))), s))
    scores.sort()
    picks = [("worst", s, v) for v, s in scores[:a.n]] + [("best", s, v) for v, s in scores[::-1][:a.n]]
    name = os.path.splitext(os.path.basename(a.pred))[0]
    y0, y1 = 100, 1948  # the solar disk

    def paint(ax, im, masks, colors):
        rgb = np.repeat(im[y0:y1, y0:y1, None].astype(np.float32) / 255.0, 3, 2)
        for m, c in zip(masks, colors):
            m = m[y0:y1, y0:y1].astype(bool)
            rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(c)
        ax.imshow(rgb)
        for m, c in zip(masks, colors):
            ax.contour(m[y0:y1, y0:y1], levels=[0.5], colors=[c], linewidths=0.7)
        ax.axis("off")

    for kind, s, v in picks:
        im = load_img(s)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        axes[0].imshow(im[y0:y1, y0:y1], cmap="gray", vmin=0, vmax=255)
        axes[0].axis("off")
        axes[0].set_title(f"{s} (H-alpha)", fontsize=9)
        rng = np.random.RandomState(0)
        pm = [mu.decode(r) for r in pred.get(s, [])]
        paint(axes[1], im, pm, [tuple(rng.uniform(0.2, 1, 3)) for _ in pm])
        axes[1].set_title(f"ours: {len(pm)} filaments, frame PQ {v:.2f}", fontsize=9)
        cols = [(1, 0.15, 0.15), (0.1, 0.9, 0.95), (1, 0.9, 0.1)]
        gm, gc = [], []
        for i, g in enumerate(gt[s]):
            for r in g:
                gm.append(mu.decode(r))
                gc.append(cols[i % 3])
        paint(axes[2], im, gm, gc)
        axes[2].set_title(f"annotators: {len(gt[s])} record(s) (red / cyan / yellow), "
                          + " + ".join(str(len(g)) for g in gt[s]) + " instances", fontsize=9)
        fig.tight_layout()
        f = os.path.join(a.out, f"{name}_{kind}_{s}.png")
        fig.savefig(f, dpi=110)
        plt.close(fig)
        print(kind, s, f"PQ {v:.3f}", f, flush=True)
    json.dump([dict(kind=k, frame=s, frame_pq=v) for k, s, v in picks], open(os.path.join(a.out, f"{name}_overlays.json"), "w"), indent=1)


def ablation(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(a.numbers))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=["#4c72b0" if not r[0].startswith("s1") else "#c44e52" for r in rows])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(min(vals) - 0.01, max(vals) + 0.006)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0005, yi, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("5-fold OOF PQ")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("cmd", choices=["stages", "overlays", "ablation"])
    ap.add_argument("--config")
    ap.add_argument("--pred")
    ap.add_argument("--numbers")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min_inst", type=float, default=5)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.20, 0.22, 0.24])
    ap.add_argument("--workers", type=int, default=12)
    a = apply_path_args(ap.parse_args())
    if a.cmd != "ablation":
        os.makedirs(a.out, exist_ok=True)
    {"stages": stages, "overlays": overlays, "ablation": ablation}[a.cmd](a)


if __name__ == "__main__":
    main()
