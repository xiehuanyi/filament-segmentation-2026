"""Upper bound of the selection stage: keep a candidate iff its TRUE annotator-averaged matched IoU q > PQ*/2
(solved by fixed-point iteration), at each single threshold level and over the nesting forest of all levels.

    python experiments/oracle.py --tag ensB --work WORK      # needs cands/<tag>_oof.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filseg.common import load_gt  # noqa: E402
from filseg.paths import P, add_path_args, apply_path_args  # noqa: E402


def select_tree(cands, value):
    """Disjoint candidates (an antichain of the nesting forest) maximising the sum of positive values."""
    n = len(cands)
    children = [[] for _ in range(n)]
    roots = []
    for j, c in enumerate(cands):
        (children[c["parent"]] if c["parent"] >= 0 else roots).append(j)
    best = np.zeros(n)
    take = np.zeros(n, bool)
    for j in reversed(range(n)):  # children have larger indices than their parents
        cs = sum(best[k] for k in children[j])
        own = max(value[j], 0.0)
        if value[j] > 0 and own >= cs:
            best[j], take[j] = own, True
        else:
            best[j] = cs
    out, stack = [], list(roots)
    while stack:
        j = stack.pop()
        if take[j]:
            out.append(j)
        else:
            stack.extend(children[j])
    return out


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tag", default="ensB")
    a = apply_path_args(ap.parse_args())
    gt = load_gt()
    C = pickle.load(open(f"{P.cands}/{a.tag}_oof.pkl", "rb"))
    N = sum(len(g) for s in C for g in gt[s])
    for t in sorted({c["t"] for cs in C.values() for c in cs}) + ["tree"]:
        pq = 0.4
        for _ in range(30):
            S = M = 0.0
            for cs in C.values():
                if t == "tree":
                    sel = [cs[j] for j in select_tree(cs, np.array([c["q"] - pq / 2 for c in cs]))] if cs else []
                else:
                    sel = [c for c in cs if c["t"] == t and c["q"] > pq / 2]
                S += sum(c["s"] for c in sel)
                M += sum(c["m"] for c in sel)
            new = S / (0.5 * N + 0.5 * M)
            if abs(new - pq) < 1e-6:
                break
            pq = new
        print(f"oracle level {t}: PQ {pq:.4f}", flush=True)


if __name__ == "__main__":
    main()
