"""Download the released weights (optional) and verify them against final/weights_manifest.csv.

    python scripts/get_weights.py --out weights --hf_repo OWNER/REPO     # download a Hugging Face model repo, then verify
    python scripts/get_weights.py --out weights                          # only verify an existing folder
    python scripts/get_weights.py --out weights --bundle inference --pick s16   # check only what one pick needs

Exit code 1 if a file is missing or its sha256 differs.
"""
import argparse
import csv
import hashlib
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--hf_repo", default=None, help="Hugging Face repo id holding the weights folder layout")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--manifest", default="final/weights_manifest.csv")
    ap.add_argument("--bundle", choices=["inference", "oof", "all"], default="all")
    ap.add_argument("--pick", default=None, help="s15 or s16: only the files this pick uses")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest)))
    if a.bundle != "all":
        rows = [r for r in rows if r["bundle"] == a.bundle]
    if a.pick:
        rows = [r for r in rows if a.pick in r["used_by"].split("+")]
    if a.hf_repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=a.hf_repo, revision=a.revision, local_dir=a.out, allow_patterns=[r["file"] for r in rows])
    bad = 0
    for r in rows:
        p = os.path.join(a.out, r["file"])
        if not os.path.exists(p):
            print("missing", r["file"])
            bad += 1
        elif os.path.getsize(p) != int(r["bytes"]) or sha256(p) != r["sha256"]:
            print("checksum mismatch", r["file"])
            bad += 1
    print(f"{len(rows) - bad}/{len(rows)} files ok ({sum(int(r['bytes']) for r in rows) / 1e9:.2f} GB)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
