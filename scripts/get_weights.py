"""Download the released weights (optional) and verify them against final/weights_manifest.csv.

    python scripts/get_weights.py --out weights --hf_repo OWNER/REPO     # download what is missing, then verify
    python scripts/get_weights.py --out weights                          # only verify an existing folder
    python scripts/get_weights.py --out weights --bundle inference --pick s16   # only what one pick needs

With --hf_repo, files already present with the right size and sha256 are not downloaded again (so a complete
folder needs no network). Exit code 1 if a file is still missing or its sha256 differs.
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
    bad = check(rows, a.out, quiet=bool(a.hf_repo))
    if a.hf_repo and bad:
        from huggingface_hub import snapshot_download
        print(f"downloading {len(bad)} files ({sum(int(r['bytes']) for r in bad) / 1e9:.2f} GB) from {a.hf_repo}", flush=True)
        for r in bad:  # a corrupt file must not be taken as already downloaded
            p = os.path.join(a.out, r["file"])
            if os.path.exists(p):
                os.remove(p)
        snapshot_download(repo_id=a.hf_repo, revision=a.revision, local_dir=a.out, allow_patterns=[r["file"] for r in bad])
        bad = check(rows, a.out)
    print(f"{len(rows) - len(bad)}/{len(rows)} files ok ({sum(int(r['bytes']) for r in rows) / 1e9:.2f} GB) in {a.out}")
    sys.exit(1 if bad else 0)


def check(rows, folder, quiet=False):
    """Rows whose file is missing or differs from the manifest (size, then sha256)."""
    bad = []
    for r in rows:
        p = os.path.join(folder, r["file"])
        if not os.path.exists(p):
            why = "missing"
        elif os.path.getsize(p) != int(r["bytes"]) or sha256(p) != r["sha256"]:
            why = "checksum mismatch"
        else:
            continue
        if not quiet:
            print(why, r["file"])
        bad.append(r)
    return bad


if __name__ == "__main__":
    main()
