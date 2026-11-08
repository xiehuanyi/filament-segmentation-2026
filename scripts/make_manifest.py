"""Writes final/weights_manifest.csv: every file of a weights folder with its size, sha256 and the picks that use it.

    python scripts/make_manifest.py WEIGHTS [--location PREFIX] [--out final/weights_manifest.csv]

bundle = inference (needed to reproduce the submissions) or oof (only needed to reproduce the OOF score without
retraining). --location is written in front of every relative path (where the file is hosted).
"""
import argparse
import csv
import hashlib
import json
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def usage(rel, cfgs):
    parts = rel.split("/")
    users, bundle = [], "inference"
    for c in cfgs:
        tag = c["tag"]
        runs = {m["run"] for m in c["maps"].values()}
        if parts[0] == "dense" and parts[1] in runs:
            users.append(c["name"])
        elif parts[0] in ("scorer", "rescorer") and parts[1].split("_")[0].split(".")[0] == tag:
            users.append(c["name"])
        elif parts[0] == "refiner" and parts[1].startswith(f"{tag}_refiner"):
            users.append(c["name"])
            if "_f" in parts[1]:
                bundle = "oof"
    return "+".join(users) or "-", bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--location", default="")
    ap.add_argument("--configs", nargs="+", default=["final/s15.json", "final/s16.json"])
    ap.add_argument("--out", default="final/weights_manifest.csv")
    a = ap.parse_args()
    cfgs = [json.load(open(c)) for c in a.configs]
    rows = []
    for root, _, files in os.walk(a.weights):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, a.weights)
            used, bundle = usage(rel, cfgs)
            rows.append(dict(file=rel, bytes=os.path.getsize(p), sha256=sha256(p), used_by=used, bundle=bundle,
                             location=f"{a.location.rstrip('/')}/{rel}" if a.location else ""))
    rows.sort(key=lambda r: r["file"])
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for b in ("inference", "oof"):
        sel = [r for r in rows if r["bundle"] == b]
        print(f"{b}: {len(sel)} files, {sum(r['bytes'] for r in sel) / 1e9:.2f} GB")
    for name in [c["name"] for c in cfgs]:
        sel = [r for r in rows if name in r["used_by"].split("+") and r["bundle"] == "inference"]
        print(f"{name} inference: {len(sel)} files, {sum(r['bytes'] for r in sel) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
