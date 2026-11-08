"""Fetch unlabelled GONG H-alpha frames from the public NSO archive (training use only).

Allowed by the hosts (forum threads 744047, 748045): unlabelled GONG frames may be used for training, never at
inference, and never within +-1 hour of a test observation. This script stays much further away:
  * only days that are neither training nor test days, between 2011-01-01 and 2022-12-31;
  * every fetched frame is >= 6 h from every test timestamp;
  * frames are taken at the same GONG site and time of day as a random training frame shifted by 2-40 days.
Images are stored as grayscale JPEG with everything outside r = 960 px blanked (the archive JPEGs carry an NSO
text banner off-disk; the competition images do not). One request at a time, >= 1 s apart.

    python -m filseg.gong.fetch --n 1500 --out PSEUDO_DIR          # needs $FIL_WORK/cache/meta.json (prep)
Writes PSEUDO_DIR/images/<stem>.jpg and PSEUDO_DIR/map.json {fetched stem: source training stem}.
"""
import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.request

import cv2
import numpy as np

from ..common import H, W, load_meta
from ..paths import add_path_args, apply_path_args

BASE = "https://gong2.nso.edu/HA/hag"
MIN_HOURS_FROM_TEST = 6
_last = [0.0]
_RR = np.hypot(*np.mgrid[:H, :W].astype(np.float32) - 1024)


def get(url, tries=4):
    for k in range(tries):
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "filament-research/1.0 (academic)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print("http", e.code, url, flush=True)
            if e.code in (403, 429):
                time.sleep(60 * (k + 1))
        except Exception as e:  # noqa: BLE001
            print("err", e, url, flush=True)
            time.sleep(5 * (k + 1))
    return None


_listing = {}


def frames_of_day(day):
    if day not in _listing:
        b = get(f"{BASE}/{day[:6]}/{day}/")
        found = sorted(set(re.findall(rb'href="(\d{14}[A-Z]h)\.jpg"', b or b""))) if b else []
        _listing[day] = [x.decode() for x in found]
    return _listing[day]


def ts(stem):
    return dt.datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def save(stem, out):
    path = f"{out}/images/{stem}.jpg"
    if os.path.exists(path):
        return True
    b = get(f"{BASE}/{stem[:6]}/{stem[:8]}/{stem}.jpg")
    if not b:
        return False
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape != (H, W):
        return False
    im[_RR > 960] = 0
    cv2.imwrite(path + ".tmp.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    os.replace(path + ".tmp.jpg", path)
    return True


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = apply_path_args(ap.parse_args())
    os.makedirs(f"{a.out}/images", exist_ok=True)
    meta = load_meta()
    train, test = meta["stems"], meta["test_stems"]
    test_t = np.array([ts(s).timestamp() for s in test])
    bad_days = {s[:8] for s in train} | {s[:8] for s in test}

    def far_from_test(t):
        return np.abs(test_t - t.timestamp()).min() >= MIN_HOURS_FROM_TEST * 3600

    mp_path = f"{a.out}/map.json"
    mp = json.load(open(mp_path)) if os.path.exists(mp_path) else {}
    rng = np.random.RandomState(a.seed)
    tries = 0
    while len(mp) < a.n and tries < a.n * 4:
        tries += 1
        src = train[rng.randint(len(train))]
        shift = int(rng.choice([-1, 1]) * rng.randint(2, 41))
        t0 = ts(src) + dt.timedelta(days=shift)
        day = t0.strftime("%Y%m%d")
        if day in bad_days or not ("20110101" <= day <= "20221231"):
            continue
        fr = [f for f in frames_of_day(day) if f[14:] == src[14:]]  # same site
        if not fr:
            continue
        best = min(fr, key=lambda f: abs((ts(f) - t0).total_seconds()))
        if best in mp or abs((ts(best) - t0).total_seconds()) > 3600 or not far_from_test(ts(best)):
            continue
        if save(best, a.out):
            mp[best] = src
            if len(mp) % 25 == 0:
                json.dump(mp, open(mp_path, "w"))
                print(len(mp), "frames", flush=True)
    json.dump(mp, open(mp_path, "w"))
    print("done", len(mp), flush=True)


if __name__ == "__main__":
    main()
