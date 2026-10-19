#!/usr/bin/env python3
"""Interleaved A/B decode-benchmark: natives on/off, alternating within one
process over many frames, best-of-N pairing.  Kills cross-run noise (this VM
wobbles by ~35%) the way naive sequential runs cannot.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/home/user/photon")
import photon.native  # noqa
import photon.transform as T
from photon.frame import Decoder

blob = open("results/atto_4k50.phtn", "rb").read()
md = json.load(open("results/real4k_photon-atto.json"))["md5_encoder"]

REAL = T.yuv_to_rgb
FFI = T.yuv_to_rgb  # patched to try native first


def set_native(on: bool):
    """on: keep patched yuv_to_rgb.  off: temporarily return None from native."""
    photon.native.yuv_to_rgb_native = (
        photon.native.yuv_to_rgb_native if on else (lambda *a: None))


ORIG_NATIVE = photon.native.yuv_to_rgb_native


def mode(fn):
    global ORIG_NATIVE
    if fn == "native":
        photon.native.yuv_to_rgb_native = ORIG_NATIVE
    else:
        photon.native.yuv_to_rgb_native = lambda *a: None


REPS = 3
BATCH = 40
res = {"native": [], "python": []}
mis = 0
# use *separate* decoders per mode so neither inherits warm state from the other
dec = {m: Decoder(blob) for m in res}
frame_i = 0
for rep in range(REPS):
    for m in ("native", "python"):
        mode(m)
        t0 = time.time()
        for j in range(BATCH):
            idx = frame_i % 150
            f = dec[m].decode_frame(idx)
            if hashlib.md5(f.tobytes()).hexdigest() != md[idx]:
                mis += 1
            del f
            frame_i += 1
        res[m].append((time.time() - t0, BATCH))
for m in res:
    times = sorted(t for t, _ in res[m])
    best = BATCH / times[0]
    med = BATCH / sorted(times)[len(times) // 2]
    print(f"{m:<8} best {best:.2f} f/s   median {med:.2f} f/s   (runs {[f'{BATCH/t:.2f}' for t,_ in res[m]]})")
print("md5 mismatches:", mis)
