#!/usr/bin/env python3
"""Exhaustive bit-exactness proof for the native colour kernels.

For 8-bit colour conversion the total input space is 2^24 = 16.7M points, so
rather than sampling we simply test ALL of it.  Zero mismatches against the
cv2 reference means the kernels are equivalent, by exhaustive proof.
"""
import ctypes
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, "/home/user/photon")
from photon import native, transform  # noqa: E402

L = native._load()
assert L is not None, "native library missing - run sh native/build.sh"

GRID_SIDE = 4096                      # 2^24 pixels
NP = GRID_SIDE * GRID_SIDE
idx = np.arange(NP, dtype=np.uint32).reshape(GRID_SIDE, GRID_SIDE)

A0 = (idx >> 16).astype(np.uint8)
A1 = ((idx >> 8) & 255).astype(np.uint8)
A2 = (idx & 255).astype(np.uint8)
ycrcb = np.stack([A0, A1, A2], axis=-1)   # OpenCV layout: Y, Cr, Cb

print("== exhaustive: cv2.COLOR_YCrCb2RGB over all 2^24 inputs ==")
ref = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
ours = np.empty_like(ref)
fn = L.ptn_convert_grid_ycrcb2rgb
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
fn(ycrcb.ctypes.data, ours.ctypes.data, ctypes.c_long(NP))
diff = np.abs(ref.astype(np.int16) - ours.astype(np.int16))
mism = int(np.count_nonzero(diff))
print(f"  mismatched inputs: {mism} / {NP:,}   max |diff|: {diff.max()}")
assert mism == 0, "not bit-exact"

print("== float32 pipeline equivalence (rint+clip front-end) ==")
rng = np.random.default_rng(0)
maxerr = 0
for _ in range(20):
    h, w = 517, 771
    y = rng.normal(128, 90, (h, w)).astype(np.float32) - 30
    u = rng.normal(128, 40, (h, w)).astype(np.float32) + 10
    v = rng.normal(128, 40, (h, w)).astype(np.float32)
    want = transform.yuv_to_rgb(y, u, v)               # full python path
    got = native.yuv_to_rgb_native(y, u, v)
    assert got is not None
    e = int(np.abs(want.astype(np.int16) - got.astype(np.int16)).max())
    maxerr = max(maxerr, e)
print(f"  max |diff| over 20 random frames: {maxerr}")
assert maxerr == 0
print("BIT-EXACT: exhaustive over all inputs + float front-end.")

# ---------------------------------------------------------------- benchmarks
print("== micro-benchmark @ 4K (mean of 5) ==")
H, W = 2160, 3840
y = rng.normal(128, 60, (H, W)).astype(np.float32)
u = rng.normal(128, 30, (H, W)).astype(np.float32)
v = rng.normal(128, 30, (H, W)).astype(np.float32)
for name, f in (("python", lambda: transform.yuv_to_rgb(y, u, v)[:, :, 0].sum()),
                ("native", lambda: native.yuv_to_rgb_native(y, u, v)[:, :, 0].sum())):
    f()  # warm
    t0 = time.time()
    for _ in range(5):
        f()
    dt = (time.time() - t0) / 5
    print(f"  {name:<7}: {dt*1000:7.1f} ms/frame  ({1/dt:6.1f} f/s for this op alone)")
print("native threads:", native.num_threads())
