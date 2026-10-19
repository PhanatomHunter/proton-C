#!/usr/bin/env python3
"""Fit the exact integer transform cv2 uses inside the 8U YCrCb<->RGB paths.

cv2 internally does one shared rounding:  out = clip((Y<<S) + k*(d-128) + C) >> S
We grid-search (S, k, C) and validate against the FULL 256x256 observed
matrices, so a 0-mismatch fit is proven exact for the whole input space.
"""
import numpy as np
import cv2

SHIFTS = (13, 14, 15, 16)


def fit_1d(obs, coeff, chandep=None):
    """obs[Y, drive] -> find S,k,C with  clip((Y<<S)+k*(d)+C)>>S == obs."""
    Y = np.arange(256, dtype=np.int64)[:, None]
    D = (np.arange(256, dtype=np.int64) - 128)[None, :]
    for S in SHIFTS:
        k0 = int(round(coeff * (1 << S)))
        for k in range(k0 - 4, k0 + 5):
            core = (Y << S) + k * D
            for C in (0, (1 << (S - 1)) - 1, 1 << (S - 1),
                      (1 << (S - 1)) + 1, (1 << S) - 1):
                pred = np.clip((core + C) >> S, 0, 255).astype(np.uint8)
                err = np.count_nonzero(pred - obs)
                if err == 0:
                    return S, k, C
    return None


def probe_1d(in_chan, out_chan):
    y = np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1), (1, 256))
    d = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256), (256, 1))
    mid = np.full((256, 256), 128, np.uint8)
    img = np.stack([y, d, mid], -1) if in_chan == 'Cr' else np.stack([y, mid, d], -1)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)[:, :, {'R': 0, 'B': 2}[out_chan]]


R = probe_1d('Cr', 'R')
B = probe_1d('Cb', 'B')
fr = fit_1d(R, 1.402)
fb = fit_1d(B, 1.772)
print("R fit (S,k,C):", fr, " err0" if fr else " NOT FOUND")
print("B fit (S,k,C):", fb, " err0" if fb else " NOT FOUND")

# G two-term:  clip((Y<<S) - k1*(Cb-128) - k2*(Cr-128) + C) >> S
cr = np.tile(np.arange(256, dtype=np.int64).reshape(256, 1) - 128, (1, 256))
cb = np.tile(np.arange(256, dtype=np.int64).reshape(1, 256) - 128, (256, 1))
img = np.stack([np.full((256, 256), 128, np.uint8),
                (cr + 128).astype(np.uint8), (cb + 128).astype(np.uint8)], -1)
G = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)[:, :, 1]
found = None
for S in SHIFTS:
    k10 = int(round(0.344136 * (1 << S))); k20 = int(round(0.714136 * (1 << S)))
    for k1 in range(k10 - 3, k10 + 4):
        for k2 in range(k20 - 3, k20 + 4):
            coreG = (np.int64(128) << S) - k1 * cb - k2 * cr
            for C in (0, (1 << (S - 1)) - 1, 1 << (S - 1), (1 << S) - 1):
                pred = np.clip((coreG + C) >> S, 0, 255).astype(np.uint8)
                if np.array_equal(pred, G):
                    found = (S, k1, k2, C)
    if found:
        break
print("G fit (S,k_cb,k_cr,C):", found, " err0" if found else " NOT FOUND")
