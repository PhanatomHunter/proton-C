#!/usr/bin/env python3
"""Extract cv2's true 8U colour transforms as lookup tables + verify structure."""
import numpy as np
import cv2


def probe(chan):
    y = np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1), (1, 256))
    d = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256), (256, 1))
    mid = np.full((256, 256), 128, np.uint8)
    img = np.stack([y, d, mid], -1) if chan == 'R' else np.stack([y, mid, d], -1)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


R = probe('R')[:, :, 0]
B = probe('B')[:, :, 2]

offR = R[128].astype(np.int32) - 128
predR = np.clip(np.arange(256)[:, None] + offR[None, :], 0, 255).astype(np.uint8)
print("R additive:", np.array_equal(R, predR))

offB = B[128].astype(np.int32) - 128
predB = np.clip(np.arange(256)[:, None] + offB[None, :], 0, 255).astype(np.uint8)
print("B additive:", np.array_equal(B, predB))

cr = np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1), (1, 256))
cb = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256), (256, 1))
img = np.stack([np.full((256, 256), 128, np.uint8), cr, cb], -1)
G = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)[:, :, 1]

a = (G[:, 128].astype(np.int32) - 128); a = a - a[128]
b = (G[128, :].astype(np.int32) - 128); b = b - b[128]
base = int(G[128, 128])
cand = np.clip(base + a[:, None] + b[None, :], 0, 255).astype(np.uint8)
print("G separable @Y=128 :", np.array_equal(G, cand),
      "maxerr", int(np.abs(G.astype(int) - cand.astype(int)).max()))

img2 = np.stack([np.full((256, 256), 200, np.uint8), cr, cb], -1)
G2 = cv2.cvtColor(img2, cv2.COLOR_YCrCb2RGB)[:, :, 1]
predG2 = np.clip(200 - 128 + base + a[:, None] + b[None, :], 0, 255).astype(np.uint8)
print("G Y-shift consistent:", np.array_equal(G2, predG2),
      "maxerr", int(np.abs(G2.astype(int) - predG2.astype(int)).max()))

np.save('/tmp/lut_offR.npy', offR.astype(np.int32))
np.save('/tmp/lut_offB.npy', offB.astype(np.int32))
np.save('/tmp/lut_Ga.npy', a.astype(np.int32))
np.save('/tmp/lut_Gb.npy', b.astype(np.int32))
print("base G:", base)
# print sample values for C baking
print("offR[:8]:", offR[:8].tolist(), "...", offR[120:136].tolist())
print("offB[:8]:", offB[:8].tolist(), "...", offB[120:136].tolist())
print("Ga  [:8]:", a[:8].tolist(), "...", a[120:136].tolist())
print("Gb  [:8]:", b[:8].tolist(), "...", b[120:136].tolist())
