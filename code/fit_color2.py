#!/usr/bin/env python3
"""Stage 2: extract cv2's per-index LUTs for R and B; verify additive model."""
import numpy as np
import cv2


def probe(in_chan):
    y = np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1), (1, 256))
    d = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256), (256, 1))
    mid = np.full((256, 256), 128, np.uint8)
    img = np.stack([y, d, mid], -1) if in_chan == 'Cr' else np.stack([y, mid, d], -1)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


R = probe('Cr')[:, :, 0]          # R[Y, Cr]
B = probe('Cb')[:, :, 2]          # B[Y, Cb]


def extract_lut(obs):
    """offset(d) = obs[Y0, d]-Y0 from whichever of Y0=0/255 avoids clipping."""
    hi = obs[255].astype(np.int32) - 255     # valid for negative offsets
    lo = obs[0].astype(np.int32)             # valid for positive offsets
    lut = np.where(hi != 0, hi, lo)          # hi==0 -> offset >= 0 (or exactly -255)
    return lut


lutR = extract_lut(R)
lutB = extract_lut(B)
predR = np.clip(np.arange(256)[:, None] + lutR[None, :], 0, 255).astype(np.uint8)
predB = np.clip(np.arange(256)[:, None] + lutB[None, :], 0, 255).astype(np.uint8)
print("R = clip(Y + lutR[Cr]) exact:", np.array_equal(R, predR))
print("B = clip(Y + lutB[Cb]) exact:", np.array_equal(B, predB))

# what rounding rule generated the LUTs?
d = np.arange(256) - 128.0
for name, lut, c in (("R", lutR, 1.402), ("B", lutB, 1.772)):
    fl = np.floor(c * d); rd = np.round(c * d); ce = np.ceil(c * d); tr = (c * d).astype(np.int64)
    for nm, cand in (("floor", fl), ("round", rd), ("ceil", ce), ("trunc", tr)):
        print(f"  {name}: {nm:<5} match={np.array_equal(lut, cand)}")
print("lutR[126:131] =", lutR[126:131].tolist(), " (expected round/trunc of 1.402*(-2..2))")
print("lutR saved:", lutR.dtype, lutR.min(), lutR.max())
np.save('/tmp/lutR.npy', lutR)
np.save('/tmp/lutB.npy', lutB)

# G from stage 1: (Y<<13 - 2818*(Cb-128) - 5849*(Cr-128) + 4096) >> 13, verify ALL Y
img = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 256 * 256, axis=1)
# build (Y, Cr, Cb) full cube 256^3 in slabs to check exhaustively
ok_all = True
for y0 in range(0, 256, 32):
    Yv = np.arange(y0, y0 + 32, dtype=np.int64)
    g_all = np.empty((32, 256, 256), np.uint8)
    ref_all = np.empty((32, 256, 256), np.uint8)
    for j, y in enumerate(Yv):
        yv = np.full((256, 256), y, np.uint8)
        cr8 = np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1), (1, 256))
        cb8 = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256), (256, 1))
        ref_all[j] = cv2.cvtColor(np.stack([yv, cr8, cb8], -1),
                                  cv2.COLOR_YCrCb2RGB)[:, :, 1]
    cr = np.tile((np.arange(256, dtype=np.int64) - 128).reshape(256, 1), (1, 256))
    cb = np.tile((np.arange(256, dtype=np.int64) - 128).reshape(1, 256), (256, 1))
    for j, y in enumerate(Yv):
        g_all[j] = np.clip((y << 13) + (-2818 * cb - 5849 * cr + 4096) >> 13, 0, 255).astype(np.uint8)
    ok_all &= np.array_equal(g_all, ref_all)
print("G S=13 fit exact for ALL (Y,Cr,Cb):", ok_all)
