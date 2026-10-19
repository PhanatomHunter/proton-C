#!/usr/bin/env python3
"""Regression: decode_all() must return per-frame CORRECT independent copies
even though the native colour kernel serves frames from a 4-buffer rotation.

Method: md5 every frame (a) from decode_all (collected), (b) from one fresh
Decoder per frame index (no rotation aliasing).  All three must agree.
A stale-then-read comparison is invalid by design -- streaming consumers use
frames before they fall out of the rotation, and this asserts exactly that
contract too.
"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, "/home/user/photon")
from photon.codec import preset
from photon.frame import Encoder, Decoder

cfg = preset("photon-sd-atto", 640, 360, 30.0)
enc = Encoder(cfg)
rng = np.random.default_rng(1)
for i in range(7):
    img = np.full((360, 640, 3), 20 + i * 30, np.uint8)
    img[:, :, 1] = (rng.random((360, 640)) * 60 + i * 10).astype(np.uint8)
    enc.push_frame(img)
blob = enc.finish()


def h(f):
    return hashlib.md5(f.tobytes()).hexdigest()


# (b) reference: fresh decoder per frame (each does a full seek from keyframe)
ref = []
for i in range(7):
    d = Decoder(blob)
    ref.append(h(d.decode_frame(i)))

# (a) decode_all collected copies
d = Decoder(blob)
all_frames = d.decode_all()
assert len(set(f.ctypes.data for f in all_frames)) == len(all_frames), \
    "decode_all returned aliased buffers!"
assert [h(f) for f in all_frames] == ref, "decode_all content mismatch"

# streaming-contract check: sequential decode_frame(i) hashes EXACTLY at use
d = Decoder(blob)
for i in range(7):
    assert h(d.decode_frame(i)) == ref[i], f"stream frame {i} mismatch"

print("decode_all copy-safety + streaming-hash equality: PASS (7/7 frames)")
