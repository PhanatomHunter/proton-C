#!/usr/bin/env python3
"""Adversarial fuzz battery for the PHOTON decoder.

A decoder consumes attacker-controlled bytes.  Rules it must keep on ANY
input: never crash with a raw python exception, never hang, never be lured
into a hostile allocation.  Everything must surface as StreamError (or a
clean partial decode).

Cases:
  1. clean baseline decode
  2. truncation sweep (every 1/16th)
  3. 800 random single-byte corruptions across the whole stream
  4. pure garbage inputs
  5. hostile header fields (huge sizes / max frames / bad version / fps)
  6. payload-length bombs (0xFFFFFFFF)
  7. rANS stream with absurd symbol count
  8. deleted middle sections

Per-case wall-clock cap via signal.alarm.  FAIL on: non-StreamError
exceptions, hangs, MemoryError.
"""
import io
import os
import signal
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/home/user/photon")

from photon.guard import StreamError                 # noqa: E402
from photon.codec import preset                       # noqa: E402
from photon.frame import Encoder, Decoder             # noqa: E402
from photon import rans                               # noqa: E402


def make_valid_stream() -> bytes:
    cfg = preset("photon-sd-atto", 640, 360, 30.0)
    enc = Encoder(cfg)
    from photon.transform import rgb_to_yuv
    rng = np.random.default_rng(0)
    for _ in range(6):
        img = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
        enc.push_frame(img)
    return enc.finish()


VALID = make_valid_stream()
HDR = 38  # STREAM_HEADER size on this codec version


def alarm(_sig, _frm):
    raise TimeoutError("case exceeded 8 s")


signal.signal(signal.SIGALRM, alarm)

results = {"streamerror": 0, "ok_or_partial": 0, "valueerror": 0,
           "OTHER": []}


def attempt(label, fn):
    """fn() may return normally or raise; classify the outcome."""
    signal.alarm(8)
    try:
        fn()
        results["ok_or_partial"] += 1
    except StreamError:
        results["streamerror"] += 1
    except TimeoutError:
        results["OTHER"].append((label, "HANG >8s"))
    except MemoryError:
        results["OTHER"].append((label, "MEMORY BOMB"))
    except Exception as e:                    # noqa: BLE001
        results["OTHER"].append((label, f"{type(e).__name__}: {e}"))
    finally:
        signal.alarm(0)


def try_decode(blob):
    d = Decoder(blob)
    n = d.nframes
    for i in range(min(n, 4)):
        _ = d.decode_frame(i)


# ---------------------------------------------------------------- 1 baseline
attempt("baseline", lambda: try_decode(VALID))

# ----------------------------------------------------------- 2 truncations
for cut in range(1, 17):
    pos = len(VALID) * cut // 16
    attempt(f"truncate@{pos}", lambda p=pos: try_decode(VALID[:p]))

# ----------------------------------------------------------- 3 byte flips
rng = np.random.default_rng(1)
ba = bytearray(VALID)
for k in range(800):
    i = int(rng.integers(0, len(ba)))
    flip = bytearray(ba)
    flip[i] ^= int(rng.integers(1, 256))
    w = flip[i]
    attempt(f"flip@{i}^{w:02x}", lambda f=bytes(flip): try_decode(f))

# ------------------------------------------------------------- 4 garbage
for k, n in enumerate((1, 32, 4096, 65536)):
    attempt(f"garbage{n}",
            lambda nn=n: try_decode(rng.bytes(nn)))

# ----------------------------------------------------- 5 hostile headers
def mutate_hdr(field_fn):
    b = bytearray(VALID)
    field_fn(b)
    return bytes(b)


import struct as _s
attempt("ver=99", lambda: try_decode(mutate_hdr(lambda b: b.__setitem__(4, 99))))
attempt("w=h=16383", lambda: try_decode(mutate_hdr(
    lambda b: _s.pack_into(">HH", b, 5, 16383, 16383))))
attempt("nframes=65535", lambda: try_decode(mutate_hdr(
    lambda b: _s.pack_into(">H", b, 13, 65535))))
attempt("fps=1e35", lambda: try_decode(mutate_hdr(
    lambda b: _s.pack_into(">f", b, 9, 1e35))))
attempt("bms=1", lambda: try_decode(mutate_hdr(
    lambda b: b.__setitem__(HDR if False else 16, 1))))

# ------------------------------------------- 6 payload length bombs
bomb = bytearray(VALID)
pos = HDR + 4  # first frame packet: after its 4-byte length
_s.pack_into(">I", bomb, HDR, 0xFFFFFFFF)   # corrupt the frame LENGTH itself
attempt("framelen-max", lambda: try_decode(bytes(bomb)))

b2 = bytearray(VALID)
attempt("consume-past-eof", lambda: try_decode(bytes(b2[:HDR + 20])))

# ----------------------------------- 7 rans stream with insane symbol count
r = rans.pack([1, 2, 3, 4, 5])
bad = bytearray(r)
bad[0] = 0xFF; bad[1] = 0xFF; bad[2] = 0xFF; bad[3] = 0xFF
attempt("rans-maxcount", lambda: rans.unpack(bytes(bad)))

bad2 = bytearray(r)
_q = bytearray(r)
_s.pack_into(">I", _q, len(r) - 4, 0xFFFFFFFF)
attempt("rans-maxplen", lambda: rans.unpack(bytes(_q)))

# ------------------------------------------------- 8 middle section delete
mid = len(VALID) // 2
attempt("hole-middle", lambda: try_decode(VALID[:mid] + VALID[mid + 500:]))

# ------------------------------------------------------------------ report
print("\n==================== FUZZ REPORT ====================")
tot = (results["streamerror"] + results["ok_or_partial"]
       + results["valueerror"] + len(results["OTHER"]))
print(f"cases: {tot}")
print(f"  clean StreamError:   {results['streamerror']}")
print(f"  decoded (ok/partial payload): {results['ok_or_partial']}")
print(f"  other/unexpected:    {len(results['OTHER'])}")
for lbl, e in results["OTHER"][:30]:
    print(f"    [!] {lbl}: {e}")
if results["OTHER"]:
    print("\nFUZZ: FAIL -- decoder must only raise StreamError on hostile input")
    sys.exit(1)
print("\nFUZZ: PASS -- all hostile input handled with clean StreamError / partial decode")
sys.exit(0)
