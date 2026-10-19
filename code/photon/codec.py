"""
PHOTON-C  ::  Codec
===================
A resolution-recursive video codec for streaming at bitrates that are absurd
relative to the pixel count.  4K at 64 kbit/s is the target operating point --
about 1/300th of what H.264 needs for the same picture and 1/1700th of raw video.

How it spends its bits
----------------------
The central decision is not "how do I code 4K well" but "how much of the 4K do I
transmit at all".  `tx_depth` chooses how many levels of a Laplacian pyramid go
over the wire; everything above that is reconstructed at the decoder by
`photon.synthesis`.

    tx_depth 0 -> only the coarse base        (~tens of kbit/s at 4K)
    tx_depth 1 -> base + one detail band      (~hundreds of kbit/s)
    tx_depth 2 -> base + two bands            (~1-3 Mbit/s)
    tx_depth 3 -> everything transmitted, nothing synthesised

Every transmitted level is coded the way a conventional hybrid codec codes a
frame: inter prediction from the previously *reconstructed* level, per-block
skip, 8x8 DCT residuals, rANS entropy coding.  Turning the knob down therefore
does not change the coding tools, only how much picture they are pointed at, and
at full depth this is simply an ordinary (unoptimised) hybrid codec.

Closed loop
-----------
The encoder reconstructs each frame exactly as the decoder will -- synthesis
included -- and predicts from that reconstruction, never from the source.
Without this, synthesised detail drifts from the encoder's notion of the
reference within a few frames and the stream collapses.  With it the borrowed
detail is self-consistent: what the decoder invents for frame N is precisely what
the encoder assumes for frame N+1.

Causality
---------
Everything the decoder needs it can compute from already-decoded data.  Intra DC
prediction in particular is *causal* (from the left and top neighbours'
reconstructed DC), not from the source plane's neighbours -- a non-causal
predictor is the classic way to build a codec whose encoder and decoder silently
disagree.

Syntax elements
---------------
Four rANS streams per level, each with a frequency table built from that level's
own statistics:

  skip runs   alternating run lengths of the skip map.  A 99%-static 4K frame
              costs 83 bytes for its entire skip map.
  mvd         motion vectors relative to their spatial predictor (left, else
              top), zigzagged.  This is why a smooth pan costs almost nothing.
  positions   zigzag-index deltas between successive nonzero coefficients in a
              block: monotone and small, so nearly free.
  levels      zigzagged quantised coefficient values.

Bitstream
---------
    "PHTN" | version | width | height | fps | nframes | depth | tx_depth |
    per-frame length-prefixed packets
Frames are independently length-prefixed, so the container is streamable and a
decoder can join the stream at any I-frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from photon import rans
from photon import synthesis as synth
try:
    from photon import native
except Exception:  # pragma: no cover - no .so in this environment
    native = None
from photon.guard import StreamError, rd
from photon.entropy import rle_decode, rle_encode, zigzag, unzigzag
import os as _os

# Recon-plane working dtype.  PHOTON_F32=1 demotes the wavelet synthesis
# sweep to float32: the LANCZOS pyramid cost halves (measured 23.6 -> 11.0 ms
# on the full-res sweep, cv2 THREADED on this box).  The causal DC chain stays
# float64 either way -- that is the one accumulator whose bit-pattern the
# closed-loop design depends on.  Because the encoder decodes its own packets
# through exactly these functions for its closed-loop recon, encoder and
# decoder switch dtype in lockstep: the roundtrip stays self-consistent by
# construction (baseline hash values move slightly; correctness does not).
RDT = np.float32 if _os.environ.get("PHOTON_F32", "0") == "1" else np.float64
from photon.motion import (decide_skip, dense_flow, estimate_mv, skip_threshold,
                           warp_plane)
from photon.transform import (BLOCK, UNZIGZAG, ZIGZAG, deblock, forward_dct,
                              from_blocks, inverse_dct, quant_matrix, rgb_to_yuv,
                              to_blocks, yuv_to_rgb)

MAGIC = b"PHTN"
VERSION = 2
# v1 -> v2: frame packets store a refresh mask PER PLANE.  In v1 only the luma
# mask was stored and the decoder applied it to chroma as well, so on any preset
# where chroma carried more detail bands than luma (all the `lean` family, i.e.
# c_tx > y_tx) the extra chroma payloads sat in the stream as dead bytes and the
# intended chroma detail band was never decoded on either side.  Accepted for
# reading: v1 streams are decoded with the old shared-mask semantics so nothing
# written before this fix becomes unreadable.

MV_CLIP = 128           # motion vectors clipped to +-128 px per axis
COEF_CLIP = 511         # quantised coefficient magnitude cap (zigzag -> 1023)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    gop: int = 30               # I-frame interval
    tx_depth: int = 2           # detail bands actually transmitted
    qp: int = 70                # JPEG-style quality factor, 1 (coarse) .. 99 (fine)
    lam: float = 4.0            # rate-distortion trade-off in the motion search
    search: int = 1             # integer refinement radius
    deblock_strength: float = 0.5
    chroma: bool = True
    synth_borrow: float = synth.BORROW_GAIN
    synth_noise: float = synth.NOISE_GAIN
    target_kbps: Optional[float] = None
    motion_max_side: int = 512  # dense-flow working resolution cap
    base_min_side: int = 96     # luma pyramid stops halving at this base size
    chroma_min_side: int = 96   # chroma pyramid stops halving at this base size

    # `base_min_side` is the most consequential knob at extreme bitrates, and it
    # is geometry rather than a quality setting: it decides how much of the frame
    # is transmitted versus reconstructed.  Small means a tiny base with almost
    # everything borrowed forward in time -- very low bitrate, permanently soft.
    # Large means the base carries the frame -- sharper, more bits.  Both travel in
    # the stream header because the decoder cannot infer them: without them the two
    # sides build pyramids of different depths and every band index is off by one.
    #
    # `chroma_min_side` is separate because chroma wants a different answer at
    # different bitrates.  Plain 4:2:0 (a one-level chroma chain, i.e. a very
    # small value) costs 7 dB of ceiling but nothing in bits at high tx_depth;
    # matching it to base_min_side gives chroma a base the same size as luma's,
    # which is what keeps the extreme presets from spending their whole budget on
    # colour -- at 640x360 a 4:2:0 chroma plane is 15x the size of the luma base.

    def padded_size(self) -> Tuple[int, int]:
        """Frame size padded so 8x8 blocking is exact at the top pyramid level.

        Only the top level needs padding.  Halving a multiple of BLOCK and
        rounding back up to BLOCK stays a multiple of BLOCK, so every level below
        is already exact.  An earlier version padded by BLOCK * 2^(depth-1) to
        "guarantee" this, which inflated 1920x1080 to 3584x6144 -- 3.5x the
        pixels, all border replicate, all of it coded.
        """
        return (((self.height + BLOCK - 1) // BLOCK) * BLOCK,
                ((self.width + BLOCK - 1) // BLOCK) * BLOCK)


# name              : (tx_depth, qp, gop, base_min_side, chroma_min_side)
#
# n_levels is deliberately NOT a preset field.  The resolution forces it: the
# pyramid halves until the base reaches base_min_side, so 4K gets more levels than
# 480p from identical settings.  A preset that named n_levels would silently
# disagree with the geometry the frame layer derives.
#
# tx_depth is how deep the transmitted bands go, and it sets the bitrate regime:
#   tx_depth = 0           base only; every band borrowed forward in time.  This
#                          is the extreme-low-bandwidth mode, and it works because
#                          detail bands correlate 0.97-0.99 frame to frame -- not
#                          because anything is being invented from nothing.
#   tx_depth = n_levels-1  every band transmitted: an ordinary hybrid codec, which
#                          is the graceful-degradation end of the same design.
#
# qp is a JPEG-style quality factor; 50 reproduces the standard quantisation
# tables on the base layer, higher is finer.
PRESETS: Dict[str, Tuple[int, int, int, int, int]] = {
    # name                : (tx_depth, qp, gop, base_min_side, chroma_min_side)
    #
    # Calibrated on real 1080p and synthetic 4K (see results/*.json).  The two
    # conclusions the calibration rests on:
    #
    #   * At extreme bitrate, base resolution matters far more than quantisation.
    #     A coarse base can only upscale to a soft image even lossless (a 1/16
    #     base caps around 23-28 dB on real footage), so the cheapest presets
    #     trade base size for fewer bands, not custom-cut quantiser tables.
    #   * Transmitting extra detail BANDS does not pay at these rates.  Adding
    #     band 1 on top of band 0 costs roughly 2x the bytes for +0.1 dB across
    #     every sweep, because inter prediction already carries long-term
    #     texture and the marginal content is off the visible end.  So the
    #     extreme and mid presets are all tx_depth = 0 (base only, refreshed in
    #     time) and they differ only in how large that base is, which is also
    #     why they read off as a clean bitrate ladder.
    #
    # base_min_side is absolute pixels, not a ratio: it does not scale with the
    # frame, which is why the ladder holds its rates as resolution climbs.  The
    # 4K and 1080p families share most settings because the base is the same
    # absolute size either way, and the synthesis engine is doing the work.

    # ---- 4K / UHD ---- ~144 kbit/s at 64x117
    "photon-atto":    (0, 66, 60,  64,  64),  # 4K base only      ~120-200 kbit/s
    "photon-femto":   (0, 70, 60,  96,  96),  # 4K base only      ~250-380 kbit/s
    "photon-nano":    (0, 72, 60, 136, 136),  # 4K base only      ~450-650 kbit/s
    "photon-lean":    (1, 76, 45, 136,  96),  # 4K + band 0       ~700-1100 kbit/s
    "photon-4k":      (3, 78, 30, 270, 136),  # 4K + 3 bands      ~1.5-3   Mbit/s
    # ---- 1080p / Full HD ----
    "photon-hd-atto": (0, 66, 60,  64,  64),  #                    ~90-160  kbit/s
    "photon-hd-femto":(0, 70, 60,  96,  96),  #                    ~180-320 kbit/s
    "photon-hd-nano": (0, 72, 60, 136, 136),  #                    ~350-550 kbit/s
    "photon-hd-lean": (1, 76, 45, 136,  96),  #                    ~600-900 kbit/s
    "photon-hd":      (3, 78, 30, 270, 136),  #                    ~1-2   Mbit/s
    # ---- 480p and below ----
    "photon-sd-atto": (0, 66, 60,  48,  48),  #                    ~20-50   kbit/s
    "photon-sd":      (0, 70, 60,  80,  80),  #                    ~80-180  kbit/s
    "photon-sd-lean": (1, 76, 45,  96,  64),  #                    ~200-450 kbit/s
    # ---- good-internet tier -------------------------------------------------
    # For users whose pipe can carry it: drop the synthesis-heavy knobs, send
    # nearly everything, behave like a traditional codec while keeping the same
    # architecture (wavelet pyramid + causal DC + band refresh).  High tx_depth,
    # full-res-size base, fine QP, short GOP, dense refresh.  Ceiling control is
    # rate control, not the preset: pass --kbps (two-pass) -- e.g. 1000 Mbit/s =
    # --kbps 1000000 -- and PHOTON stays under it with measured PSNR, at the
    # same low compute per frame as the rest of the family.
    "photon-good":    (2, 88, 16, 500, 250),  # 1080p full detail  ~4-10 Mbit/s
    "photon-good-4k": (2, 88, 16, 800, 250),  # 4K full detail     ~10-25 Mbit/s
    "photon-good-sd": (2, 88, 16, 320, 160),  # <=480p full detail ~0.8-1.5 Mbit/s
}


def preset(name: str, width: int, height: int, fps: float = 30.0,
           target_kbps: Optional[float] = None) -> Config:
    txd, qp, gop, bms, cms = PRESETS[name]
    return Config(width=width, height=height, fps=fps, gop=gop, qp=qp,
                  tx_depth=txd, base_min_side=bms, chroma_min_side=cms,
                  target_kbps=target_kbps)


# --------------------------------------------------------------------------- #
#  Packing helpers
# --------------------------------------------------------------------------- #
def _pack_ints(values: Sequence[int]) -> bytes:
    blob = rans.pack([int(v) for v in values])
    return struct.pack(">I", len(blob)) + blob


def _unpack_ints(buf: bytes, off: int) -> Tuple[List[int], int]:
    from .guard import rd
    n = rd(buf, ">I", off, "int-stream length")[0]
    off += 4
    if n == 0:
        return [], off
    syms, _ = rans.unpack(buf[off:off + n])
    return list(syms), off + n


def _pack_idx(idx: Sequence[int]) -> bytes:
    """Code a sorted list of block indices as the gaps between them.

    Raw indices have to go in as gaps, not as themselves.  The rANS alphabet is
    bounded by PROB_SCALE (4096) because every present symbol needs a frequency
    of at least 1 and the frequencies sum to exactly that, and a 4K detail band
    has 32,400 blocks -- so coding indices directly overflows the coder once the
    band gets large enough.  Gaps are almost always 0 or 1, so the alphabet is a
    handful of symbols and the cost is a fraction of a bit per coded block
    instead of thirteen bits.
    """
    gaps: List[int] = []
    prev = 0
    for i in idx:
        i = int(i)
        gaps.append(i - prev)
        prev = i + 1
    return _pack_ints(gaps)


def _unpack_idx(buf: bytes, off: int, n: int) -> Tuple[List[int], int]:
    gaps, off = _unpack_ints(buf, off)
    out: List[int] = []
    prev = 0
    for g in gaps[:n]:
        out.append(prev + int(g))
        prev = prev + int(g) + 1
    return out, off


def _zz(v: int) -> int:
    v = max(-COEF_CLIP, min(COEF_CLIP, int(v)))
    return (-v * 2 - 1) if v < 0 else (v * 2)


def _unzz(u: int) -> int:
    return (-((u + 1) >> 1)) if (u & 1) else (u >> 1)


# --------------------------------------------------------------------------- #
#  Level coders (the causal DC prediction is inline in each -- the standalone
#  helper that used to live here was dead code and has been removed)
# --------------------------------------------------------------------------- #
def _encode_intra_level(cur: np.ndarray, qp: int, level: int, chroma: bool,
                        dbs: float) -> Tuple[bytes, np.ndarray, dict]:
    h, w = cur.shape
    nbh, nbw = h // BLOCK, w // BLOCK
    nb = nbh * nbw
    qm = quant_matrix(level, qp, chroma)

    blk = to_blocks(cur)
    src_dc = blk.mean(axis=(1, 2)).reshape(nbh, nbw)
    # Rounded to an integer BEFORE use, not just before serialisation.  The DC
    # chain is causal, so whatever the first block predicts from propagates into
    # every block that follows: storing this as float32 while computing with
    # float64 introduces a ~1e-6 discrepancy at block (0,0) that the chain then
    # amplifies into a visible error across the whole plane.  Anything that
    # crosses the bitstream boundary has to be quantised to something both sides
    # reproduce exactly.
    global_dc = float(int(round(float(src_dc.mean()))))

    # causal pass: predict DC, quantise the residual, reconstruct immediately so
    # the next block's predictor uses the *reconstructed* value.
    recon_dc = np.zeros((nbh, nbw), dtype=np.float64)
    dc_res = np.zeros((nbh, nbw), dtype=np.float64)
    for by in range(nbh):
        for bx in range(nbw):
            has_l, has_t = bx > 0, by > 0
            if has_l and has_t:
                p = 0.5 * (recon_dc[by, bx - 1] + recon_dc[by - 1, bx])
            elif has_l:
                p = recon_dc[by, bx - 1]
            elif has_t:
                p = recon_dc[by - 1, bx]
            else:
                p = global_dc
            r = src_dc[by, bx] - p
            q = float(qm[0, 0])
            rq = float(np.round(r / q))
            dc_res[by, bx] = rq
            recon_dc[by, bx] = p + rq * q

    pred_blocks = np.repeat(recon_dc.reshape(-1, 1, 1), BLOCK * BLOCK, axis=1)
    pred_blocks = pred_blocks.reshape(nb, BLOCK, BLOCK)

    coefs = forward_dct(blk - pred_blocks)
    # Flatten to (nb, 64) immediately: the DC term is then index 0 of each row
    # rather than element [0, 0] of a 3-D block, and every later stage
    # (zigzag, stream framing, inverse transform) works on one consistent shape.
    cq = np.round(coefs / qm[None]).astype(np.int64).reshape(nb, BLOCK * BLOCK)
    cq[:, 0] = 0                          # DC is carried by dc_res, not here
    nonzero = (cq != 0).any(axis=1)
    cq = cq * nonzero[:, None]

    recon_blocks = inverse_dct(
        (cq * qm.ravel()[None]).astype(np.float64).reshape(nb, BLOCK, BLOCK)
    ) + pred_blocks
    recon = from_blocks(recon_blocks, h, w)
    if dbs > 0:
        recon = deblock(recon, dbs)
    recon = np.clip(recon.astype(RDT), 0, 255)

    coded_idx = np.flatnonzero(nonzero)
    counts, positions, levels = _coef_to_stream(cq[:, ZIGZAG], coded_idx)
    dc_syms = [_zz(int(v)) for v in np.round(dc_res).astype(np.int64).ravel()]

    parts = bytearray()
    parts += struct.pack(">BBHH", 1, int(qp), nbh, nbw)
    parts += struct.pack(">BBf", int(level), int(chroma), float(dbs))
    parts += struct.pack(">fI", global_dc, len(coded_idx))
    parts += _pack_ints(dc_syms)
    parts += _pack_idx(coded_idx.tolist())
    parts += _pack_ints(counts)
    parts += _pack_ints(positions)
    parts += _pack_ints(levels)
    stats = {"intra": True, "n_coded": int(nonzero.sum()), "skip_pct": 0.0}
    return bytes(parts), recon, stats


def _decode_intra_level(buf: bytes, off: int,
                        full_shape: Tuple[int, int]) -> Tuple[np.ndarray, int, dict]:
    _, qp, nbh, nbw = rd(buf, ">BBHH", off, "level header"); off += 6
    level, chroma, dbs = rd(buf, ">BBf", off, "level header"); off += 6
    global_dc, n_coded = rd(buf, ">fI", off, "level header"); off += 8
    nb = nbh * nbw
    # attacker-controlled geometry must agree with the trusted size chain or
    # this frame wants us to allocate a 65535^2-block numpy bomb below.
    if (nbh * BLOCK, nbw * BLOCK) != tuple(full_shape):
        raise StreamError(
            f"level geometry {nbh*BLOCK}x{nbw*BLOCK} != expected {full_shape}")
    if n_coded > nb:
        raise StreamError(f"n_coded {n_coded} exceeds blocks {nb}")
    h, w = nbh * BLOCK, nbw * BLOCK
    qm = quant_matrix(level, qp, bool(chroma))

    dc_syms, off = _unpack_ints(buf, off)
    coded_idx_list, off = _unpack_idx(buf, off, n_coded)
    counts, off = _unpack_ints(buf, off)
    positions, off = _unpack_ints(buf, off)
    levels, off = _unpack_ints(buf, off)
    coded_idx = np.array(coded_idx_list[:n_coded], dtype=np.int64) if n_coded else np.zeros(0, np.int64)
    if len(dc_syms) != nb:
        raise StreamError(f"dc symbol count {len(dc_syms)} != blocks {nb}")
    if coded_idx.size and (coded_idx.max() >= nb or coded_idx.min() < 0):
        raise StreamError(f"coded block index out of range (nb={nb})")

    dc_res = np.array([_unzz(u) for u in dc_syms], dtype=np.float64).reshape(nbh, nbw)

    recon_dc = None
    try:
        from . import native
        recon_dc = native.intra_dc_scan(dc_res, float(global_dc), float(qm[0, 0]))
    except Exception:
        recon_dc = None
    if recon_dc is None:
        recon_dc = np.zeros((nbh, nbw), dtype=np.float64)
        for by in range(nbh):
            for bx in range(nbw):
                has_l, has_t = bx > 0, by > 0
                if has_l and has_t:
                    p = 0.5 * (recon_dc[by, bx - 1] + recon_dc[by - 1, bx])
                elif has_l:
                    p = recon_dc[by, bx - 1]
                elif has_t:
                    p = recon_dc[by - 1, bx]
                else:
                    p = global_dc
                recon_dc[by, bx] = p + dc_res[by, bx] * float(qm[0, 0])

    pred_blocks = np.repeat(recon_dc.reshape(-1, 1, 1), BLOCK * BLOCK, axis=1)
    pred_blocks = pred_blocks.reshape(nb, BLOCK, BLOCK).astype(RDT)

    cq = _stream_to_coef(counts, positions, levels, coded_idx, nb)

    recon_blocks = inverse_dct(
        (cq * qm.ravel()[None]).astype(RDT).reshape(nb, BLOCK, BLOCK)
    ) + pred_blocks
    recon = np.zeros(full_shape, dtype=RDT)
    recon[:h, :w] = from_blocks(recon_blocks, h, w)
    if dbs > 0:
        recon = deblock(recon, dbs)
    recon = np.clip(recon, 0, 255)
    return recon, off, {"intra": True, "skip_pct": 0.0}


# --------------------------------------------------------------------------- #
#  Inter level
# --------------------------------------------------------------------------- #
def _encode_inter_level(cur: np.ndarray, ref: np.ndarray, qp: int, level: int,
                        lam: float, search: int, dbs: float, chroma: bool,
                        motion_max_side: int) -> Tuple[bytes, np.ndarray, np.ndarray, dict]:
    h, w = cur.shape
    nbh, nbw = h // BLOCK, w // BLOCK
    qm = quant_matrix(level, qp, chroma)

    flow = dense_flow(ref, cur, max_side=max(64, motion_max_side >> level))
    seed = flow[:nbh * BLOCK, :nbw * BLOCK].reshape(
        nbh, BLOCK, nbw, BLOCK, 2).mean(axis=(1, 3)).astype(np.float32)
    mv, sad, pred = estimate_mv(cur, ref, seed, BLOCK, lam, search)
    skip = decide_skip(sad, BLOCK, skip_threshold(qp, level, chroma))

    nb = nbh * nbw
    res_blocks = to_blocks(cur[:nbh * BLOCK, :nbw * BLOCK]) - to_blocks(pred)
    coefs = forward_dct(res_blocks)
    cq = np.round(coefs / qm[None]).astype(np.int64).reshape(nb, BLOCK * BLOCK)
    coded = (~skip.reshape(-1)) & ((cq != 0).any(axis=1))
    cq = cq * coded[:, None]

    recon_blocks = inverse_dct(
        (cq * qm.ravel()[None]).astype(np.float64).reshape(nb, BLOCK, BLOCK)
    ) + to_blocks(pred)
    recon = np.zeros_like(cur, dtype=np.float64)
    recon[:nbh * BLOCK, :nbw * BLOCK] = from_blocks(recon_blocks, nbh * BLOCK, nbw * BLOCK)
    if dbs > 0:
        recon = deblock(recon, dbs)
    recon = np.clip(recon.astype(RDT), 0, 255)

    coded_idx = np.flatnonzero(coded)
    counts, positions, levels = _coef_to_stream(cq[:, ZIGZAG], coded_idx)

    mvpred = np.zeros_like(mv)
    mvpred[:, 1:] = mv[:, :-1]
    mvpred[1:, 0] = mv[:-1, 0]
    mvd = np.clip(np.round(mv - mvpred).astype(np.int64), -MV_CLIP, MV_CLIP)
    mvd_syms = [zigzag(int(v)) for v in mvd.ravel()]

    skip_runs = rle_encode(skip.reshape(-1).astype(np.int32).tolist())

    parts = bytearray()
    parts += struct.pack(">BBHH", 0, int(qp), nbh, nbw)
    parts += struct.pack(">BBf", int(level), int(chroma), float(dbs))
    parts += struct.pack(">fI", 0.0, len(coded_idx))
    parts += _pack_ints(skip_runs)
    parts += _pack_idx(coded_idx.tolist())
    parts += _pack_ints(mvd_syms)
    parts += _pack_ints(counts)
    parts += _pack_ints(positions)
    parts += _pack_ints(levels)
    stats = {"intra": False, "n_coded": int(coded.sum()),
             "skip_pct": 100.0 * float(skip.mean())}
    return bytes(parts), recon, flow, stats


def _decode_inter_level(buf: bytes, off: int, ref: np.ndarray,
                        full_shape: Tuple[int, int]) -> Tuple[np.ndarray, int, dict]:
    _, qp, nbh, nbw = rd(buf, ">BBHH", off, "level header"); off += 6
    level, chroma, dbs = rd(buf, ">BBf", off, "level header"); off += 6
    _gdc, n_coded = rd(buf, ">fI", off, "level header"); off += 8
    nb = nbh * nbw
    if (nbh * BLOCK, nbw * BLOCK) != tuple(full_shape):
        raise StreamError(
            f"level geometry {nbh*BLOCK}x{nbw*BLOCK} != expected {full_shape}")
    if n_coded > nb:
        raise StreamError(f"n_coded {n_coded} exceeds blocks {nb}")
    h, w = nbh * BLOCK, nbw * BLOCK
    qm = quant_matrix(level, qp, bool(chroma))

    skip_runs, off = _unpack_ints(buf, off)
    coded_idx_list, off = _unpack_idx(buf, off, n_coded)
    mvd_syms, off = _unpack_ints(buf, off)
    counts, off = _unpack_ints(buf, off)
    positions, off = _unpack_ints(buf, off)
    levels, off = _unpack_ints(buf, off)

    skip_flat = rle_decode(skip_runs, nb)
    if len(skip_flat) != nb:
        raise StreamError(f"skip map size {len(skip_flat)} != blocks {nb}")
    if len(mvd_syms) != nb * 2:
        raise StreamError(f"mvd count {len(mvd_syms)} != 2*blocks {2*nb}")
    coded_idx_i = np.array(coded_idx_list[:n_coded], dtype=np.int64) \
        if n_coded else np.zeros(0, np.int64)
    if coded_idx_i.size and (coded_idx_i.max() >= nb or coded_idx_i.min() < 0):
        raise StreamError(f"coded block index out of range (nb={nb})")
    skip = np.array(skip_flat, dtype=bool).reshape(nbh, nbw)
    mz = np.asarray(mvd_syms, dtype=np.int64)
    mvd = ((mz >> 1) ^ -(mz & 1)).astype(np.float32).reshape(nbh, nbw, 2)
    mv = _spatial_mv_reconstruct(mvd)

    pred = warp_plane(ref.astype(RDT), mv, BLOCK)[:h, :w]

    coded = (~skip).reshape(-1)
    coded_idx = np.array(coded_idx_list[:n_coded], dtype=np.int64) if n_coded else np.zeros(0, np.int64)
    # a block can only be coded if it is not skipped; trust the skip map and
    # intersect so a corrupt count cannot make us walk off the block array
    coded_idx = coded_idx[coded_idx < nb]
    coded_idx = coded_idx[coded[coded_idx]]
    cq = _stream_to_coef(counts, positions, levels, coded_idx, nb)

    recon_blocks = inverse_dct(
        (cq * qm.ravel()[None]).astype(RDT).reshape(nb, BLOCK, BLOCK)
    ) + to_blocks(pred)
    recon = np.zeros(full_shape, dtype=RDT)
    recon[:h, :w] = from_blocks(recon_blocks, h, w)
    if dbs > 0:
        recon = deblock(recon, dbs)
    recon = np.clip(recon, 0, 255)
    return recon, off, {"intra": False, "skip_pct": 100.0 * float(skip.mean())}


def _spatial_mv_reconstruct(mvd: np.ndarray) -> np.ndarray:
    """Haskell-style scan reconstruction, vectorised.
    Row-major: mv[r,c] = (c>0 ? mv[r,c-1] : mv[r-1,0]) + mvd[r,c].
    Row form:        mv[r,0] = cumsum_down(mvd[:,0])[r]
                     mv[r,1:] = the same chain continued rightwards, i.e.
                     cumsum of [mv[r,0], mvd[r,1], mvd[r,2], ...].
    numpy cumsum is per-lane sequential, matching the python loop's add
    ORDER exactly ((mv0+m1)+m2+...), so the f32 sums are bit-identical."""
    col0 = np.cumsum(mvd[:, 0, :], axis=0)
    if mvd.shape[1] == 1:
        out = mvd.copy()
        out[:, 0, :] = col0
        return out
    chained = np.concatenate((col0[:, None, :], mvd[:, 1:, :]), axis=1)
    return np.cumsum(chained, axis=1)


def _peek_intra(buf: bytes, off: int) -> bool:
    return struct.unpack_from(">B", buf, off)[0] == 1


# --------------------------------------------------------------------------- #
#  Coefficient stream framing
# --------------------------------------------------------------------------- #
# The encoder walks the coded blocks and emits (position delta, level) pairs.
# The decoder has to know exactly how many blocks to walk and must stop at the
# same place -- otherwise it consumes the next syntax element's symbols as
# coefficients and the rest of the frame decodes to noise.  So `n_coded` is
# written explicitly and both sides loop over precisely that many blocks.
def _coef_to_stream(cq_zz: np.ndarray, coded_idx: np.ndarray
                    ) -> Tuple[List[int], List[int], List[int]]:
    """Flatten nonzero coefficients into three parallel streams.

    `counts[k]` is the number of nonzeros in coded block k, and it is what tells
    the decoder where one block's coefficients stop and the next block's begin.

    This is load-bearing.  Without an explicit per-block boundary the decoder has
    to infer the end of a block, and any inference that can be wrong (a position
    delta that overruns 64, a run of zero levels) desynchronises the two sides
    from that block onwards: every later position gets attributed to the wrong
    block and the frame decodes to noise while the bitstream still parses
    cleanly.  Counts are tiny integers with a peaked distribution, so under rANS
    they cost well under a bit per coded block.
    """
    positions: List[int] = []
    levels: List[int] = []
    counts: List[int] = []
    for b in coded_idx:
        nz = np.flatnonzero(cq_zz[b])
        counts.append(len(nz))
        prev = 0
        for k in nz:
            positions.append(int(k) - prev)
            prev = int(k) + 1
            levels.append(_zz(int(cq_zz[b][k])))
    return counts, positions, levels


def _stream_to_coef(counts: List[int], positions: List[int], levels: List[int],
                    coded_idx: np.ndarray, nb: int) -> np.ndarray:
    if native is not None:
        c_a = np.asarray(counts, dtype=np.int64)
        p_a = np.asarray(positions, dtype=np.int64)
        l_a = np.asarray(levels, dtype=np.int64)
        ci = np.ascontiguousarray(coded_idx, dtype=np.int64)
        out = native.stream_to_coef(c_a, p_a, l_a, ci, nb)
        if out is not None:
            return out[:, UNZIGZAG]
    cq_zz = np.zeros((nb, BLOCK * BLOCK), dtype=np.int64)
    pi = li = 0
    for j, b in enumerate(coded_idx):
        n = counts[j] if j < len(counts) else 0
        pos = 0
        for _ in range(n):
            if pi >= len(positions) or li >= len(levels):
                break
            pos += positions[pi]
            pi += 1
            if pos >= BLOCK * BLOCK:
                break
            cq_zz[b, pos] = _unzz(levels[li])
            li += 1
            pos += 1
    return cq_zz[:, UNZIGZAG]
