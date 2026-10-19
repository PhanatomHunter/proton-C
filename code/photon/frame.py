"""
PHOTON-C  ::  Frame coding, container, rate control
===================================================

The two measurements this design is built on
--------------------------------------------
Everything here follows from two numbers measured on real footage, because the
obvious design is wrong and only measurement says so.

1.  **Temporal detail transfer is near-perfect.**  Consecutive frames' detail
    bands correlate at 0.97-0.999, and the previous frame's band, scaled by a
    single fitted gain, predicts the current one with 12-29 dB less MSE than the
    band itself carries.  High spatial-frequency content in natural video is
    overwhelmingly static texture that only appears to move.

2.  **Spatial detail transfer is worthless.**  An upsampled lower band predicts
    the band above it with correlation 0.03-0.13, worth +0.0 to +0.1 dB.  A
    two-term predictor using the base's own Laplacian does no better.  Detail
    cannot be extrapolated across scales.  Sharpening a coarse image does not
    invent texture, and any "synthesis from the base alone" is decoration.

Put together: **detail can only come from a frame that transmitted it.**  A GOP
must carry the detail at least once, after which every frame borrows it forward
at essentially zero cost.  That is what makes the bitrate arithmetic work, and it
dictates the progressive refresh schedule below.

Progressive refresh
-------------------
A conventional keyframe codes every band intra in one frame.  At 4K that is
megabytes in a single frame -- a spike that breaks streaming, which is the use
case.  So PHOTON-C spreads it: frame 0 codes the base intra, frame 1 adds band 0
intra, frame 2 adds band 1.  Peak frame size stays near steady state, the stream
is decodable from frame 0 onward, and detail ramps in over a handful of frames
instead of arriving in a spike.  Bands not yet available are synthesised -- a
graceful softness, not an error.

One plane abstraction
---------------------
Luma and both chroma planes go through the same `_Plane` object with different
geometry.  They were separate code paths originally, and the two disagreeing is
what produced the worst bugs in this codec: chroma derived its pyramid depth from
the luma chain, so at deep luma pyramids the chroma base landed at 1/16 of the
frame and cost 7 dB; fixed to plain 4:2:0 it then became 15x larger than the luma
base at extreme presets and ate the entire bitrate budget.  Neither is right for
all presets.  The correct answer is that chroma has its own depth knob
(`chroma_min_side`), and the only way to keep that from being a third source of
divergence is for both to run the same code.

Pyramid indexing
----------------
    shapes[0]     base,           coarsest
    shapes[k + 1] detail band k
    shapes[-1]    full resolution for that plane

`tx_depth` is the deepest band ever transmitted.  0 means base only -- for 4K a
72x120 luma base carries the whole frame -- and n_levels-1 transmits everything,
degrading to an ordinary hybrid codec.

Closed loop
-----------
Band state is stored explicitly (`self.bands[k]`), never recovered by
re-decomposing the reconstruction: rebuilding a pyramid from an already-collapsed
pyramid does not hand back the same bands.  Encoder and decoder both drive the
same `_Reconstructor`, so the reference the encoder predicts from is by
construction the reconstruction the decoder will produce.

Container
---------
    "PHTN" | ver | w | h | fps | nframes | n_levels | tx_depth
           | base_min_side | chroma_min_side
    then per frame: length(u32) | packet
Frames are independently length-prefixed, so a decoder can stream frame by frame,
can seek, and can join at any keyframe.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from photon import synthesis as synth
from photon.codec import (MAGIC, VERSION, Config, RDT,
                          _decode_inter_level, _decode_intra_level,
                          _encode_inter_level, _encode_intra_level, _peek_intra)
from photon.motion import dense_flow
from photon.transform import BLOCK, rgb_to_yuv, yuv_to_rgb, l4_up2_add

# plane-parallel decode pool: the three pyramid reconstructions are independent
# until yuv_to_rgb (each _Plane owns its own base/bands); cv2/ctypes release the
# GIL, so this is real parallelism on multicore.  PHOTON_PLANES=0 = sequential
# (debug aid for streaming-order determinism).
from concurrent.futures import ThreadPoolExecutor
_POOL = None
try:
    import os as _os
    # On <=2 cores the pool only adds scheduling overhead around already-OMP
    # kernels; measured within noise.  Enable by default from 4 cores up.
    # PHOTON_PLANES=0 forces sequential (debug), =1 forces parallel.
    _PLANES_ENV = _os.environ.get("PHOTON_PLANES", "auto")
    _PLANES_OK = ((_PLANES_ENV == "auto" and (_os.cpu_count() or 1) >= 4)
                  or _PLANES_ENV == "1")
    _POOL = (ThreadPoolExecutor(max_workers=3, thread_name_prefix="p-plane")
             if _PLANES_OK else None)
finally:
    pass

STREAM_FORMAT = ">4sBHHfHBBHBBH"        # ..., base_min_side, chroma_min_side, qp, gop
STREAM_HEADER = struct.calcsize(STREAM_FORMAT)

# Every offset advance in this module comes from struct.calcsize rather than being
# counted by hand.  Big-endian mode packs without alignment padding, so ">BBBHHH"
# is 9 bytes and not the 8 the eye expects; being one byte out makes the next
# length field read as garbage and the frame unpacker walk off the end of the
# buffer.
FRAME_FORMAT = ">BBBBHHH"             # n_planes, n_amps, pad, pad, mask_y, mask_c0, mask_c1
# masks are stored per plane because each plane has its own refresh schedule and
# band count (see chroma_tx_depth).  Hist: v1 stored ONE mask (luma's) for all.
FRAME_HEADER = struct.calcsize(FRAME_FORMAT)

def chroma_tx_depth(luma_chain, chroma_chain, tx_depth):
    """How many chroma bands to transmit, derived from the luma choice.

    One rule, applied to both planes: carry detail up to the same *absolute*
    spatial resolution.  Chroma's pyramid is anchored at half the frame (4:2:0 at
    the top) and has its own base size, so it generally has a different number of
    levels than luma, and a single shared `tx_depth` cannot mean the same thing
    for both.

    Sharing it was wrong at both ends.  At the quality presets -- where
    `chroma_min_side` is deliberately tiny so chroma resolves to plain 4:2:0 --
    luma's tx_depth pointed past the end of chroma's chain and chroma collapsed to
    an 8x8 base: a single block of colour for a 3840x2160 frame.  Matching on
    absolute resolution instead means chroma transmits *more* of its own bands
    than luma does (its bands are smaller), which is both correct and cheap.
    """
    if tx_depth <= 0 or len(chroma_chain) < 2:
        return 0
    cutoff = luma_chain[min(tx_depth, len(luma_chain) - 1)][0]
    n = 0
    for k in range(len(chroma_chain) - 1):
        if chroma_chain[k + 1][0] <= cutoff:
            n = k + 1
        else:
            break
    return n


MIN_BASE_SIDE = 96      # luma pyramid stops halving at this base size
MIN_LEVELS = 2          # always leave the synthesis engine something to do
REFRESH_SPREAD = 1      # bands brought up intra per frame during a refresh


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
# There is exactly ONE size chain per plane and everything derives from it: the
# padded frame size, the pyramid depth, the band shapes, and the geometry written
# into the stream header.  An earlier version had the chain implemented three
# times -- here to pick n_levels, in synthesis.build_pyramid to pick band sizes,
# and in chroma_shapes to pick chroma sizes -- and they rounded differently, so
# the frame layer asked for a 3-level pyramid and got a 4-level one back.  Every
# index downstream was then off by a band.  Duplicated geometry that can disagree
# is not a style problem, it is a bug that has not happened yet.
_CHAIN_CACHE: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}


def size_chain(h: int, w: int, min_side: int) -> List[Tuple[int, int]]:
    """[base, band0, ..., full resolution], coarsest first, for one plane."""
    key = (int(h), int(w), int(min_side))
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    ms = max(int(min_side), BLOCK)
    # Only the TOP level needs padding: halving a multiple of BLOCK and rounding
    # back up to BLOCK stays a multiple of BLOCK, so every level below is already
    # block-exact.  Padding by BLOCK * 2^(depth-1) instead -- which an earlier
    # version did to "guarantee" this -- inflated 1920x1080 to 3584x6144: 3.5x
    # the pixels, all border replicate, all of it coded.
    ph = ((h + BLOCK - 1) // BLOCK) * BLOCK
    pw = ((w + BLOCK - 1) // BLOCK) * BLOCK
    chain = [(ph, pw)]
    while min(ph, pw) > ms:
        nh = max(BLOCK, -(-(ph // 2) // BLOCK) * BLOCK)      # ceil to BLOCK
        nw = max(BLOCK, -(-(pw // 2) // BLOCK) * BLOCK)
        if (nh, nw) == chain[-1]:
            break
        chain.append((nh, nw))
        ph, pw = nh, nw
    chain.reverse()
    _CHAIN_CACHE[key] = chain
    return chain


def padded_size(h: int, w: int, min_side: int = MIN_BASE_SIDE) -> Tuple[int, int]:
    return size_chain(h, w, min_side)[-1]


def luma_shapes(h: int, w: int, base_min_side: int) -> List[Tuple[int, int]]:
    return size_chain(h, w, base_min_side)


def chroma_shapes(padded: Tuple[int, int], chroma_min_side: int
                  ) -> List[Tuple[int, int]]:
    """Chroma chain: starts at half the padded frame (4:2:0) and halves from there.

    `chroma_min_side` is in absolute pixels, so setting it equal to
    `base_min_side` gives chroma a base of roughly the same size as luma's -- the
    balanced choice, and the one the extreme presets need.  Leaving chroma at
    plain 4:2:0 (a one-entry chain) is what the high-quality presets want; it
    costs nothing extra there because those presets transmit the chroma bands
    anyway.
    """
    ph, pw = padded
    return size_chain(ph // 2, pw // 2, chroma_min_side)


def pyramid_depth(h: int, w: int, min_side: int = MIN_BASE_SIDE) -> int:
    return max(len(size_chain(h, w, min_side)), MIN_LEVELS)


# --------------------------------------------------------------------------- #
#  Stream / frame framing
# --------------------------------------------------------------------------- #
def write_stream_header(cfg: Config, n_levels: int, nframes: int) -> bytes:
    if not (0 <= int(cfg.chroma_min_side) <= 255):
        raise ValueError(
            f"chroma_min_side {cfg.chroma_min_side} does not fit the u8 header "
            f"slot (max 255) -- choose a smaller value")
    return struct.pack(STREAM_FORMAT, MAGIC, VERSION, cfg.width, cfg.height,
                       float(cfg.fps), nframes, n_levels, cfg.tx_depth,
                       int(cfg.base_min_side), int(cfg.chroma_min_side),
                       int(cfg.qp), int(cfg.gop))


def read_stream_header(buf: bytes) -> Tuple[dict, int]:
    from .guard import rd, StreamError, check_header_fields, MAGIC as GMAGIC, MAX_VERSION
    (magic, ver, w, h, fps, nframes, n_levels, tx_depth,
     bms, cms, qp, gop) = rd(buf, STREAM_FORMAT, 0, "stream header")
    if magic != GMAGIC:
        raise StreamError(f"not a PHOTON-C stream (magic {magic!r})")
    if not (1 <= ver <= MAX_VERSION):
        raise StreamError(f"unsupported PHOTON-C stream version {ver}")
    if not (1.0 <= float(fps) <= 240.0):
        raise StreamError(f"insane fps {fps}")
    # v1 is accepted for reading: its frame packets share the luma mask across
    # planes, which is exactly what a v1 decoder would have done, so decoding it
    # that way reproduces the writer's intent bit-exactly.
    return check_header_fields(
        dict(version=ver, width=w, height=h, fps=fps, nframes=nframes,
             n_levels=n_levels, tx_depth=tx_depth, base_min_side=bms,
             chroma_min_side=cms, qp=qp, gop=gop)), STREAM_HEADER


def _pack_frame(masks: List[int], plane_payloads: List[List[bytes]],
                amps: np.ndarray) -> bytes:
    """`masks[p]` bit k set => band k of plane p is present in this frame.

    Bit semantics rather than a count, because with progressive refresh the bands
    present are not contiguous.
    """
    out = bytearray()
    a = np.asarray(amps, dtype=np.float16)
    my = masks[0] if len(masks) > 0 else 0
    m0 = masks[1] if len(masks) > 1 else 0
    m1 = masks[2] if len(masks) > 2 else 0
    out += struct.pack(FRAME_FORMAT, len(plane_payloads), len(a), 0, 0,
                       my & 0xFFFF, m0 & 0xFFFF, m1 & 0xFFFF)
    for payloads in plane_payloads:
        out += struct.pack(">B", len(payloads))
        for p in payloads:
            out += struct.pack(">I", len(p)) + p
    # amplitudes as float16: they only scale texture, so 3 digits of relative
    # precision is far more than enough and it halves the cost
    out += a.tobytes()
    return bytes(out)


def _unpack_frame(buf: bytes, off: int, ver: int = 2) -> Tuple[dict, int]:
    from .guard import rd, need, StreamError, MAX_PLANES, MAX_AMPS, MAX_PACKET
    if ver >= 2:
        n_planes, n_amps, _p1, _p2, my, m0, m1 = rd(
            buf, FRAME_FORMAT, off, "frame header")
        off += FRAME_HEADER
        masks = [my, m0, m1]
    else:
        # v1 layout: n_planes, n_amps, mask_lo, mask_hi, single shared mask
        n_planes, n_amps, lo, hi = rd(buf, ">BBBB", off, "frame header v1")
        off += 4
        masks = [(lo | (hi << 8)) & 0xFFFF] * 3
    if not (1 <= n_planes <= MAX_PLANES):
        raise StreamError(f"insane plane count {n_planes} at {off}")
    if not (0 <= n_amps <= MAX_AMPS):
        raise StreamError(f"insane amp count {n_amps} at {off}")
    planes: List[List[bytes]] = []
    for _ in range(n_planes):
        npay = rd(buf, ">B", off, "payload count")[0]; off += 1
        if not (0 <= npay <= MAX_PLANES * 16):
            raise StreamError(f"insane payload count {npay} at {off}")
        pays = []
        for _ in range(npay):
            n = rd(buf, ">I", off, "payload length")[0]; off += 4
            if n > MAX_PACKET:
                raise StreamError(f"payload {n} B too large at {off}")
            need(buf, off, n, "payload bytes")
            pays.append(buf[off:off + n]); off += n
        planes.append(pays)
    need(buf, off, 2 * n_amps, "amps")
    amps = np.frombuffer(buf[off:off + 2 * n_amps], dtype=np.float16).astype(np.float64).copy()
    off += 2 * n_amps
    return dict(mask=masks[0], masks=masks, planes=planes, amps=amps), off


def bands_in_mask(mask: int, n_bands: int) -> List[int]:
    return [k for k in range(n_bands) if mask & (1 << k)]


# --------------------------------------------------------------------------- #
#  One plane
# --------------------------------------------------------------------------- #
class _Plane:
    """Pyramid state + code/decode/synthesise for a single plane.

    Both the encoder and the decoder drive this, which is the point: a band that
    is coded here is reconstructed here, so there is no second implementation to
    drift out of agreement with the first.
    """

    def __init__(self, shapes: List[Tuple[int, int]], tx_depth: int,
                 is_chroma: bool, tag: str, borrow: float, noise: float):
        self.shapes = shapes
        self.n_bands = len(shapes) - 1
        self.tx_depth = max(0, min(tx_depth, self.n_bands))
        self.is_chroma = is_chroma
        self.tag = tag
        self.borrow = borrow
        self.noise = noise
        self.offset = 128.0 if not is_chroma else 0.0
        # Encoder-only: which of THIS plane's bands still need an intra refresh.
        # Each plane keeps its own because each has a different band count; the
        # decoder never reads it, since intra-vs-inter is flagged in the payload.
        self.pending_refresh: List[int] = []
        # Detail bands are signed about zero.  The plane coder clips to [0,255],
        # so luma bands are offset by 128 on the way in and back on the way out.
        # Chroma bands need no offset because chroma is coded as a plane already
        # sitting in [0,255].
        self.reset()

    def reset(self) -> None:
        self.base: Optional[np.ndarray] = None
        self.bands: List[Optional[np.ndarray]] = [None] * self.n_bands
        self.pending_refresh = list(range(self.tx_depth))

    # ---- source decomposition ------------------------------------------- #
    def decompose(self, plane: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        return synth.build_pyramid(plane, self.shapes)

    # ---- encoding -------------------------------------------------------- #
    def encode(self, src: np.ndarray, keybase: bool,
               cfg: Config, qp: int) -> Tuple[List[bytes], int, List[float]]:
        base, details = self.decompose(src)
        payloads: List[bytes] = []
        mask = 0

        # Progressive refresh, on this plane's own band indices.  A GOP boundary
        # re-arms the list; each frame then brings up REFRESH_SPREAD bands intra
        # and codes the rest inter.  Spreading it is what keeps a keyframe from
        # being a multi-megabyte spike -- at 4K a conventional all-intra keyframe
        # would blow through any streaming buffer.
        if keybase:
            self.pending_refresh = list(range(self.tx_depth))
        take = self.pending_refresh[:REFRESH_SPREAD]
        self.pending_refresh = self.pending_refresh[REFRESH_SPREAD:]

        if keybase or self.base is None:
            pl, _, _ = _encode_intra_level(base, qp, 0, self.is_chroma,
                                           cfg.deblock_strength)
        else:
            pl, _, _, _ = _encode_inter_level(base, self.base, qp, 0, cfg.lam,
                                              cfg.search, cfg.deblock_strength,
                                              self.is_chroma, cfg.motion_max_side)
        payloads.append(pl)

        for k in range(self.tx_depth):
            shape = self.shapes[k + 1]
            band = details[k] if k < len(details) else np.zeros(shape)
            if band.shape != tuple(shape):
                band = cv2.resize(band, (shape[1], shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            band = band + self.offset
            ref = None if self.bands[k] is None else self.bands[k] + self.offset
            if k in take or ref is None:
                pl, _, _ = _encode_intra_level(band, qp, k + 1, self.is_chroma,
                                               cfg.deblock_strength)
            else:
                pl, _, _, _ = _encode_inter_level(band, ref, qp, k + 1, cfg.lam,
                                                  cfg.search, cfg.deblock_strength,
                                                  self.is_chroma, cfg.motion_max_side)
            payloads.append(pl)
            mask |= (1 << k)

        # amplitudes of the bands this frame will NOT have, so the decoder can
        # calibrate what it synthesises in their place
        amps = synth.band_rms(details) if details else np.zeros(0)
        missing = [k for k in range(self.n_bands) if k >= self.tx_depth]
        out_amps = [float(amps[k]) if k < len(amps) else 0.0 for k in missing]
        return payloads, mask, out_amps

    # ---- decoding -------------------------------------------------------- #
    def decode_base(self, payloads: List[bytes]) -> None:
        """Decode just the base layer.  Split out from decode_bands because the
        motion field is derived from the base, and the bands need that field to
        warp borrowed detail -- so the base has to land first."""
        if payloads:
            pl = payloads[0]
            if self.base is None or _peek_intra(pl, 0):
                self.base, _, _ = _decode_intra_level(pl, 0, self.shapes[0])
            else:
                self.base, _, _ = _decode_inter_level(pl, 0, self.base, self.shapes[0])
        if self.base is None:
            self.base = np.full(self.shapes[0], 128.0)

    def decode_bands(self, payloads: List[bytes], mask: int, amps: np.ndarray,
                     motion: Optional[np.ndarray], frame_idx: int) -> np.ndarray:
        """Decode this frame's detail payloads, synthesise what is missing, and
        return the full-resolution plane."""
        prev = list(self.bands)          # last frame's bands: the borrow source
        present = bands_in_mask(mask, self.n_bands)
        i = 1                            # payloads[0] is the base, already done
        for k in present:
            if i >= len(payloads):
                break
            pl = payloads[i]; i += 1
            shape = self.shapes[k + 1]
            ref = None if self.bands[k] is None else self.bands[k] + self.offset
            if ref is None or _peek_intra(pl, 0):
                rec, _, _ = _decode_intra_level(pl, 0, shape)
            else:
                rec, _, _ = _decode_inter_level(pl, 0, ref, shape)
            self.bands[k] = rec - self.offset

        cur = self.base.astype(RDT)
        amp_i = 0
        for k in range(self.n_bands):
            shape = self.shapes[k + 1]
            b = self.bands[k]
            # exact-double levels: fused ring kernel (f32), one pass, no
            # separately-materialised coarse_up.  Anything else -> cv2 chain.
            fused = (RDT is np.float32 and b is not None
                     and tuple(b.shape) == tuple(shape)
                     and tuple(cur.shape) == (shape[0] // 2, shape[1] // 2)
                     and shape[0] == 2 * cur.shape[0]
                     and shape[1] == 2 * cur.shape[1])
            if fused:
                cur = l4_up2_add(cur.astype(np.float32, copy=False),
                                 b.astype(np.float32, copy=False))
                continue
            coarse_up = cv2.resize(cur, (shape[1], shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
            if b is None:
                a = float(amps[amp_i]) if amp_i < len(amps) else 0.0
                amp_i += 1
                self.bands[k] = synth.synthesise_band(
                    prev[k], motion, coarse_up, shape, a,
                    seed=(frame_idx * 977 + (k + 1) * 31 + 7),
                    guide=None, borrow_gain=self.borrow, noise_gain=self.noise
                ).astype(RDT, copy=False)   # keep f32 planes clean of f64
            elif b.shape != tuple(shape):
                self.bands[k] = cv2.resize(b, (shape[1], shape[0]),
                                           interpolation=cv2.INTER_LANCZOS4)
            cur = coarse_up + self.bands[k]
        return cur

    @property
    def n_amps(self) -> int:
        """How many amplitude values this plane writes into a frame packet."""
        return max(0, self.n_bands - self.tx_depth)


# --------------------------------------------------------------------------- #
#  Shared reconstruction path  (BOTH encoder and decoder run this)
# --------------------------------------------------------------------------- #
class _Reconstructor:
    def __init__(self, cfg_shapes: List[Tuple[int, int]],
                 c_shapes: List[Tuple[int, int]], tx_depth: int,
                 use_chroma: bool, borrow: float, noise: float):
        self.shapes = cfg_shapes
        self.padded = cfg_shapes[-1]
        self.cshapes = c_shapes
        self.use_chroma = use_chroma
        self.y_tx = max(0, min(tx_depth, len(cfg_shapes) - 2))
        self.c_tx = chroma_tx_depth(cfg_shapes, c_shapes, self.y_tx)
        self.y = _Plane(cfg_shapes, self.y_tx, False, "y", borrow, noise)
        self.c = [_Plane(c_shapes, self.c_tx, True, f"c{i}", borrow, noise)
                  for i in range(2)]
        self.flow_shape = (max(16, self.padded[0] // 4), max(16, self.padded[1] // 4))
        self.frame_idx = 0
        self.prev_small: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.y.reset()
        for p in self.c:
            p.reset()
        self.prev_small = None
        self.frame_idx = 0

    def motion_field(self, small: np.ndarray) -> Optional[np.ndarray]:
        """Dense motion at reduced resolution, derived -- never transmitted.

        The single biggest bit saving in the frame layer.  A raw float16 flow
        field at 1/4 resolution costs 8 bytes per pixel there, which on 4K is
        ~184 kB per frame: more than every coefficient in the frame put together,
        at a bitrate where the whole frame has to fit in a few hundred bytes.  It
        would have dwarfed the codec.

        It is also unnecessary.  Both sides hold the previous reconstruction and
        the current decoded base, so both run the same estimate and get
        bit-identical fields -- provided the inputs are reconstructions on both
        sides, never the source on the encoder's side.
        """
        if self.prev_small is None:
            self.prev_small = small
            return None
        mv = dense_flow(self.prev_small, small, max_side=max(small.shape))
        self.prev_small = small
        return mv

    def rebuild(self, f: dict) -> np.ndarray:
        planes = f["planes"]
        amps = f["amps"]
        masks = f.get("masks") or [f["mask"]] * 3
        mask = masks[0]

        # 1. luma base, so the motion field can be derived from something both
        #    sides hold
        self.y.decode_base(planes[0] if planes else [])
        base_small = cv2.resize(self.y.base, (self.flow_shape[1], self.flow_shape[0]),
                                interpolation=cv2.INTER_AREA)
        mv = self.motion_field(base_small)

        # 2.+3. plane-parallel band decode.  The three pyramid reconstructions are
        # independent until yuv_to_rgb; each _Plane holds its own base/bands, and
        # cv2/ctypes calls release the GIL, so this is true parallelism on
        # multicore.  Chroma reuses the luma motion field (warp_band rescales it
        # per level, magnitudes included — right for 2x-larger chroma pixels).
        # base decode is cheap (≈1 ms/plane) and must precede the band judge:
        # do chroma bases up front, then farm the three band sweeps across the
        # pool on top of each plane's freshly decoded base.
        luma_pls = planes[0] if planes else []
        off = self.y.n_amps
        chroma_jobs = []
        for pi in range(2):
            cp = self.c[pi]
            have = (self.use_chroma and len(planes) > pi + 1 and planes[pi + 1])
            cp.decode_base(planes[pi + 1] if have else [])
            job_amps = amps[off:off + cp.n_amps] if have else None
            off += cp.n_amps if have else 0
            chroma_jobs.append((cp, planes[pi + 1] if have else [],
                                masks[pi + 1], job_amps))

        def _run(cp, pls, msk, am):
            if pls:
                return cp.decode_bands(pls, msk, am if am is not None else am,
                                       mv, self.frame_idx)
            return self._recompose(cp, None)

        if _POOL is not None:
            fut_y = _POOL.submit(self.y.decode_bands, luma_pls, mask, amps, mv,
                                 self.frame_idx)
            fut_c = [_POOL.submit(_run, *chroma_jobs[pi]) for pi in range(2)]
            y_full = fut_y.result()
            cfull = [fut_c[0].result(), fut_c[1].result()]
        else:
            y_full = self.y.decode_bands(luma_pls, mask, amps, mv, self.frame_idx)
            cfull = [_run(*chroma_jobs[pi]) for pi in range(2)]

        # Chroma stays at half res: yuv_to_rgb's fused native kernel (or its
        # dyadic-tap numpy fallback) upsamples inline.  Kills two full-res
        # f64 resizes (~23 ms/frame at 4K) plus 132 MB of intermediate traffic.
        rgb = yuv_to_rgb(y_full, cfull[0], cfull[1])
        self.frame_idx += 1
        return rgb

    @staticmethod
    def _recompose(plane: _Plane, motion: Optional[np.ndarray]) -> np.ndarray:
        cur = plane.base.astype(RDT)
        for k in range(plane.n_bands):
            shape = plane.shapes[k + 1]
            cur = cv2.resize(cur, (shape[1], shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)
            b = plane.bands[k]
            if b is not None:
                if b.shape != tuple(shape):
                    b = cv2.resize(b, (shape[1], shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
                    plane.bands[k] = b
                cur = cur + b
        return cur


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #
class Encoder:
    """Stateful encoder.  push_frame() per frame, then finish()."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shapes = luma_shapes(cfg.height, cfg.width, cfg.base_min_side)
        self.n_levels = len(self.shapes)
        self.padded = self.shapes[-1]
        self.cshapes = chroma_shapes(self.padded, cfg.chroma_min_side)
        self.n_bands = self.n_levels - 1
        self.rec = _Reconstructor(self.shapes, self.cshapes, cfg.tx_depth,
                                  cfg.chroma, cfg.synth_borrow, cfg.synth_noise)
        self.enc_planes = [self.rec.y] + (self.rec.c if cfg.chroma else [])
        self.frames: List[bytes] = []
        self.stats: List[dict] = []
        self.last_rgb: Optional[np.ndarray] = None   # closed-loop recon of last push
        self.frame_idx = 0

    # ---- helpers --------------------------------------------------------- #
    def _to_planes(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ph, pw = self.padded
        if rgb.shape[0] != ph or rgb.shape[1] != pw:
            rgb = cv2.copyMakeBorder(rgb, 0, ph - rgb.shape[0], 0, pw - rgb.shape[1],
                                     cv2.BORDER_REPLICATE)
        y, u, v = rgb_to_yuv(rgb)
        ch, cw = self.cshapes[-1]
        u = cv2.resize(u, (cw, ch), interpolation=cv2.INTER_AREA)
        v = cv2.resize(v, (cw, ch), interpolation=cv2.INTER_AREA)
        return y, u, v

    def _schedule(self) -> bool:
        """Whether this frame starts a GOP, i.e. re-codes the base intra.

        Which *bands* get refreshed is no longer decided here: each plane runs its
        own progressive refresh over its own band indices, because luma and chroma
        have different band counts.  See chroma_tx_depth.
        """
        return (self.frame_idx % self.cfg.gop) == 0

    # ---- public API ------------------------------------------------------ #
    def push_frame(self, rgb: np.ndarray, qp: Optional[int] = None) -> dict:
        t0 = time.time()
        cfg = self.cfg
        qp = cfg.qp if qp is None else int(qp)
        keybase = self._schedule()

        rgb = np.ascontiguousarray(rgb[:, :, :3])
        srcs = self._to_planes(rgb)

        plane_payloads: List[List[bytes]] = []
        masks: List[int] = []
        amps: List[float] = []
        for i, plane in enumerate(self.enc_planes):
            pays, mask, pa = plane.encode(srcs[i], keybase, cfg, qp)
            plane_payloads.append(pays)
            masks.append(mask)
            amps.extend(pa)

        packet = _pack_frame(masks, plane_payloads, np.asarray(amps, np.float64))

        # ---- closed loop: reconstruct exactly as the decoder will ---- #
        f, _ = _unpack_frame(packet, 0)   # encoder always writes v2
        self.last_rgb = self.rec.rebuild(f)

        self.frames.append(packet)
        st = {"keybase": keybase, "bytes": len(packet), "qp": qp,
              "refresh": [list(p.pending_refresh) for p in self.enc_planes],
              "plane_bytes": [[len(p) for p in pp] for pp in plane_payloads],
              "enc_time": time.time() - t0}
        self.stats.append(st)
        self.frame_idx += 1
        return st

    def finish(self) -> bytes:
        hdr = write_stream_header(self.cfg, self.n_levels, len(self.frames))
        body = bytearray()
        for f in self.frames:
            body += struct.pack(">I", len(f)) + f
        return hdr + bytes(body)

    @property
    def total_bytes(self) -> int:
        return STREAM_HEADER + sum(len(f) + 4 for f in self.frames)

    @property
    def bitrate_kbps(self) -> float:
        dur = max(len(self.frames), 1) / max(self.cfg.fps, 1e-6)
        return 8.0 * self.total_bytes / dur / 1000.0


# --------------------------------------------------------------------------- #
#  Decoder
# --------------------------------------------------------------------------- #
class Decoder:
    def __init__(self, stream: bytes):
        self.buf = stream
        self.meta, off = read_stream_header(stream)
        self.off = off
        self.shapes = luma_shapes(self.meta["height"], self.meta["width"],
                                  self.meta["base_min_side"])
        self.n_levels = len(self.shapes)
        if self.n_levels != self.meta["n_levels"]:
            from .guard import StreamError
            raise StreamError(
                f"stream says {self.meta['n_levels']} pyramid levels but the "
                f"geometry gives {self.n_levels} -- header and size chain disagree")
        self.padded = self.shapes[-1]
        self.cshapes = chroma_shapes(self.padded, self.meta["chroma_min_side"])
        self.rec = self._fresh()
        self.frame_offsets: List[int] = []
        self._index_frames()

    def _fresh(self) -> _Reconstructor:
        return _Reconstructor(self.shapes, self.cshapes, self.meta["tx_depth"],
                              True, synth.BORROW_GAIN, synth.NOISE_GAIN)

    def _index_frames(self) -> None:
        from .guard import StreamError, MAX_PACKET
        off, buf = self.off, self.buf
        declared = self.meta["nframes"]
        for _ in range(declared):
            if off + 4 > len(buf):
                break
            (n,) = struct.unpack_from(">I", buf, off)
            if n > MAX_PACKET:
                raise StreamError(f"frame packet {n} B too large at {off}")
            if off + 4 + n > len(buf):
                break
            self.frame_offsets.append(off + 4)
            off += 4 + n
        if len(self.frame_offsets) == 0 and declared > 0:
            raise StreamError(
                f"header declares {declared} frames but the stream body "
                f"is empty/truncated")

    @property
    def nframes(self) -> int:
        return len(self.frame_offsets)

    @property
    def fps(self) -> float:
        return self.meta["fps"]

    @property
    def size(self) -> Tuple[int, int]:
        return self.meta["width"], self.meta["height"]

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])

    def _is_keybase(self, i: int) -> bool:
        """A frame is a seek point iff its base layer was coded intra, which the
        peek function reads from the payload's first byte without decoding."""
        f, _ = _unpack_frame(self.buf, self.frame_offsets[i], self.meta['version'])
        return bool(f["planes"] and f["planes"][0]) and _peek_intra(f["planes"][0][0], 0)

    def _is_full_refresh(self, i: int) -> bool:
        """Frame where EVERY plane's base AND every present band payload is
        intra -- the only states that a fresh reconstructor can decode
        bit-identically, because bands refreshed progressively across GOPs
        (REFRESH_SPREAD) have inter payloads whose references simply do not
        exist in a mid-GOP fresh state.  Keybase-only seeking landed on
        exactly those payloads and failed mid-decode with an unrelated-
        looking dc-count StreamError; fixed by rounding seeks back to the
        last genuinely decodeable anchor."""
        f, _ = _unpack_frame(self.buf, self.frame_offsets[i], self.meta['version'])
        planes = f["planes"]
        if not (planes and planes[0]) or not _peek_intra(planes[0][0], 0):
            return False
        masks = f.get("masks") or [f.get("mask")] * 3
        for pi, pls in enumerate(planes):
            if not pls:
                continue
            # present bands on this plane
            n_bands = (len(self.shapes) - 1) if pi == 0 else (len(self.cshapes) - 1)
            if pi > 0 and not self.meta.get("chroma", True):
                continue
            present = bands_in_mask(masks[min(pi, len(masks) - 1)], n_bands)
            for k in present:
                if k + 1 >= len(pls) or not _peek_intra(pls[k + 1], 0):
                    return False
        return True

    def seek(self, i: int) -> None:
        j = i
        while j > 0 and not self._is_full_refresh(j):
            j -= 1
        self.rec = self._fresh()
        self.rec.frame_idx = j
        while self.rec.frame_idx < i:
            f, _ = _unpack_frame(self.buf, self.frame_offsets[self.rec.frame_idx], self.meta['version'])
            self.rec.rebuild(f)

    def _crop(self, rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(rgb[:self.meta["height"], :self.meta["width"]])

    def decode_frame(self, i: int) -> np.ndarray:
        from .guard import StreamError
        if not (0 <= i < len(self.frame_offsets)):
            raise StreamError(f"frame index {i} out of range "
                              f"(0..{len(self.frame_offsets) - 1})")
        if i != self.rec.frame_idx:
            self.seek(i)
        f, _ = _unpack_frame(self.buf, self.frame_offsets[i], self.meta['version'])
        return self._crop(self.rec.rebuild(f))

    def decode_all(self) -> List[np.ndarray]:
        # np.asarray_copy: the native colour kernel serves frames from a
        # 4-buffer rotation, so a frame collected without a copy is silently
        # overwritten three conversions later.  Streaming consumers that
        # use each frame before decode_frame(i+3) can hold decode_frame()'s
        # return directly.
        out = []
        for i in range(self.nframes):
            f, _ = _unpack_frame(self.buf, self.frame_offsets[i], self.meta['version'])
            out.append(self._crop(self.rec.rebuild(f)).copy())
        return out

    def frame_bytes(self, i: int) -> int:
        (n,) = struct.unpack_from(">I", self.buf, self.frame_offsets[i] - 4)
        return n


# --------------------------------------------------------------------------- #
#  Rate control
# --------------------------------------------------------------------------- #
def encode_clip(frames: List[np.ndarray], cfg: Config,
                target_kbps: Optional[float] = None,
                probe: int = 8, verbose: bool = False) -> Tuple[bytes, Encoder]:
    """Encode a clip, optionally hunting the qp that hits `target_kbps`.

    Two-pass in the simplest form that works: probe a few frames at two qp values,
    interpolate on log-rate (close to linear in qp), refine by bisection, then
    encode the whole clip.  A production codec would do buffer-constrained
    per-frame QP modulation; for a fixed-bandwidth streaming profile the
    clip-level solution is simpler and more stable, and it is what the presets are
    calibrated against.
    """
    if target_kbps is None:
        enc = Encoder(cfg)
        for f in frames:
            enc.push_frame(f)
        return enc.finish(), enc

    n = min(probe, len(frames))
    idx = [int(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
    probe_frames = [frames[i] for i in idx]

    def rate_at(qp: int) -> float:
        c = Config(**{**cfg.__dict__, "qp": int(qp)})
        e = Encoder(c)
        for f in probe_frames:
            e.push_frame(f)
        return e.bitrate_kbps

    lo, hi = 25, 95
    r_lo, r_hi = rate_at(lo), rate_at(hi)
    if verbose:
        print(f"    probe qp={lo} -> {r_lo:.1f} kbps ; qp={hi} -> {r_hi:.1f} kbps")
    if target_kbps <= r_lo:
        best = lo
    elif target_kbps >= r_hi:
        best = hi
    else:
        best = lo + (hi - lo) * (math.log(target_kbps / r_lo) /
                                 max(math.log(r_hi / r_lo), 1e-6))
        best = int(round(min(max(best, lo), hi)))
        for _ in range(4):
            r = rate_at(best)
            if verbose:
                print(f"    qp={best} -> {r:.1f} kbps (target {target_kbps:.1f})")
            if abs(r - target_kbps) / target_kbps < 0.05:
                break
            if r > target_kbps:
                hi = best
            else:
                lo = best
            best = int(round((lo + hi) / 2))

    cfg2 = Config(**{**cfg.__dict__, "qp": int(best)})
    enc = Encoder(cfg2)
    for f in frames:
        enc.push_frame(f)
    return enc.finish(), enc
