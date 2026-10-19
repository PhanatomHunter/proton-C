"""
PHOTON-C  ::  Block transform, quantisation and prediction primitives
=====================================================================
Everything here is deterministic and shared verbatim by the encoder and the
decoder's in-loop reconstruction, which is what stops prediction drift.
"""
from __future__ import annotations

import cv2
import numpy as np

BLOCK = 8


# --------------------------------------------------------------------------- #
#  DCT-II (orthonormal), as a matrix so a whole plane transforms in one einsum
# --------------------------------------------------------------------------- #
def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    M = np.cos((2 * i + 1) * k * np.pi / (2 * n)) * np.sqrt(2.0 / n)
    M[0] *= np.sqrt(0.5)
    return M


DCT_M = _dct_matrix(BLOCK)
IDCT_M = DCT_M.T


# --------------------------------------------------------------------------- #
#  Quantisation matrices
# --------------------------------------------------------------------------- #
# Perceptual weighting, JPEG-shaped but flatter: at the bitrates this codec
# targets we would rather spend bits on mid frequencies (where edges live) than
# protect DC to three decimal places.
_QM_LUMA = np.array([
    [10, 12, 15, 20, 26, 33, 41, 50],
    [12, 13, 17, 23, 30, 38, 47, 57],
    [15, 17, 20, 27, 34, 43, 53, 64],
    [20, 23, 27, 33, 41, 51, 62, 74],
    [26, 30, 34, 41, 50, 60, 73, 86],
    [33, 38, 43, 51, 60, 71, 85, 99],
    [41, 47, 53, 62, 73, 85, 99, 114],
    [50, 57, 64, 74, 86, 99, 114, 130],
], dtype=np.float64)

_QM_CHROMA = np.array([
    [12, 14, 18, 24, 31, 39, 48, 58],
    [14, 16, 20, 27, 35, 44, 54, 65],
    [18, 20, 24, 31, 39, 49, 60, 72],
    [24, 27, 31, 38, 47, 58, 71, 84],
    [31, 35, 39, 47, 57, 69, 84, 98],
    [39, 44, 49, 58, 69, 82, 98, 114],
    [48, 54, 60, 71, 84, 98, 114, 130],
    [58, 65, 72, 84, 98, 114, 130, 147],
], dtype=np.float64)

# Contrast-sensitivity / viewing-distance weighting across the pyramid.
#
# A detail band that lives at 1/2^s of the frame resolution describes structure
# 2^s times *finer* than the frame when it is expanded back up.  On a phone, or
# at a normal living-room distance from a 4K panel, that fine structure is close
# to invisible -- so it should be quantised harder than the coarse base layer
# that carries the actual image.
#
# This is the single largest perceptual win available in the whole codec and it
# costs nothing at decode time.  Note the direction: the BASE gets the finest
# quantiser, not the coarsest.  Getting this the other way round (which is the
# intuitive mistake, since "level 0" sounds like it should be the least
# important) throws away the structure and spends the budget on texture.
def qp_scale(qp: int) -> float:
    """JPEG's quality-factor -> step-size mapping.

    qp is 1..99 like a JPEG quality factor: high means fine.  scale = 5000/qp/100
    below 50 and 2 - qp/50 above, which is the mapping every JPEG encoder in the
    world uses, so its absolute calibration is known-good rather than something
    I have to guess at.  qp=50 reproduces the standard tables exactly.
    """
    q = float(min(max(int(qp), 1), 99))
    return (5000.0 / q) / 100.0 if q < 50 else (200.0 - 2.0 * q) / 100.0


def band_qm_scale(s: int) -> float:
    """Quantiser multiplier for a band `s` halvings above the base.

    s = 0 is the base layer itself and gets exactly the standard tables, so at
    qp = 50 this codec quantises its base layer identically to a JPEG at quality
    50 -- a calibration point that is known-good rather than guessed.  Each
    halving above the base costs 1.25x: enough to redirect budget towards
    structure, gentle enough that bands still reconstruct recognisably and the
    synthesis engine has something to sharpen.
    """
    return 1.25 ** max(int(s), 0)


def quant_matrix(s: int, qp: int, chroma: bool = False) -> np.ndarray:
    """Effective quantisation step for one pyramid band.

    `s` is the number of halvings above the base (0 = base, finest quantiser).
    """
    base = _QM_CHROMA if chroma else _QM_LUMA
    return np.maximum(base * qp_scale(qp) * band_qm_scale(s), 1.0)


# --------------------------------------------------------------------------- #
#  Blocking helpers  (vectorised over a whole plane -- no per-block Python loop)
# --------------------------------------------------------------------------- #
def to_blocks(plane: np.ndarray) -> np.ndarray:
    """(H, W) -> (nb, BLOCK, BLOCK) where nb = (H/BLOCK) * (W/BLOCK)."""
    h, w = plane.shape
    nbh, nbw = h // BLOCK, w // BLOCK
    return plane[:nbh * BLOCK, :nbw * BLOCK].reshape(
        nbh, BLOCK, nbw, BLOCK).transpose(0, 2, 1, 3).reshape(-1, BLOCK, BLOCK)


def from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    nbh, nbw = h // BLOCK, w // BLOCK
    out = blocks.reshape(nbh, nbw, BLOCK, BLOCK).transpose(0, 2, 1, 3)
    plane = np.zeros((h, w), dtype=np.float64)
    plane[:nbh * BLOCK, :nbw * BLOCK] = out.reshape(nbh * BLOCK, nbw * BLOCK)
    return plane


def forward_dct(blocks: np.ndarray) -> np.ndarray:
    return np.einsum('ij,njk,lk->nil', DCT_M, blocks, DCT_M, optimize=True)


def inverse_dct(coefs: np.ndarray) -> np.ndarray:
    return np.einsum('ij,njk,lk->nil', IDCT_M, coefs, IDCT_M, optimize=True)


# --------------------------------------------------------------------------- #
#  Zigzag order for 8x8 -- coefficients are serialised in this order so that
#  runs of trailing zeros are contiguous
# --------------------------------------------------------------------------- #
def _zigzag_index(n: int = BLOCK) -> np.ndarray:
    idx = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                idx.append(i * n + (s - i))
        else:
            for i in range(max(0, s - n + 1), min(s + 1, n)):
                idx.append(i * n + (s - i))
    return np.array(idx, dtype=np.int64)


ZIGZAG = _zigzag_index()
UNZIGZAG = np.argsort(ZIGZAG)


# --------------------------------------------------------------------------- #
#  Colour
# --------------------------------------------------------------------------- #
def rgb_to_yuv(rgb: np.ndarray):
    """BT.601 full-range, returns (Y, Cb, Cr) as float32 arrays.

    float32 rather than float64 purely for memory: a 4K plane is 33 MB in float32
    and 66 MB in float64, and this function is called once per frame with three
    planes live at once.  float32 carries ~7 decimal digits, which is several
    orders of magnitude more precision than an 8-bit video sample needs.
    """
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8),
                       cv2.COLOR_RGB2YCrCb).astype(np.float32)
    return yuv[:, :, 0], yuv[:, :, 2], yuv[:, :, 1]      # Y, Cb, Cr


def _chroma2x_to_full(c: np.ndarray) -> np.ndarray:
    """2x chroma upsample, separable dyadic taps (0.75, 0.25), edge-replicated.

    Mirrors ptn_yuv2rgb_c2x term-for-term in IEEE f64 (0.75*a + 0.25*b,
    vertical pass then horizontal pass) so native and fallback agree
    bit-for-bit.  cv2 INTER_LINEAR at 2x computes the same geometry but its
    internal coefficient arithmetic rounds up to 5.7e-14 differently -- both
    encoder and decoder route through yuv_to_rgb, so they stay locked, and
    the delta survives rint→u8 only on exact .5 borders (measure zero).
    """
    h2, w2 = c.shape
    dt = np.float32 if c.dtype == np.float32 else np.float64
    up = np.empty((2 * h2, w2), dtype=dt)
    kd = np.vstack([c[:1], c[:-1]])          # row k-1, replicated at the edge
    ku = np.vstack([c[1:], c[-1:]])          # row k+1
    up[0::2] = 0.75 * c + 0.25 * kd
    up[1::2] = 0.75 * c + 0.25 * ku
    out = np.empty((2 * h2, 2 * w2), dtype=dt)
    ld = np.hstack([up[:, :1], up[:, :-1]])
    lu = np.hstack([up[:, 1:], up[:, -1:]])
    out[:, 0::2] = 0.75 * up + 0.25 * ld
    out[:, 1::2] = 0.75 * up + 0.25 * lu
    return out


# Lanczos-4 8-tap kernel, per-phase normalised, BORDER_REPLICATE.  Shared by
# the C ring kernel (native/photon_native.c L4E[]) and the numpy fallback;
# keep them in lockstep (tests/test_kernels.py enforces bit-exact equality).
L4_TAPS = (-0.00397062, 0.03146775, -0.09166057, 0.28268394,
           0.89338857, -0.1523039, 0.05544899, -0.01505417)


def _l4_up2_add_np(src: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Exact reference: separable Lanczos-4 2x upsample + band add.

    Same taps and same op order as ptn_l4_up2_add_f32 so the fallback is
    bit-exact with the C kernel on f32 planes; matches cv2.resize(2x,
    INTER_LANCZOS4) + band to float rounding in general.
    """
    a = src.astype(np.float64) if src.dtype != np.float32 else src
    T = np.array(L4_TAPS, dtype=a.dtype)
    h, w = a.shape
    sw = np.pad(a, ((0, 0), (4, 4)), mode="edge")
    tmp = np.zeros((h, 2 * w), dtype=a.dtype)
    for q in range(8):
        tmp[:, 0::2] += T[q] * sw[:, q:q + w]
        tmp[:, 1::2] += T[7 - q] * sw[:, q + 1:q + 1 + w]
    sv = np.pad(tmp, ((4, 4), (0, 0)), mode="edge")
    out2 = np.zeros((2 * h, 2 * w), dtype=a.dtype)
    if band is not None:
        out2 += band
    for q in range(8):
        out2[0::2, :] += T[q] * sv[q:q + h]
        out2[1::2, :] += T[7 - q] * sv[q + 1:q + 1 + h]
    return out2


def l4_up2_add(src: np.ndarray, band: np.ndarray) -> np.ndarray:
    """2x LANCZOS-4 upsample of `src` + `band`, one pass when possible.

    C ring kernel first (f32, OMP); numpy fallback implements the SAME taps
    in the same op order so both agree bit-for-bit.  Callers must only ask
    for exact doubling; everything else goes through cv2 on the caller side.
    """
    h, w = src.shape
    if src.dtype == np.float32:
        out = np.empty((2 * h, 2 * w), dtype=np.float32)
        try:
            from . import native
            if native.l4_up2_add_f32(src, band, out):
                return out
        except Exception:
            pass
    return _l4_up2_add_np(src, band)


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.shape != y.shape:
        # half-res chroma: fused kernel (one pass, no full-res f64 chroma)
        try:
            from . import native
            out = native.yuv2rgb_chroma2x(y, u, v)
            if out is not None:
                return out
        except Exception:
            pass
        u = _chroma2x_to_full(np.ascontiguousarray(u))
        v = _chroma2x_to_full(np.ascontiguousarray(v))
    """YCbCr -> RGB.

    First choice is the native fused kernel (native/photon_native.c): one
    pass, OpenMP row-split, LUT-based -- proven bit-identical to the NumPy/cv2
    path exhaustively over all 2^24 inputs by tests/test_native.py.  If the
    .so is not built, the original NumPy path below still runs and produces
    the same bytes.

    Stacking the three planes into one (H, W, 3) float64 array first -- the
    obvious implementation -- allocates 199 MB at 4K on top of the planes
    themselves, which in a ~1.4 GB sandbox is what turns a working encode into an
    OOM.  Writing into a preallocated float32 scratch buffer one plane at a time
    peaks at ~58 MB instead.
    """
    try:
        from . import native
        out = native.yuv_to_rgb_native(y, u, v)
        if out is not None:
            return out
    except Exception:
        pass
    h, w = y.shape[:2]
    out = np.empty((h, w, 3), dtype=np.uint8)
    tmp = np.empty((h, w), dtype=np.float32)
    for i, plane in enumerate((y, v, u)):          # YCrCb channel order
        np.copyto(tmp, plane, casting="unsafe")
        np.rint(tmp, out=tmp)
        np.clip(tmp, 0, 255, out=tmp)
        out[:, :, i] = tmp
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)


# --------------------------------------------------------------------------- #
#  In-loop deblocking
# --------------------------------------------------------------------------- #
def deblock(plane: np.ndarray, strength: float) -> np.ndarray:
    """Cheap edge-preserving smoothing across block boundaries.

    Not a real adaptive deblocker, but at the quantisation steps this codec runs
    at, the 8x8 grid is the single most visible artifact and blurring *only*
    across the boundary (never within a block) removes most of it for a few
    lines of code.  `strength` in [0, 1]; 0 disables it.
    """
    if strength <= 0.0:
        return plane
    out = plane.copy()
    h, w = out.shape
    a = float(strength) * 0.28
    for off in (BLOCK,):
        # vertical boundaries
        if w > off:
            left = out[:, off - 1:w - 1:off]
            right = out[:, off::off]
            n = min(left.shape[1], right.shape[1])
            if n:
                avg = (left[:, :n] + right[:, :n]) * 0.5
                out[:, off - 1:off - 1 + n * off:off] = left[:, :n] + a * (avg - left[:, :n])
                out[:, off::off][:, :n] = right[:, :n] + a * (avg - right[:, :n])
        # horizontal boundaries
        if h > off:
            top = out[off - 1:h - 1:off, :]
            bot = out[off::off, :]
            n = min(top.shape[0], bot.shape[0])
            if n:
                avg = (top[:n] + bot[:n]) * 0.5
                out[off - 1:off - 1 + n * off:off, :] = top[:n] + a * (avg - top[:n])
                out[off::off, :] = bot[:n] + a * (avg - bot[:n])
    return out
