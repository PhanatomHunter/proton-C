"""
PHOTON-C  ::  Motion field
==========================
Dense-flow seeded, block-refined motion estimation with a rate-distortion skip
decision.  Fully vectorised: no per-block Python loop anywhere.

Why not plain hierarchical block matching
-----------------------------------------
Block matching is the most expensive thing an encoder does, and at ultra-low
bitrates it optimises the wrong quantity: it minimises prediction error with no
regard for how many bits the resulting *vectors* cost.  Two vectors one pixel
apart in a smooth pan are nearly free to predict from each other, and block
matching has no idea.

So the field is seeded with dense optical flow (OpenCV DIS: subpixel, smooth by
construction, and orders of magnitude cheaper than full search at this
resolution), then every block chooses between a small candidate set by
**rate-distortion** cost  J = SAD + lambda * bits(vector | spatial predictor).
Signalling the difference from the neighbour rather than the raw vector is the
important part -- it is why a global pan collapses to a stream of zeros.

Candidates, all evaluated for every block simultaneously:
    dense-flow seed, zero motion, 8 integer neighbours of the seed,
    left neighbour's winner, top neighbour's winner, top-left's winner.
The neighbour candidates need the previous pass's result, so this is two
vectorised passes rather than a raster scan.  Each pass is a handful of
`warp_plane` calls, each of which is one numpy gather over the whole plane.

Skip decision
-------------
A block is skipped when its best prediction is already better than what the
quantiser could represent: if the residual is smaller than the rounding error of
the first nonzero coefficient, coding it buys literally nothing and costs bits.
Skips are the dominant symbol in a static scene and are coded as runs, so at 99%
density a 4K skip map costs 83 bytes.  That is the mechanism which makes
"literally extreme low bandwidth" arithmetically possible rather than
aspirational.
"""

from __future__ import annotations

import cv2
import numpy as np

DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
BORDER = 16          # edge-replication margin so pan vectors stay legal


# --------------------------------------------------------------------------- #
#  Dense flow seed
# --------------------------------------------------------------------------- #
_dis = None


def dense_flow(ref_gray: np.ndarray, cur_gray: np.ndarray,
               max_side: int = 512) -> np.ndarray:
    """Dense (dx, dy) field, float32, at the resolution of `cur_gray`.

    Computed on a downscaled copy and upsampled: motion fields are smooth, so
    downscaling costs almost no accuracy and most of the runtime.
    """
    global _dis
    h, w = cur_gray.shape
    if max(h, w) < 12 or h < 12 or w < 12:
        return np.zeros((h, w, 2), dtype=np.float32)
    scale = min(1.0, max_side / float(max(h, w)))
    # DIS requires single-channel CV_8U.  Converting here (rather than asking
    # every caller to pre-convert) keeps the interface in the same float domain
    # as the rest of the codec.
    a = np.clip(ref_gray, 0, 255).astype(np.uint8)
    b = np.clip(cur_gray, 0, 255).astype(np.uint8)
    if scale < 1.0:
        # DIS needs at least 12x12 input; below that the field is meaningless
        # anyway, so cheapest-correct answer is "zero motion".
        sw, sh = max(12, int(round(w * scale))), max(12, int(round(h * scale)))
        a = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (sw, sh), interpolation=cv2.INTER_AREA)

    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(DIS_PRESET)
        _dis.setUseSpatialPropagation(True)
    flow = _dis.calc(a, b, None)
    if scale < 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return flow.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Vectorised block machinery
# --------------------------------------------------------------------------- #
def _grid(h: int, w: int, bs: int):
    nbh, nbw = h // bs, w // bs
    oy, ox = np.mgrid[0:bs, 0:bs]
    by, bx = np.mgrid[0:nbh, 0:nbw]
    return nbh, nbw, oy, ox, by, bx


def warp_plane(plane: np.ndarray, mv: np.ndarray, bs: int) -> np.ndarray:
    """Per-block translation warp of `plane` by integer field `mv` (nbh, nbw, 2)
    in (dx, dy) pixels.  One clipped gather, no Python loop over blocks."""
    if (plane.dtype == np.float32 and plane.flags.c_contiguous and bs >= 1
            and plane.shape[0] % bs == 0 and plane.shape[1] % bs == 0):
        try:
            from . import native
            out = native.warp_plane_f32(plane, mv, bs)
            if out is not None:
                return out
        except Exception:
            pass
    h, w = plane.shape
    nbh, nbw, oy, ox, by, bx = _grid(h, w, bs)
    pad = np.pad(plane, BORDER, mode='edge')

    dy = np.round(mv[:, :, 1]).astype(np.int32)[:, :, None, None]
    dx = np.round(mv[:, :, 0]).astype(np.int32)[:, :, None, None]
    srcy = np.clip(by[:, :, None, None] * bs + oy[None, None] + dy + BORDER, 0, pad.shape[0] - 1)
    srcx = np.clip(bx[:, :, None, None] * bs + ox[None, None] + dx + BORDER, 0, pad.shape[1] - 1)

    g = pad[srcy, srcx]                                  # (nbh, nbw, bs, bs)
    return g.transpose(0, 2, 1, 3).reshape(nbh * bs, nbw * bs)


def block_sad(cur: np.ndarray, pred: np.ndarray, bs: int) -> np.ndarray:
    """SAD per bs x bs block -> (nbh, nbw)."""
    h, w = cur.shape
    nbh, nbw = h // bs, w // bs
    d = np.abs(cur[:nbh * bs, :nbw * bs] - pred[:nbh * bs, :nbw * bs])
    return d.reshape(nbh, bs, nbw, bs).sum(axis=(1, 3)).astype(np.float64)


def _egolomb_bits_arr(v: np.ndarray) -> np.ndarray:
    """Exp-Golomb codeword length for an integer array (zigzag mapped)."""
    u = np.where(v < 0, (-v) * 2 - 1, v * 2).astype(np.int64)
    n = np.frexp((u + 1).astype(np.float64))[1].astype(np.int64)   # bit_length
    return (2 * n - 1).astype(np.float64)


def mv_bits(mv: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Bits to signal `mv` given spatial predictor `pred` -> (nbh, nbw)."""
    d = np.round(mv - pred).astype(np.int64)
    return (_egolomb_bits_arr(d[:, :, 0]) + _egolomb_bits_arr(d[:, :, 1]) + 2.0)


# --------------------------------------------------------------------------- #
#  Rate-distortion motion estimation
# --------------------------------------------------------------------------- #
def estimate_mv(cur: np.ndarray, ref: np.ndarray, seed: np.ndarray, bs: int,
                lam: float, search: int = 1):
    """Choose one integer MV per block, minimising J = SAD + lam * bits.

    Returns (mv float32 (nbh,nbw,2), sad (nbh,nbw), pred plane).
    """
    h, w = cur.shape
    nbh, nbw, _, _, _, _ = _grid(h, w, bs)
    curc = cur[:nbh * bs, :nbw * bs].astype(np.float64)
    refc = ref.astype(np.float64)

    seedq = np.round(seed[:nbh, :nbw]).astype(np.float32)
    zero = np.zeros_like(seedq)

    best_mv = seedq.copy()
    best_pred = warp_plane(refc, best_mv, bs)
    best_sad = block_sad(curc, best_pred, bs)
    # cost so far: seed is coded relative to itself -> ~2 bits
    best_j = best_sad + lam * mv_bits(best_mv, best_mv)

    def _try(cand):
        nonlocal best_mv, best_pred, best_sad, best_j
        p = warp_plane(refc, cand, bs)
        s = block_sad(curc, p, bs)
        j = s + lam * mv_bits(cand, best_mv)
        better = j < best_j
        if better.any():
            bb = better[:, :, None]
            best_mv = np.where(bb, cand, best_mv)
            best_sad = np.where(better, s, best_sad)
            best_j = np.where(better, j, best_j)
            best_pred = np.where(better[:, :, None, None],
                                 p.reshape(nbh, nbw, bs, bs),
                                 best_pred.reshape(nbh, nbw, bs, bs)
                                 ).reshape(nbh * bs, nbw * bs)

    # --- pass 1: global candidates + integer refinement of the seed -------- #
    _try(zero)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            if dx or dy:
                _try(seedq + np.array([dx, dy], dtype=np.float32))

    # --- pass 2: spatial predictors from pass 1 ---------------------------- #
    left = np.zeros_like(best_mv)
    top = np.zeros_like(best_mv)
    topleft = np.zeros_like(best_mv)
    left[:, 1:] = best_mv[:, :-1]
    top[1:, :] = best_mv[:-1, :]
    topleft[1:, 1:] = best_mv[:-1, :-1]
    for cand in (left, top, topleft):
        _try(cand)
    # one refinement round around the spatial winner
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            if dx or dy:
                _try(left + np.array([dx, dy], dtype=np.float32))

    pred = warp_plane(refc, best_mv, bs)
    return best_mv.astype(np.float32), block_sad(curc, pred, bs), pred


def upsample_mv(mv: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Scale a motion field to a finer resolution (both size and magnitude)."""
    th, tw = target_shape
    sh, sw = mv.shape[:2]
    if (th, tw) == (sh, sw):
        return mv.copy()
    out = cv2.resize(mv, (tw, th), interpolation=cv2.INTER_LINEAR)
    out[:, :, 0] *= tw / float(sw)
    out[:, :, 1] *= th / float(sh)
    return out


def downsample_mv(mv: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    sh, sw = mv.shape[:2]
    if (th, tw) == (sh, sw):
        return mv.copy()
    out = cv2.resize(mv, (tw, th), interpolation=cv2.INTER_AREA)
    out[:, :, 0] *= tw / float(sw)
    out[:, :, 1] *= th / float(sh)
    return out


# --------------------------------------------------------------------------- #
#  Skip decision
# --------------------------------------------------------------------------- #
def skip_threshold(qp: int, level: int = 0, chroma: bool = False) -> float:
    """Mean absolute residual per pixel below which a block is skipped.

    Tied to the quantisation step rather than hand-tuned: if the residual is
    smaller than what the quantiser rounds to zero anyway, transmitting it buys
    literally nothing.  0.42 is the expected mean of a uniform rounding error.

    It has to be the quantisation step of *this band*, not the base band's.  The
    bands are quantised progressively more coarsely as they get finer (that is
    the whole point of the perceptual weighting), so a fixed threshold borrowed
    from level 0 is far too tight for the upper bands: blocks whose residual
    would quantise to nothing are still transmitted, and the finest band -- the
    one that is supposed to be nearly free -- ends up being the expensive one.
    """
    from photon.transform import quant_matrix
    q = quant_matrix(level, qp, chroma)
    return 0.42 * float(q[0, 3])


def decide_skip(sad: np.ndarray, bs: int, thresh: float) -> np.ndarray:
    return (sad / float(bs * bs)) < thresh
