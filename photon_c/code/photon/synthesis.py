"""
PHOTON-C  ::  Detail Synthesis Engine
=====================================
The part that makes "4K over a few tens of kilobits" arithmetically possible
rather than merely aspirational.

The premise
-----------
At 64 kbit/s, 30 fps, 3840x2160, each frame gets 267 bytes for 8.3 million
pixels: one byte per 31,000 pixels.  No transform, no prediction and no entropy
coder can deliver that from pixels -- the information is simply not there to be
coded.  So PHOTON-C does not transmit the pixels.  It transmits the *structure*
(a coarse base layer plus motion) and the decoder reconstructs the fine detail it
was never sent.

That is only defensible because of a specific fact about natural video: high
spatial-frequency content is overwhelmingly **texture**, and texture is
(a) locally stationary, (b) temporally persistent, and (c) perceptually tolerant
of being wrong in phase.  Nobody can tell that bark is being rendered from last
frame's bark shifted two pixels.  Everybody can tell that a face is blurred.  So:
synthesise the texture, transmit the structure.

Three mechanisms, combined
--------------------------
1. **Temporal detail borrowing.**  Warp the previously reconstructed frame's
   missing detail bands forward along the current motion field.  Detail visible
   last frame is usually still visible, just moved.  This cannot drift, because
   the encoder runs the identical synthesis inside its loop filter -- both sides
   borrow from the same reconstructed history.

2. **Edge-directed reconstruction.**  Where borrowing is unavailable (newly
   revealed areas, cuts, keyframes) the band is rebuilt from the level below with
   a guided filter using the decoded luma as guide.  This puts sharp transitions
   where the decoded structure says they belong instead of smearing them, which
   is the difference between a frame that looks "detailed" and one that looks
   "upscaled".

3. **Calibrated stochastic texture.**  A deterministic, frame-seeded noise field
   scaled to the measured band statistics.  Deterministic means it costs zero
   bits and is bit-identical on encoder and decoder; frame-seeded means the grain
   moves instead of freezing, which is what real texture does.

Energy calibration
------------------
Synthesising texture at the wrong amplitude is worse than not synthesising it:
too much reads as grain over everything, too little reads as blur.  So the
encoder measures the RMS of each band it is *not* going to transmit -- it has the
original, so this is free -- and writes one small amplitude per level per frame.
A handful of bytes buys correctly-exposed texture at every bitrate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BORROW_GAIN = 0.86      # how much of the borrowed detail to keep
NOISE_GAIN = 0.55       # stochastic amplitude relative to calibrated RMS
GUIDE_RADIUS = 4
GUIDE_EPS = 48.0
GUIDE_MIX = 0.85


# --------------------------------------------------------------------------- #
#  Laplacian pyramid, indexed by resolution
# --------------------------------------------------------------------------- #
def build_pyramid(plane: np.ndarray,
                  shapes: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Laplacian decomposition at the resolutions given by `shapes`.

    `shapes` is [base, band0, ..., full resolution], coarsest first, with
    shapes[-1] equal to the (already padded) plane size.  The caller supplies it
    rather than this function deriving it, because the size chain is geometry that
    the encoder, the decoder, the bitstream header and the coder all have to agree
    on exactly.  Deriving it locally -- even with the same formula -- is how the
    codec ends up with three chains that round differently: this one used to floor
    each halving to a block multiple while the frame layer ceil'd it, so for a
    184-row plane this produced a 16-row base where the frame layer expected 24,
    and the reference and the source disagreed in size from frame 1 onward.

    Returns (base, details) where details[k] is the band at shapes[k + 1], so
    details[-1] has exactly the full resolution.
    """
    if not shapes:
        raise ValueError("build_pyramid needs at least one shape")
    src = plane.astype(np.float64)
    if src.shape[:2] != tuple(shapes[-1]):
        src = cv2.resize(src, (shapes[-1][1], shapes[-1][0]),
                         interpolation=cv2.INTER_AREA)

    # downsample chain, finest -> coarsest, at exactly the requested sizes
    chain = [src]
    for (hh, ww) in reversed(shapes[:-1]):
        chain.append(cv2.resize(chain[-1], (ww, hh), interpolation=cv2.INTER_AREA))
    chain.reverse()                                   # chain[0] is the base

    base = chain[0]
    details: List[np.ndarray] = []
    for k in range(1, len(chain)):
        up = cv2.resize(chain[k - 1], (chain[k].shape[1], chain[k].shape[0]),
                        interpolation=cv2.INTER_LANCZOS4)
        details.append(chain[k] - up)
    return base, details


def collapse_pyramid(base: np.ndarray, details: List[np.ndarray]) -> np.ndarray:
    """Inverse of build_pyramid: upsample-and-add back up the chain."""
    cur = base.astype(np.float64)
    for d in details:
        cur = cv2.resize(cur, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cur = cur + d
    return cur


def band_rms(details: List[np.ndarray]) -> np.ndarray:
    return np.array([float(np.sqrt(np.mean(d ** 2))) for d in details], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Edge-directed reconstruction
# --------------------------------------------------------------------------- #
def guided_detail(coarse_up: np.ndarray, guide: Optional[np.ndarray],
                  radius: int = GUIDE_RADIUS, eps: float = GUIDE_EPS,
                  mix: float = GUIDE_MIX) -> np.ndarray:
    """Recover detail by guided-filtering `coarse_up` with `guide`.

    Returns the *detail* component (not the filtered image), so it can be added
    to a band.  With guide=None this returns zeros and the caller falls back to
    plain interpolation.

    `guide` must carry information `coarse_up` does not.  Passing the same array
    as both arguments is not a no-op and not harmless: the guided filter becomes
    a ~1.5x unsharp mask of its own input, so this returns roughly `coarse_up`
    again as "detail", which then gets added to `coarse_up`.  The band comes out
    about twice as hot as it should be with none of the information that was
    supposed to justify it -- measurably worse PSNR than not synthesising at all.
    Callers must pass a different source (a warped previous band, or a decoded
    lower-resolution band upsampled to this one).
    """
    if guide is None or guide is coarse_up:
        return np.zeros_like(coarse_up)
    if guide.shape != coarse_up.shape:
        guide = cv2.resize(guide, (coarse_up.shape[1], coarse_up.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    g = np.clip(guide, 0, 255).astype(np.float32)
    p = coarse_up.astype(np.float32)
    r = (radius, radius)
    mean_i = cv2.boxFilter(g, cv2.CV_32F, r)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, r)
    corr_i = cv2.boxFilter(g * g, cv2.CV_32F, r)
    corr_ip = cv2.boxFilter(g * p, cv2.CV_32F, r)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, r)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, r)
    q = mean_a * g + mean_b
    return mix * (q.astype(np.float64) - p.astype(np.float64))


def deterministic_noise(shape: Tuple[int, int], seed: int) -> np.ndarray:
    """Zero-mean unit-variance noise that is a pure function of (shape, seed).

    Encoder and decoder call this with identical arguments, so the texture is
    bit-identical without transmitting a single bit.  The seed mixes frame index
    with level index: successive frames get different grain (otherwise texture
    visibly freezes) while staying perfectly reproducible.

    Slightly band-limited rather than white, because white noise reads as TV
    static while filtered noise reads as film grain or sensor texture, which is
    what real high-frequency image content actually looks like.
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n = rng.standard_normal(shape)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    s = float(n.std())
    return n / s if s > 1e-9 else n


# --------------------------------------------------------------------------- #
#  Synthesis of one missing band
# --------------------------------------------------------------------------- #
def warp_band(band: np.ndarray, motion: Optional[np.ndarray],
              shape: Tuple[int, int]) -> np.ndarray:
    """Warp a detail band forward along a motion field, rescaling the field to
    the band's resolution.  `motion` is (h, w, 2) in (dx, dy) at its own scale."""
    th, tw = shape
    d = band
    if d.shape != (th, tw):
        d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_LINEAR)
    if motion is None:
        return d.astype(np.float64)
    mv = motion
    if mv.shape[:2] != (th, tw):
        sx = tw / float(mv.shape[1])
        sy = th / float(mv.shape[0])
        mv = cv2.resize(mv, (tw, th), interpolation=cv2.INTER_LINEAR).copy()
        mv[:, :, 0] *= sx
        mv[:, :, 1] *= sy
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    mapx = np.clip(xx - mv[:, :, 0], 0, tw - 1)
    mapy = np.clip(yy - mv[:, :, 1], 0, th - 1)
    return cv2.remap(d.astype(np.float32), mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float64)


def synthesise_band(prev_band: Optional[np.ndarray],
                    motion: Optional[np.ndarray],
                    coarse_up: np.ndarray,
                    shape: Tuple[int, int],
                    amplitude: float,
                    seed: int,
                    guide: Optional[np.ndarray] = None,
                    borrow_gain: float = BORROW_GAIN,
                    noise_gain: float = NOISE_GAIN) -> np.ndarray:
    """Reconstruct one detail band that was never transmitted.

    The governing rule: **only add energy that is correlated with the truth.**
    Uncorrelated energy is by definition pure MSE -- it cannot help, and the
    earlier version of this function violated the rule twice over.  It
    renormalised the result to the encoder's measured RMS regardless of whether
    any informative term existed, so on a keyframe, with no history to borrow, it
    scaled a noise field up to full band amplitude and injected roughly 9 RMS of
    pure error per band.  Measured cost: 24.1 dB for a plain Lanczos upscale of
    the base versus 21.6 dB for the "synthesised" version.  The synthesis engine
    was making the picture worse than doing nothing.

    So the amplitude the encoder transmits *scales* the informative terms; it
    never manufactures content on its own.  With nothing to borrow and nothing to
    extrapolate from, the correct output is zero and the picture degrades
    gracefully to a soft-but-clean upscale.
    """
    th, tw = shape
    out = np.zeros((th, tw), dtype=np.float64)

    # (1) temporal borrowing -- the dominant term, and by a wide margin.
    #     Measured on real footage: consecutive frames' detail bands correlate at
    #     0.97-0.999, and the optimally-gained previous band predicts the current
    #     one with 12-29 dB less MSE than the band itself carries.  High spatial
    #     frequency content in natural video is overwhelmingly static texture
    #     that only appears to move because the camera or the subject does.
    if prev_band is not None:
        d = warp_band(prev_band, motion, (th, tw))

        # (2) edge-directed correction of the borrow, guided by the decoded
        #     structure.  The guide here genuinely carries information the
        #     borrowed band does not -- it is this frame's edges, not last
        #     frame's -- so the guided filter is doing real work: shifting the
        #     borrowed texture onto the current geometry instead of leaving it
        #     where the motion estimate put it.
        if guide is not None and guide is not coarse_up:
            out += d + guided_detail(d, guide)
        else:
            out += d
        out *= borrow_gain

    # (3) calibrated stochastic texture, but only where the band above says
    #     there IS texture.  Grain over a flat sky is visible damage; grain over
    #     bark is invisible and stops the surface looking plastic.  Modulating by
    #     the local energy of what was already synthesised keeps it in the second
    #     regime.  This term costs PSNR by construction and is here for
    #     perceptual reasons, so it is deliberately small and switchable.
    if noise_gain > 0 and amplitude > 1e-6 and float(np.abs(out).max()) > 1e-6:
        n = deterministic_noise((th, tw), seed)
        w = np.minimum(np.abs(out) / max(amplitude, 1e-6), 1.0)
        out += noise_gain * amplitude * n * w

    return out


# --------------------------------------------------------------------------- #
#  Per-decoder history of synthesised bands
# --------------------------------------------------------------------------- #
class SynthHistory:
    """Borrowed-detail history.  Encoder and decoder each own one and both reset
    it at keyframes, so it can never drift between them."""

    def __init__(self):
        self.bands: Dict[Tuple[str, int], np.ndarray] = {}

    def reset(self) -> None:
        self.bands.clear()

    def get(self, plane: str, level: int) -> Optional[np.ndarray]:
        return self.bands.get((plane, level))

    def put(self, plane: str, level: int, band: np.ndarray) -> None:
        self.bands[(plane, level)] = band


def synthesise_upper_bands(decoded_bands: List[Optional[np.ndarray]],
                           base: np.ndarray,
                           shapes: List[Tuple[int, int]],
                           amplitudes: np.ndarray,
                           motion_full: Optional[np.ndarray],
                           frame_idx: int,
                           keyframe: bool,
                           plane: str,
                           history: SynthHistory,
                           borrow_gain: float = BORROW_GAIN,
                           noise_gain: float = NOISE_GAIN,
                           guide: Optional[np.ndarray] = None
                           ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Fill every None entry in `decoded_bands` by synthesis, bottom-up.

    `shapes` is [base_shape, detail0_shape, ..., detailN_shape] with detailN at
    full resolution.  Returns (complete_bands, full_resolution_plane).
    """
    bands = list(decoded_bands)
    cur = base.astype(np.float64)

    for k in range(len(bands)):
        L = k + 1                       # level index; 0 is the base
        shape = shapes[L]
        if L >= len(shapes):
            break
        coarse_up = cv2.resize(cur, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if bands[k] is None:
            mv = None
            if motion_full is not None:
                mv = motion_full
            amp = float(amplitudes[k]) if k < len(amplitudes) else 0.0
            prev = None if keyframe else history.get(plane, L)
            band = synthesise_band(prev, mv, coarse_up, shape, amp,
                                   seed=(frame_idx * 977 + L * 31 + 7),
                                   guide=guide,
                                   borrow_gain=borrow_gain, noise_gain=noise_gain)
            bands[k] = band
        else:
            b = bands[k]
            if b.shape != shape:
                b = cv2.resize(b, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)
                bands[k] = b

        history.put(plane, L, bands[k])
        cur = coarse_up + bands[k]

    return bands, cur
