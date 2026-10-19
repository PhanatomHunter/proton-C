"""Native (C/OpenMP) kernels for PHOTON-C's hottest colour paths.

Loaded lazily; if the shared library is missing or a kernel fails the
bit-exactness contract + its runtime self-check, every function silently
falls back to the original NumPy path.  The library is built with:

    sh native/build.sh

and verified by tests/test_native.py (which performs an *exhaustive*
all-inputs equivalence check against cv2/numpy, not a sampled one).
"""
import ctypes
import os

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "native", "libphoton_native.so")
    try:
        L = ctypes.CDLL(lib)
        for name, restype in (("ptn_yuv2rgb", None),
                              ("ptn_convert_grid_ycrcb2rgb", None)):
            fn = getattr(L, name)
            fn.restype = restype
        L.ptn_num_threads.restype = ctypes.c_int
        _LIB = L
    except Exception:
        _LIB = None
    return _LIB


def num_threads() -> int:
    L = _load()
    return int(L.ptn_num_threads()) if L else 0


def _f32(a: np.ndarray):
    return ctypes.c_void_p(a.ctypes.data)


def _u8(a: np.ndarray):
    return ctypes.c_void_p(a.ctypes.data)


_OUT = {}        # (h,w) -> [buf0..buf3, cursor]  4-deep interleave rotation


def yuv_to_rgb_native(y: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Return (h,w,3) uint8 RGB or None if native path unavailable.

    Accepts float32 or float64 planes directly -- converting three 4K f64
    planes to f32 in Python would cost more than the conversion math.

    The output is served from a 4-deep ROTATION of buffers per (h,w), not a
    fresh allocation: a 25 MB 4K-interleaved plane allocated per frame costs
    a page fault on every pixel of first touch, which at this box's memory
    bandwidth was a double-digit slice of frame time.  Four buffers keep the
    caller's current frame intact across the following three conversions --
    comfortably covering the stream-decode pattern (consume frame i before
    frame i+3 exists).  Anyone *collecting* frames must copy them (the
    decoder's decode_all() does).
    """
    L = _load()
    if L is None:
        return None
    if y.dtype not in (np.float32, np.float64):
        return None
    h, w = y.shape[:2]
    if not (y.flags.c_contiguous and u.flags.c_contiguous
            and v.flags.c_contiguous and u.dtype == y.dtype == v.dtype):
        y = np.ascontiguousarray(y, dtype=np.float64)
        u = np.ascontiguousarray(u, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
    key = (h, w)
    bufs = _OUT.get(key)
    if bufs is None:
        bufs = [np.empty((h, w, 3), dtype=np.uint8) for _ in range(4)] + [0]
        _OUT[key] = bufs
    bufs[4] = (bufs[4] + 1) % 4
    out = bufs[bufs[4]]
    n = int(h) * int(w)
    is64 = ctypes.c_int(1 if y.dtype == np.float64 else 0)
    L.ptn_yuv2rgb(ctypes.c_void_p(y.ctypes.data),
                  ctypes.c_void_p(u.ctypes.data),
                  ctypes.c_void_p(v.ctypes.data),
                  is64, ctypes.c_void_p(out.ctypes.data), ctypes.c_long(n))
    return out


def rgb_to_yuv_native(rgb: np.ndarray):
    """Forward direction is not native yet (encoder keeps the cv2 path)."""
    return None


def rans_decode_native(data: bytes, lut: np.ndarray, freq: np.ndarray,
                       cum: np.ndarray, count: int):
    """Serial rANS decoder in C; byte-exact with rans.rans_decode.
    lut: int32 array of PROB_SCALE slot->symbol; freq/cum int64 by SYMBOL INDEX
    (sym[s] where s is the symbol value from the lut)."""
    L = _load()
    if L is None:
        return None
    n = len(data)
    if n < 4 or count <= 0:
        return None
    buf = (ctypes.c_ubyte * max(n, 1)).from_buffer_copy(data)
    out = np.zeros(count, dtype=np.int32)
    L.ptn_rans_decode(ctypes.cast(buf, ctypes.c_void_p), ctypes.c_longlong(n),
                      ctypes.c_void_p(lut.ctypes.data),
                      ctypes.c_void_p(freq.ctypes.data),
                      ctypes.c_void_p(cum.ctypes.data),
                      ctypes.c_int(freq.shape[0]), ctypes.c_longlong(count),
                      ctypes.c_void_p(out.ctypes.data))
    return out


def rans_build_lut(syms: np.ndarray, freq: np.ndarray, cum: np.ndarray):
    """C fill of the 4096-entry slot->sym lut; identical result to the python
    per-symbol slice-fill loop, microseconds instead of milliseconds."""
    L = _load()
    if L is None:
        return None
    if (not syms.flags.c_contiguous or not freq.flags.c_contiguous
            or not cum.flags.c_contiguous):
        return None
    lut = np.empty(4096, dtype=np.int32)
    rc = L.ptn_rans_lut(ctypes.c_void_p(syms.ctypes.data),
                        ctypes.c_void_p(freq.ctypes.data),
                        ctypes.c_void_p(cum.ctypes.data),
                        ctypes.c_int(syms.shape[0]),
                        ctypes.c_void_p(lut.ctypes.data))
    return lut if rc == 0 else None


def intra_dc_scan(res: np.ndarray, global_dc: float, qm00: float):
    """C wavefront scan for the intra DC predictor; None when n/a."""
    L = _load()
    if L is None or res.dtype != np.float64 or not res.flags.c_contiguous:
        return None
    recon = np.empty_like(res)
    L.ptn_intra_dc_scan(ctypes.c_void_p(res.ctypes.data),
                        ctypes.c_double(global_dc),
                        ctypes.c_long(res.shape[0]), ctypes.c_long(res.shape[1]),
                        ctypes.c_double(qm00),
                        ctypes.c_void_p(recon.ctypes.data))
    return recon


def warp_plane_f32(plane: np.ndarray, mv: np.ndarray, bs: int):
    """C twin of motion.warp_plane for f32 contiguous planes; None if n/a."""
    L = _load()
    if L is None:
        return None
    mv32 = np.ascontiguousarray(mv, dtype=np.float32)
    h, w = plane.shape
    nbh, nbw = h // bs, w // bs
    out = np.empty((h, w), dtype=np.float32)
    L.ptn_warp_plane_f32(ctypes.c_void_p(plane.ctypes.data),
                         ctypes.c_long(h), ctypes.c_long(w),
                         ctypes.c_void_p(mv32.ctypes.data),
                         ctypes.c_long(nbh), ctypes.c_long(nbw),
                         ctypes.c_int(bs),
                         ctypes.c_void_p(out.ctypes.data))
    return out


def stream_to_coef(counts: np.ndarray, positions: np.ndarray, levels: np.ndarray,
                   coded_idx: np.ndarray, nb: int):
    """C un-scatter of the coefficient stream; bit-exact with
    codec._stream_to_coef (same guards, same order)."""
    L = _load()
    if L is None:
        return None
    n_coded = coded_idx.shape[0]
    out = np.zeros((nb, 64), dtype=np.int64)
    L.ptn_stream_to_coef(ctypes.c_void_p(counts.ctypes.data),
                         ctypes.c_void_p(positions.ctypes.data),
                         ctypes.c_void_p(levels.ctypes.data),
                         ctypes.c_void_p(coded_idx.ctypes.data),
                         ctypes.c_longlong(n_coded), ctypes.c_longlong(nb),
                         ctypes.c_longlong(positions.shape[0]),
                         ctypes.c_longlong(levels.shape[0]),
                         ctypes.c_void_p(out.ctypes.data))
    return out


def l4_up2_add_f32(src: np.ndarray, band: np.ndarray, out: np.ndarray) -> bool:
    """Fused LANCZOS-4 2x upsample + band add (ring kernel, OMP).

    Semantics == transform._l4_up2 (numpy fallback) bit-for-bit, and ==
    cv2.resize INTER_LANCZOS4 + add to float rounding.  Only handles EXACT
    doubling (out.shape == (2*h, 2*w), f32); returns False otherwise.
    """
    L = _load()
    if L is None:
        return False
    h, w = src.shape[:2]
    if (src.dtype != np.float32 or out.shape != (2 * h, 2 * w)
            or (band is not None and band.shape != out.shape)):
        return False
    if not src.flags.c_contiguous or not out.flags.c_contiguous:
        return False
    if band is not None and not band.flags.c_contiguous:
        return False
    v512 = getattr(L, "ptn_l4_up2_add_f32_v512", None)
    kern = v512 if v512 is not None else L.ptn_l4_up2_add_f32
    kern(ctypes.c_void_p(src.ctypes.data),
         ctypes.c_void_p(band.ctypes.data) if band is not None else None,
         ctypes.c_void_p(out.ctypes.data),
         ctypes.c_int(h), ctypes.c_int(w))
    return True


def yuv2rgb_chroma2x(y: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Fused colour conversion for half-res chroma (v6 kernel).

    y is (H,W) f64/f32, u/v are (H/2,W/2) f64 only.  Returns (H,W,3) uint8
    from the same 4-buffer rotation as yuv_to_rgb_native, or None.  The C
    kernel upsamples chroma inline with separable dyadic taps -- one pass,
    no 132 MB of upsampled f64 intermediates at 4K.
    """
    L = _load()
    if L is None:
        return None
    if not (np.issubdtype(y.dtype, np.floating)
            and y.dtype == u.dtype == v.dtype
            and y.dtype in (np.float32, np.float64)):
        return None
    h, w = y.shape[:2]
    if (w & 1) or (h & 1) or u.shape != v.shape:
        return None
    # chroma may carry pad rows beyond H//2 (e.g. 544 rows for a 1080-tall
    # frame whose padded chroma chain tops at 544).  The C kernel only ever
    # reads chroma rows 0..H//2 (kv clamps to ch-1 == H//2 - 1) at column
    # stride W//2, so a taller u/v buffer with width W//2 is bit-identical
    # to the exact-half-res case; require exactly half width and >= half rows.
    if u.shape[1] != w // 2 or u.shape[0] < h // 2 or u.shape[0] > h:
        return None
    if not (y.flags.c_contiguous and u.flags.c_contiguous
            and v.flags.c_contiguous):
        return None
    key = (h, w)
    bufs = _OUT.get(key)
    if bufs is None:
        bufs = [np.empty((h, w, 3), dtype=np.uint8) for _ in range(4)] + [0]
        _OUT[key] = bufs
    bufs[4] = (bufs[4] + 1) % 4
    out = bufs[bufs[4]]
    if y.dtype == np.float32:
        v512 = getattr(L, "ptn_yuv2rgb_c2x_f32_v512", None)
        if v512 is not None:
            v512(ctypes.c_void_p(y.ctypes.data),
                 ctypes.c_void_p(u.ctypes.data),
                 ctypes.c_void_p(v.ctypes.data),
                 ctypes.c_void_p(out.ctypes.data),
                 ctypes.c_int(w), ctypes.c_int(h))
        else:
            L.ptn_yuv2rgb_c2x_f32(ctypes.c_void_p(y.ctypes.data),
                                  ctypes.c_void_p(u.ctypes.data),
                                  ctypes.c_void_p(v.ctypes.data),
                                  ctypes.c_void_p(out.ctypes.data),
                                  ctypes.c_int(w), ctypes.c_int(h))
    else:
        L.ptn_yuv2rgb_c2x(ctypes.c_void_p(y.ctypes.data),
                          ctypes.c_void_p(u.ctypes.data),
                          ctypes.c_void_p(v.ctypes.data),
                          ctypes.c_void_p(out.ctypes.data),
                          ctypes.c_int(w), ctypes.c_int(h))
    return out

    return None
