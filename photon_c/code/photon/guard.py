"""Defensive parsing for untrusted PHOTON streams (.phtn).

A decoder consumes attacker-controlled bytes.  Every struct.unpack / length
prefix / symbol count that comes out of a stream is a potential crash or
memory bomb, so all parsing of stream bytes goes through StreamError-raising
helpers here instead of raw struct calls.  Design rules:

  * never crash on any input:  every failure is a clean StreamError
  * never allocate attacker-influenced sizes before validating them
  * never loop attacker-influenced counts without a cap
  * the *writer* side of the format is untouched: guarding the reader
    cannot break format compatibility
"""
import struct

MAGIC = b"PHTN"
MAX_VERSION = 2

# hard resource ceilings (absolute, generous: 16K frames, 60 fps x 30 min)
MAX_W = 16384
MAX_H = 16384
MAX_PIXELS = MAX_W * MAX_H
MAX_FRAMES = 65535            # header field is u16 anyway
MAX_LEVELS = 9                # 2^9 = 512x downscale; nobody needs more
MAX_AMPS = 32                 # bands per frame never exceeds this
MAX_PLANES = 4                # y + 2 chroma (+future alpha)
MAX_PAYLOADS_PER_FRAME = MAX_PLANES * MAX_LEVELS
MAX_PACKET = 1 << 30          # single frame packet, 1 GB
MAX_SYMS = 1 << 22            # one rANS stream, 4M symbols
MAX_BLOCKS = MAX_PIXELS // 64


class StreamError(Exception):
    """Any problem with stream content: truncated, corrupt, hostile."""


def need(buf: bytes, off: int, nbytes: int, what: str = "field") -> None:
    """Bounds check before touching buf[off:off+nbytes]."""
    if off < 0 or off + nbytes > len(buf):
        raise StreamError(
            f"unexpected end of stream reading {what} "
            f"(need {nbytes} B at {off}, have {len(buf)})")


def rd(buf: bytes, fmt: str, off: int, what: str = "field") -> tuple:
    """struct.unpack_from with bounds checking and StreamError."""
    n = struct.calcsize(fmt)
    need(buf, off, n, what)
    try:
        return struct.unpack_from(fmt, buf, off)
    except struct.error as e:           # pragma: no cover - need() catches first
        raise StreamError(f"bad {what} at {off}: {e}") from e


def u16(buf: bytes, off: int, what: str = "u16") -> int:
    return rd(buf, ">H", off, what)[0]


def u32(buf: bytes, off: int, what: str = "u32") -> int:
    return rd(buf, ">I", off, what)[0]


def u8(buf: bytes, off: int, what: str = "u8") -> int:
    return rd(buf, ">B", off, what)[0]


def check_header_fields(meta: dict) -> dict:
    """Validate decoded stream-header scalars before any allocation."""
    w, h = int(meta["width"]), int(meta["height"])
    if not (8 <= w <= MAX_W and 8 <= h <= MAX_H and w * h <= MAX_PIXELS):
        raise StreamError(f"insane frame size {w}x{h} in header")
    nf = int(meta["nframes"])
    if not (0 <= nf <= MAX_FRAMES):
        raise StreamError(f"insane frame count {nf}")
    v = int(meta.get("n_levels", 0))
    if not (1 <= v <= MAX_LEVELS):
        raise StreamError(f"insane pyramid depth {v}")
    gop = int(meta.get("gop", 30))
    if not (1 <= gop <= 1000):
        raise StreamError(f"insane gop {gop}")
    for key in ("base_min_side", "chroma_min_side"):
        v = int(meta.get(key, 96))
        if not (8 <= v <= 2048):
            raise StreamError(f"insane {key} {v}")
    if not (0 <= int(meta.get("tx_depth", 0)) <= 16):
        raise StreamError("insane tx_depth")
    if not (1 <= int(meta.get("qp", 70)) <= 99):
        raise StreamError("insane qp")
    return meta
