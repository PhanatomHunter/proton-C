"""
Streaming benchmark harness: PHOTON-C against libx264 / libx265 / libaom-av1.

Why this is disk-backed rather than in-memory
--------------------------------------------
Two reasons, both learned the hard way.

*Rate control needs a clip long enough to converge.*  An 8-frame 4K clip is
dominated by its single I-frame, and x265 given `-b:v 350k` on it produced
10,767 kb/s -- it never had a chance to reach the target.  Measuring a codec's
efficiency at a bitrate it could not actually hit is not a comparison, it is a
category error.  So the benchmarks here run on clips long enough for every
encoder's rate control to settle.

*A 4K frame is 25 MB.*  Holding 60 of them plus their reconstructions plus the
metric intermediates does not fit in this sandbox's ~1.4 GB.  So frames live in a
raw I420 file on disk and are streamed one at a time, with quality metrics
accumulated incrementally instead of over a list of decoded frames.

Both codecs read the *same* file, so there is no possibility of one side getting
an easier input than the other.  That file is also the ground truth for metrics:
references are decoded back to I420 and compared against it, and PHOTON-C's
decoder output is compared against it the same way.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photon.codec import Config, preset                       # noqa: E402
from photon.frame import Decoder, Encoder, luma_shapes        # noqa: E402

# Scratch space must be on the real disk, not /tmp.  In this sandbox /tmp is a
# 992 MB RAM-backed tmpfs: a 600 MB raw 4K source there both fails to fit and
# silently consumes memory the encoders need, and leftover benchmark temp
# directories keep holding it after the run.  Setting this at import time means
# every tempfile.mkdtemp() in the harness lands on disk.
_SCRATCH = "/home/user/photon/out/tmp"
os.makedirs(_SCRATCH, exist_ok=True)
os.environ["TMPDIR"] = _SCRATCH
tempfile.tempdir = _SCRATCH

FFMPEG: Optional[str] = None


def ffmpeg_exe() -> Optional[str]:
    global FFMPEG
    if FFMPEG:
        return FFMPEG
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMPEG
    except Exception:
        pass
    for c in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if shutil.which(c):
            FFMPEG = shutil.which(c)
            return FFMPEG
    return None


def run(cmd: List[str], timeout: int = 7200) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on the luma-weighted grayscale, 11x11 uniform window.

    Implemented directly rather than pulling skimage, so the number is
    reproducible from numpy alone and matches the standard formulation.  Computed
    in float32 with intermediates folded in place: in float64 with every term held
    live, an 11x11-window SSIM on a 4K plane needs eight 66 MB arrays at once --
    over half a gigabyte, which is what pushed a 1.4 GB sandbox to the edge of an
    OOM.
    """
    from scipy.ndimage import uniform_filter
    x = a if a.ndim == 2 else 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    y = b if b.ndim == 2 else 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    C1 = np.float32((0.01 * 255) ** 2)
    C2 = np.float32((0.03 * 255) ** 2)
    f = lambda z: uniform_filter(z, 11, mode="reflect")
    mx, my = f(x), f(y)
    vx = f(x * x); vx -= mx * mx
    vy = f(y * y); vy -= my * my
    cov = f(x * y); cov -= mx * my
    num = (2 * mx * my + C1) * (2 * cov + C2)
    den = (mx * mx + my * my + C1) * (vx + vy + C2)
    out = float(np.mean(num / den))
    del x, y, mx, my, vx, vy, cov, num, den
    return out


class MetricAcc:
    """Accumulates per-frame quality so frames never have to be held in a list."""

    def __init__(self, want_ssim: bool = True):
        self.psnr: List[float] = []
        self.ssim: List[float] = []
        self.want_ssim = want_ssim

    def add(self, src: np.ndarray, rec: np.ndarray) -> None:
        n = min(src.shape[0], rec.shape[0]), min(src.shape[1], rec.shape[1])
        self.psnr.append(psnr(src[:n[0], :n[1]], rec[:n[0], :n[1]]))
        if self.want_ssim:
            self.ssim.append(ssim(src[:n[0], :n[1]], rec[:n[0], :n[1]]))

    def result(self) -> dict:
        return dict(psnr=float(np.mean(self.psnr)),
                    psnr_min=float(np.min(self.psnr)),
                    ssim=float(np.mean(self.ssim)) if self.ssim else None)


# --------------------------------------------------------------------------- #
#  I420 frame source (streamed from disk)
# --------------------------------------------------------------------------- #
class YuvSource:
    """Random-access stream of RGB frames backed by a raw I420 file.

    I420 stores Y as h*w bytes then U and V as h/4*w bytes each, so a whole frame
    is exactly a (3h/2, w) byte image and cv2.cvtColor's I420 mode consumes it
    directly -- no manual plane slicing, which is where a reshape goes wrong.
    """

    def __init__(self, path: str, w: int, h: int, nframes: int, fps: float = 30.0):
        self.path, self.w, self.h, self.nframes, self.fps = path, w, h, nframes, fps
        self.fsz = h * w * 3 // 2

    def __len__(self) -> int:
        return self.nframes

    def __getitem__(self, i: int) -> np.ndarray:
        with open(self.path, "rb") as fh:
            fh.seek(int(i) * self.fsz)
            buf = fh.read(self.fsz)
        if len(buf) < self.fsz:
            raise IndexError(f"frame {i} past end of {self.path}")
        yuv = np.frombuffer(buf, np.uint8).reshape(self.h * 3 // 2, self.w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return bgr[:, :, ::-1].copy()

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.nframes):
            yield self[i]


def write_yuv_frame(fh, rgb: np.ndarray) -> None:
    h, w = rgb.shape[:2]
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, ::-1]), cv2.COLOR_BGR2YUV_I420)
    fh.write(yuv.tobytes())


def write_yuv(frames, path: str) -> None:
    with open(path, "wb") as fh:
        for f in frames:
            write_yuv_frame(fh, f)


# --------------------------------------------------------------------------- #
#  Sources
# --------------------------------------------------------------------------- #
def fractal_texture(height: int, width: int, rng,
                    cells=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                    exp: float = 0.3) -> np.ndarray:
    """A texture field with roughly the spectrum natural images have.

    Octaves of noise summed with weight `cell ** exp`.  exp = 0 is equal energy
    per octave, exp = 1 is heavily low-pass; 0.3 was chosen by calibration, and
    the calibration is the reason this function exists in this form.

    The honest way to check a synthetic source is to ask how much survives
    downsampling, because that is exactly what a base-layer codec does to it.
    Measured base-only reconstruction ceilings (lossless base, so purely a
    property of the source):

        real Big Buck Bunny 1080p   1/4: 27.4 dB   1/8: 25.0 dB   1/15: 23.2 dB
        real Sintel 1080p           1/4: 35.4 dB   1/8: 30.3 dB   1/15: 27.3 dB
        this field, exp = 0.0       1/4: 21.6 dB   1/8: 20.3 dB
        this field, exp = 0.3       1/4: ~35 dB    1/8: ~32 dB      <- chosen
        this field, exp = 0.5       1/4: 38.3 dB   1/8: 34.9 dB
        this field, exp = 1.0       1/4: 57.9 dB   1/8: 51.5 dB

    Two earlier versions were both wrong, in opposite directions, and both would
    have produced confidently meaningless benchmark numbers.  The first used a
    single tile of *white* noise: equal energy at every frequency up to Nyquist, so
    downsampling 8x retained almost nothing and the ceiling was 18.7 dB -- harder
    than any real source, which made every codec including the references look
    terrible and made PHOTON-C's quality look flat in bitrate, because no base
    size could have captured that content.  The second weighted octaves by cell
    size (exp = 1), far too smooth: a 51 dB ceiling means the base layer alone
    nearly reconstructs the frame and the whole premise goes untested.

    A test source that is harder than reality does not make a result conservative.
    It makes it meaningless, and it does so silently.

    The finest octaves (cell 1 and 2) are included because real 4K carries
    pixel-scale sensor grain; without them the top pyramid band holds 0.2% of the
    detail energy where real footage holds ~18%.
    """
    acc = np.zeros((height, width), dtype=np.float32)
    for cell in cells:
        if cell <= 1:
            up = rng.normal(0, 1, (height, width)).astype(np.float32)
        else:
            n = rng.normal(0, 1, (max(2, height // cell + 2),
                                  max(2, width // cell + 2))).astype(np.float32)
            up = cv2.resize(n, (width, height), interpolation=cv2.INTER_CUBIC)
            del n
        acc += (float(cell) ** exp) * up
        del up
    acc -= acc.mean()
    acc /= max(float(acc.std()), 1e-6)
    return acc


def make_source_yuv(path: str, nframes: int, width: int, height: int,
                    fps: float, kind: str = "synthetic4k",
                    src_video: Optional[str] = None,
                    seed: int = 7, strip: int = 120) -> None:
    """Write `nframes` frames of a source to a raw I420 file, one strip at a time.

    `kind` is either "synthetic4k" (generated, see fractal_texture for why) or
    "video" (decoded from src_video).  Generated content is a weaker claim than
    real footage and is labelled as such wherever these numbers appear; what it is
    still valid for is the *ratio* between codecs at matched bitrate, since every
    codec reads this same file.
    """
    if kind == "video":
        if not src_video:
            raise ValueError("kind='video' needs src_video")
        cap = cv2.VideoCapture(src_video)
        with open(path, "wb") as fh:
            n = 0
            while n < nframes:
                ok, f = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                    if not ok:
                        break
                f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
                write_yuv_frame(fh, f[:, :, ::-1])
                n += 1
        cap.release()
        return

    rng = np.random.default_rng(seed)
    tex = fractal_texture(height, width, rng)
    xs = np.arange(width, dtype=np.float32)
    with open(path, "wb") as fh:
        for i in range(nframes):
            t = i / fps
            dx = 40.0 * np.sin(2 * np.pi * t * 0.13)
            dy = 25.0 * np.cos(2 * np.pi * t * 0.09)
            img = np.empty((height, width, 3), dtype=np.uint8)
            texf = np.roll(tex, (int(dy * 3), int(dx * 3)), axis=(0, 1))
            for y0 in range(0, height, strip):
                y1 = min(y0 + strip, height)
                ys = np.arange(y0, y1, dtype=np.float32)[:, None]
                xr = xs[None, :]
                r = (96 + 60 * np.sin((xr + dx * 6) / 620.0)
                     + 30 * np.cos((ys + dy * 6) / 480.0))
                g = (110 + 55 * np.cos((xr - dx * 5) / 540.0 + 1.1)
                     + 25 * np.sin((ys + dy * 5) / 400.0))
                b = 128 + 50 * np.sin((xr + ys + dx * 4) / 700.0 + 2.0)
                # a hard-edged moving object: what exposes a blurry codec
                obx = width * 0.5 + 500 * np.sin(2 * np.pi * t * 0.21)
                oby = height * 0.5 + 260 * np.cos(2 * np.pi * t * 0.17)
                m = ((xr - obx) ** 2 + (ys - oby) ** 2) < 240.0 ** 2
                r = np.where(m, 225.0, r); g = np.where(m, 70.0, g); b = np.where(m, 55.0, b)
                sh = texf[y0:y1]
                r += 34 * sh; g += 30 * sh; b += 26 * sh
                img[y0:y1] = np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
                del r, g, b, m, sh
            del texf
            write_yuv_frame(fh, img)
            del img
    del tex


def load_clip(path: str, nframes: int, width: int, height: int,
              fps: float = 30.0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FPS, fps)
    out = []
    while len(out) < nframes:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
        out.append(f[:, :, ::-1].copy())
    cap.release()
    return out


# --------------------------------------------------------------------------- #
#  Reference encoders
# --------------------------------------------------------------------------- #
# Every reference gets the same GOP length as PHOTON-C so the keyframe amortisation
# is identical, and two-pass ABR so each actually hits its target bitrate.
GOP = 60

# Reference settings, discovered empirically on this sandbox (2 CPU, 2 GB total,
# ~1.6 GB actually free once stray processes and the RAM-backed /tmp are cleared).
#
# Two variants per codec, because the honest answer depends on the resolution:
#
#   "-full"  the encoder's own good settings (x264 preset slow -tune psnr, x265
#            preset medium, AV1 cpu-used 4).  These run fine at 1080p and are what
#            the codec-efficiency table should be read from.
#   plain    reduced-memory settings, needed at 4K where frame buffers dominate.
#            x264 slow with ref=2/rc-lookahead=10/bframes=2, x265 medium with
#            pools=none/frame-threads=1/ref=1/rc-lookahead=5/no-wpp.
#
# The reduced settings are weaker than the encoders' defaults, which favours
# PHOTON-C, so any 4K comparison using them is stated with that caveat attached.
# They also overshoot their bitrate targets (measured: x264 ~1.24x, x265 ~3.4x,
# because a five-frame lookahead cannot do rate control properly).  That is why
# results are reported as rate-distortion *points* -- measured kilobits per second
# against measured quality -- and compared by interpolating at matched actual
# rate, never by assuming an encoder hit the number it was asked for.
#
# A hard per-command timeout is not optional here.  An encoder being OOM-thrashed
# rather than killed burns wall clock indefinitely: one such probe cost 984 s,
# more than the entire rest of the benchmark.
REF_ENCODERS = {
    "libx264": dict(codec="libx264", npass=2, timeout=420,
                    extra=["-preset", "slow",
                           "-x264-params", "ref=2:rc-lookahead=10:bframes=2"]),
    "libx265": dict(codec="libx265", npass=2, timeout=600,
                    extra=["-preset", "medium",
                           "-x265-params", "pools=none:frame-threads=1:ref=1:"
                                           "rc-lookahead=5:bframes=1:no-wpp=1:"
                                           "log-level=error"]),
    "libaom-av1": dict(codec="libaom-av1", npass=2, timeout=900,
                       extra=["-cpu-used", "8", "-row-mt", "1", "-tiles", "2x2",
                              "-threads", "2"]),
    # Single-threaded full-preset x264.  x264 allocates lookahead and reference
    # buffers per frame thread, so -threads 1 is the difference between fitting in
    # this box's ~1.6 GB at 4K and being SIGKILLed -- and unlike reducing ref or
    # rc-lookahead it costs no coding efficiency at all, only wall clock.  This is
    # the setting to prefer at 4K.
    "libx264-full-t1": dict(codec="libx264", npass=2, timeout=1500,
                            extra=["-preset", "slow", "-tune", "psnr", "-threads", "1"]),
    "libx264-full": dict(codec="libx264", npass=2, timeout=600,
                         extra=["-preset", "slow", "-tune", "psnr"]),
    "libx265-full": dict(codec="libx265", npass=2, timeout=900,
                         extra=["-preset", "medium"]),
    "libaom-av1-full": dict(codec="libaom-av1", npass=2, timeout=1800,
                            extra=["-cpu-used", "4", "-row-mt", "1",
                                   "-tiles", "2x2"]),
}
for _k, _v in REF_ENCODERS.items():
    _v["extra"] = _v["extra"] + ["-g", str(GOP), "-keyint_min", str(GOP)]


def encode_reference(name: str, src_yuv: str, w: int, h: int, nframes: int,
                     fps: float, target_kbps: float, workdir: str) -> dict:
    """Two-pass encode of `src_yuv` to a target bitrate.

    Returns dict(path=..., kbps=actual, secs=...).  The actual rate is measured
    from the file, never taken from the target: at low bitrates on short clips the
    two can differ by an order of magnitude, and quoting the target would be
    quoting an intention rather than a result.
    """
    spec = REF_ENCODERS[name]
    ff = ffmpeg_exe()
    out = os.path.join(workdir, f"{name}_{int(target_kbps)}.mkv")
    base = [ff, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", src_yuv]
    passlog = os.path.join(workdir, f"{name}_pass")
    npass = int(spec.get("npass", 2))
    tmo = int(spec.get("timeout", 1800))
    t0 = time.time()
    for pnum in range(1, npass + 1):
        cmd = base + ["-c:v", spec["codec"]] + spec["extra"] + [
            "-b:v", f"{int(target_kbps)}k", "-pix_fmt", "yuv420p"]
        if npass > 1:
            cmd += ["-pass", str(pnum), "-passlogfile", passlog]
        cmd += ["-f", "null", "/dev/null"] if pnum < npass else ["-an", out]
        # A hard timeout matters here: an encoder that is being OOM-thrashed rather
        # than killed can burn fifteen minutes before it dies, and one such probe
        # cost more wall clock than the entire benchmark.
        rc, msg = run(cmd, timeout=tmo)
        if rc != 0:
            raise RuntimeError(f"{name} pass {pnum} failed (rc={rc}): {msg[-500:]}")
    secs = time.time() - t0
    nbytes = os.path.getsize(out)
    dur = nframes / fps
    return dict(path=out, bytes=nbytes, kbps=8.0 * nbytes / dur / 1000.0, secs=secs)


def decode_to_yuv(video_path: str, out_yuv: str) -> None:
    ff = ffmpeg_exe()
    rc, msg = run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", video_path,
                   "-pix_fmt", "yuv420p", "-f", "rawvideo", out_yuv])
    if rc != 0:
        raise RuntimeError(f"decode failed: {msg[-600:]}")


# --------------------------------------------------------------------------- #
#  PHOTON-C, streamed
# --------------------------------------------------------------------------- #
def run_photon(src: YuvSource, cfg: Config, target_kbps: Optional[float] = None,
               want_ssim: bool = True) -> dict:
    """Encode then decode a whole clip, streaming frames through both.

    Metrics are accumulated per frame and frames are dropped immediately, so peak
    memory is one source frame plus one reconstruction plus the codec's working
    set -- not one of each per frame in the clip.
    """
    acc = MetricAcc(want_ssim)
    t0 = time.time()
    if target_kbps is not None:
        # probe on a spread of frames, then encode the whole clip at the chosen qp
        from photon.frame import encode_clip
        probe = [src[i] for i in
                 (0, src.nframes // 4, src.nframes // 2, src.nframes - 1)]
        blob, enc = encode_clip(probe, cfg, target_kbps=target_kbps)
        probe_bytes = len(blob)
        del probe, blob
        enc2 = Encoder(Config(**{**cfg.__dict__, "qp": enc.cfg.qp}))
        t0 = time.time()
        for i in range(src.nframes):
            enc2.push_frame(src[i])
        blob = enc2.finish()
        enc = enc2
    else:
        enc = Encoder(cfg)
        for i in range(src.nframes):
            enc.push_frame(src[i])
        blob = enc.finish()
    enc_s = time.time() - t0

    t1 = time.time()
    dec = Decoder(blob)
    for i in range(dec.nframes):
        acc.add(src[i], dec.decode_frame(i))
    dec_s = time.time() - t1

    r = acc.result()
    dur = src.nframes / cfg.fps
    nbytes = len(blob)
    return dict(codec="photon", bytes=nbytes, kbps=8.0 * nbytes / dur / 1000.0,
                bpp=8.0 * nbytes / (src.nframes * src.h * src.w * 3),
                psnr=r["psnr"], psnr_min=r["psnr_min"], ssim=r["ssim"],
                enc_s=enc_s, dec_s=dec_s,
                enc_fps=src.nframes / max(enc_s, 1e-9),
                dec_fps=src.nframes / max(dec_s, 1e-9),
                tx_depth=cfg.tx_depth, qp=cfg.qp, gop=cfg.gop,
                base_min_side=cfg.base_min_side,
                chroma_min_side=cfg.chroma_min_side,
                n_levels=len(luma_shapes(src.h, src.w, cfg.base_min_side)),
                base_shape=str(luma_shapes(src.h, src.w, cfg.base_min_side)[0]),
                blob=blob)


def measure_reference(name: str, src: YuvSource, info: dict,
                      workdir: str, want_ssim: bool = True) -> dict:
    dec_yuv = os.path.join(workdir, f"{name}_dec.yuv")
    decode_to_yuv(info["path"], dec_yuv)
    dec = YuvSource(dec_yuv, src.w, src.h, src.nframes)
    acc = MetricAcc(want_ssim)
    n = min(src.nframes, dec.nframes)
    for i in range(n):
        acc.add(src[i], dec[i])
    r = acc.result()
    os.remove(dec_yuv)
    return dict(codec=name, bytes=info["bytes"], kbps=info["kbps"],
                bpp=8.0 * info["bytes"] / (n * src.h * src.w * 3),
                psnr=r["psnr"], psnr_min=r["psnr_min"], ssim=r["ssim"],
                enc_s=info["secs"], dec_s=None,
                enc_fps=n / max(info["secs"], 1e-9), dec_fps=None,
                nframes=n)
