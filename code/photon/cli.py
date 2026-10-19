"""
PHOTON-C command line.

    python3 -m photon presets
    python3 -m photon encode  in.mp4  -o out.phtn  --preset photon-atto
    python3 -m photon encode  in.mp4  -o out.phtn  --kbps 250
    python3 -m photon decode  out.phtn -o recon.mp4
    python3 -m photon info    out.phtn
    python3 -m photon bench   in.mp4  --res 640x360 --frames 24

Everything streams.  Frames are piped to and from ffmpeg rather than accumulated
in a list, because a 4K frame is 25 MB and this sandbox has ~1.6 GB: a 60-frame
4K clip does not fit in memory as decoded RGB, and neither does its
reconstruction.  Streaming also means encode and decode work on inputs longer
than RAM, which is the only behaviour that makes sense for a streaming codec.

The decoder writes its output through ffmpeg as H.264 so it can actually be
watched.  That re-encode is at crf 12, which is visually transparent but not
bit-exact -- `decode --raw` writes the I420 planes straight out with no second
codec in the path, and `--png` dumps frames as images.  Quality numbers reported
by `bench` are measured against the source directly and never involve this step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photon import __version__                                    # noqa: E402
from photon.codec import Config, MAGIC, PRESETS, preset          # noqa: E402
from photon.frame import (Decoder, Encoder, encode_clip,          # noqa: E402
                          luma_shapes, chroma_shapes)


# --------------------------------------------------------------------------- #
#  ffmpeg plumbing
# --------------------------------------------------------------------------- #
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    for c in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("ffmpeg not found.  pip install imageio-ffmpeg, or put ffmpeg on PATH.")


def probe(path: str) -> Tuple[int, int, float, int]:
    """(width, height, fps, nframes) for a video file, via ffprobe or ffmpeg."""
    ff = ffmpeg_exe()
    p = subprocess.run([ff, "-hide_banner", "-i", path], capture_output=True, text=True)
    err = p.stderr
    w = h = 0
    import re
    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    fps = 30.0
    m = re.search(r"(\d+(?:\.\d+)?) fps", err)
    if m:
        fps = float(m.group(1))
    n = 0
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        n = int(round(dur * fps))
    return w, h, fps, n


def frame_reader(path: str, w: int, h: int, fps: float,
                 limit: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames by piping rawvideo out of ffmpeg."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps)]
    if limit:
        # -frames:v is an output option, so it belongs after the filtergraph and
        # before the output URL, not next to the input.
        cmd += ["-frames:v", str(limit)]
    cmd += ["-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsz = w * h * 3
    n = 0
    try:
        while True:
            buf = p.stdout.read(fsz)
            if len(buf) < fsz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            n += 1
            if limit and n >= limit:
                break
    finally:
        p.stdout.close()
        p.wait()


def frame_writer(path: str, w: int, h: int, fps: float, crf: int = 12
                 ) -> Tuple[subprocess.Popen, int]:
    """Open an ffmpeg pipe that turns rawvideo RGB on stdin into an H.264 file."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p, w * h * 3


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #

def encode_twopass(frames_fn, cfg: Config, target_kbps: float,
                   max_passes: int = 4, tol: float = 0.08,
                   verbose: bool = True) -> Tuple[bytes, Encoder, dict]:
    """Full-clip, full-encode rate control.  `frames_fn()` must return a FRESH
    iterator of RGB frames on every call (the function re-reads the source on
    each pass), so nothing from the source has to live in memory.

    Why not probe-based qp search: a 12-frame probe often lands on the wrong
    side of the clip's content (measured: a 200 kbps target produced 476 kbps,
    because the probing frames were easier to code than the clip).  Encoding the
    whole clip and measuring what it actually produced is slower but honest --
    which, for a user-facing knob, is worth more than iteration-count elegance.

    Sign convention, important and easy to get backwards (it burned a pass in
    development): qp here follows the JPEG QUALITY-FACTOR convention, not x264's.
    Higher qp means a *finer* quantiser and therefore *more* bits.  At the
    calibration extremes: qp=100 is visually indistinguishable from lossless,
    qp~30 is heavily quantised.  Doubling the bitrate corresponds to roughly 8
    qp in the mid range; the step is damped against oscillation and clamped.
    """
    hist = []
    qp = cfg.qp
    best = None
    for attempt in range(1, max_passes + 1):
        cfg2 = Config(**{**cfg.__dict__, "qp": int(qp), "target_kbps": None})
        enc = Encoder(cfg2)
        for f in frames_fn():
            enc.push_frame(f)
        kbps = enc.bitrate_kbps
        err = kbps / target_kbps
        hist.append(dict(qp=int(qp), kbps=kbps, err=err))
        if verbose:
            print(f"  pass {attempt}: qp={int(qp)} -> {kbps:.1f} kbps (target {target_kbps:.1f})")
        if best is None or abs(err - 1) < abs(best[2] - 1):
            best = (enc, enc.finish(), err)
        if abs(err - 1) <= tol:
            best = (enc, enc.finish(), err)
            break
        # err > 1  (output too fat)  -> coarser quantiser -> LOWER qp in this
        # codec's convention.  See the sign note above.
        step = 8.0 * np.log2(err)
        step = max(-14.0, min(14.0, step))
        qp = int(min(max(qp - round(step), 30), 96))
    return best[1], best[0], dict(target=target_kbps, hist=hist, final_err=best[2])


# Resolution-bucketed bases for rate control: cheapest first.  At very low
# targets the quantiser alone cannot reach the goal without destroying the
# base -- the sweep data (results/photon_4k.json etc.) shows the attainable
# floor for a given (tx_depth, base) ladder rung, so RC walks the ladder until
# the floor is no longer above the target.
_RC_LADDER = {
    "4k": ["photon-atto", "photon-femto", "photon-nano", "photon-lean", "photon-4k",
           "photon-good-4k"],
    "hd": ["photon-hd-atto", "photon-hd-femto", "photon-hd-nano", "photon-hd-lean",
           "photon-hd", "photon-good"],
    "sd": ["photon-sd-atto", "photon-sd", "photon-sd-lean", "photon-hd-femto",
           "photon-good-sd"],
}


def rc_family(width: int, height: int) -> List[str]:
    px = width * height
    if px >= 3_000_000:
        return _RC_LADDER["4k"]
    if px >= 800_000:
        return _RC_LADDER["hd"]
    return _RC_LADDER["sd"]


def encode_rate_controlled(frames_fn, width: int, height: int, fps: float,
                           target_kbps: float, start_preset: Optional[str] = None,
                           verbose: bool = True) -> Tuple[bytes, Encoder, dict]:
    """Walk the preset ladder to a target bitrate.

    Runs encode_twopass on each ladder rung, coarsest first (or starting from
    `start_preset` and proceeding coarser-then-finer).  Stops as soon as a rung
    lands within 15% overshoot -- over-engineering finer RP granularity buys
    nothing a viewer can see.
    """
    fam = rc_family(width, height)
    if start_preset is not None:
        # the user picked a preset explicitly: honour it as the anchor.  If it
        # is not a ladder member, run two-pass on it alone; otherwise start the
        # walk AT the chosen rung (up first, then down) instead of the cheapest
        # one -- with a ceiling, choosing a worse rung than requested is wrong.
        if start_preset not in PRESETS:
            raise SystemExit(f"unknown preset {start_preset!r}")
        order = [start_preset] + [p for p in fam if p != start_preset] \
                if start_preset not in fam else \
                fam[fam.index(start_preset):] + list(reversed(fam[:fam.index(start_preset)]))
    else:
        # start from the cheapest rung; it is far cheaper to walk up than down
        # because overshoot means the whole encode has to run again.
        order = fam
    best = None
    for name in order:
        cfg = preset(name, width, height, fps)
        blob, enc, rc = encode_twopass(frames_fn, cfg, target_kbps,
                                       max_passes=3, tol=0.08, verbose=verbose)
        err = enc.bitrate_kbps / target_kbps
        if verbose:
            print(f"  preset {name}: settled qp={enc.cfg.qp} -> {enc.bitrate_kbps:.1f} kbps "
                  f"(err {err:.2f}x)")
        if best is None or abs(err - 1) < abs(best[3] - 1):
            best = (blob, enc, name, err)
        # ceiling semantics: a rung within +/-8% of target, or any rung that
        # comes in under target after its full pass, settles.  Walking further
        # up the ladder after that only spends quality on unreachable rates.
        if 0.92 <= err <= 1.08 or err < 1.0:
            best = (blob, enc, name, err)
            break
    return best[0], best[1], dict(preset=best[2], target=target_kbps, err=best[3])

def cmd_presets(args) -> int:
    print("PHOTON-C presets\n")
    print(f"{'name':18s} {'tx':>3s} {'qp':>4s} {'gop':>4s} {'luma base':>10s} "
          f"{'chroma base':>10s}   {'levels':>6s}   intended use")
    uses = {
        "photon-4k":      "4K, every band transmitted -- quality mode",
        "photon-lean":    "4K, three bands -- good quality, low rate",
        "photon-nano":    "4K, two bands",
        "photon-femto":   "4K, one band",
        "photon-atto":    "4K, base layer only -- the extreme mode",
        "photon-hd":      "1080p, quality mode",
        "photon-hd-lean": "1080p, two bands",
        "photon-hd-nano": "1080p, one band",
        "photon-hd-atto": "1080p, base layer only",
        "photon-sd":      "480p and below",
        "photon-sd-atto": "480p, base layer only -- audio-call rates",
        "photon-good":    "GOOD-INTERNET 1080p: full detail, ~4-10 Mbit/s",
        "photon-good-4k": "GOOD-INTERNET 4K: full detail, ~10-25 Mbit/s",
        "photon-good-sd": "GOOD-INTERNET <=480p: full detail, ~0.8-1.5 Mbit/s",
    }
    for name, (txd, qp, gop, bms, cms) in PRESETS.items():
        res = "4k" if ("4k" in name) else ("hd" if (name == "photon-good" or
               name.split("-")[1] == "hd") else "sd")
        w, h = {"4k": (3840, 2160), "hd": (1920, 1080), "sd": (854, 480)}[res]
        sh = luma_shapes(h, w, bms)
        cs = chroma_shapes(sh[-1], cms)
        print(f"{name:18s} {txd:3d} {qp:4d} {gop:4d} {str(sh[0]):>10s} "
              f"{str(cs[0]):>10s}   {len(sh):6d}   {uses.get(name,'')}")
    print("""
tx_depth is how many detail bands above the base are transmitted; the rest are
synthesised on both sides from the previous frame's bands plus a derived motion
field, and cost nothing.  luma/chroma base are absolute pixel sizes, not ratios:
that is the point -- the base does not grow with the frame, which is why the
bitrate holds up as resolution goes up.""")
    return 0


def cmd_encode(args) -> int:
    sw, sh_, sfps, sn = probe(args.input)
    w = args.width or sw
    h = args.height or sh_
    fps = args.fps or sfps or 30.0
    n = args.frames or sn or 0
    print(f"input  {args.input}: {sw}x{sh_} @ {sfps:.3g} fps, ~{sn} frames")
    print(f"coding {w}x{h} @ {fps:g} fps" + (f", first {n} frames" if args.frames else ""))

    if args.preset:
        cfg = preset(args.preset, w, h, fps, target_kbps=args.kbps)
    else:
        cfg = Config(width=w, height=h, fps=fps, gop=args.gop or 60, qp=args.qp or 72,
                     tx_depth=args.tx_depth if args.tx_depth is not None else 1,
                     base_min_side=args.base_min_side or 96,
                     chroma_min_side=args.chroma_min_side or 96,
                     target_kbps=args.kbps)
    lsh = luma_shapes(h, w, cfg.base_min_side)
    csh = chroma_shapes(lsh[-1], cfg.chroma_min_side)
    print(f"config qp={cfg.qp} tx_depth={cfg.tx_depth} gop={cfg.gop} "
          f"luma_base={lsh[0]} ({len(lsh)} levels) chroma_base={csh[0]} ({len(csh)} levels)")

    print(f"rate control: two-pass full-clip search" if args.kbps else "")
    t0 = time.time()
    if args.kbps:
        make_iter = lambda: frame_reader(args.input, w, h, fps, n)
        blob, enc, rc = encode_rate_controlled(make_iter, w, h, fps, args.kbps,
                                               start_preset=args.preset)
        cfg = Config(**{**enc.cfg.__dict__})
    else:
        enc = Encoder(cfg)
        for f in frame_reader(args.input, w, h, fps, n):
            enc.push_frame(f)
        blob = enc.finish()
    secs = time.time() - t0

    nf = len(enc.frames)
    if nf == 0:
        raise SystemExit(
            "no frames were read from the input -- the ffmpeg pipe produced "
            "nothing.  Check that the file plays and the resolution/fps "
            "detection above looks right.")
    with open(args.output, "wb") as fh:
        fh.write(blob)
    kbps = enc.bitrate_kbps
    raw = float(w) * h * 3 * 8 * fps / 1000.0
    print(f"\nwrote {args.output}  {len(blob)/1024:.1f} KiB  ({len(blob)/max(nf,1):.0f} B/frame)")
    print(f"  {kbps:.1f} kbps   {8*len(blob)/(max(nf,1)*w*h*3):.5f} bpp   "
          f"{raw/max(kbps,1e-9):,.0f}x smaller than raw RGB")
    print(f"  encoded {nf} frames in {secs:.1f}s = {nf/max(secs,1e-9):.2f} fps "
          f"({'real time' if nf/max(secs,1e-9) >= fps else f'{fps/(nf/max(secs,1e-9)):.1f}x slower than real time'})")
    kf = [s["bytes"] for s in enc.stats if s["keybase"]]
    pf = [s["bytes"] for s in enc.stats if not s["keybase"]]
    if kf and pf:
        print(f"  keyframes {len(kf)} (mean {np.mean(kf):.0f} B), "
              f"delta frames {len(pf)} (mean {np.mean(pf):.0f} B)")
    return 0


def cmd_decode(args) -> int:
    with open(args.input, "rb") as fh:
        blob = fh.read()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{args.input} is not a PHOTON-C stream (magic {blob[:4]!r})")
    from .guard import StreamError
    try:
        dec = Decoder(blob)
    except StreamError as e:
        raise SystemExit(f"fatal: stream is corrupt/invalid: {e}") from None
    print(f"stream {args.input}: {dec.width}x{dec.height} @ {dec.fps:g} fps, "
          f"{dec.nframes} frames, {dec.n_levels} pyramid levels")

    if args.raw:
        t0 = time.time()
        with open(args.output, "wb") as fh:
            import cv2
            for i in range(dec.nframes):
                try:
                    rgb = dec.decode_frame(i)
                except StreamError as e:
                    raise SystemExit(
                        f"fatal: frame {i} failed to decode: {e}") from None
                yuv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, ::-1]),
                                   cv2.COLOR_BGR2YUV_I420)
                fh.write(yuv.tobytes())
        print(f"wrote raw I420 {args.output} in {time.time()-t0:.1f}s "
              f"({dec.nframes/(time.time()-t0):.2f} fps)")
        return 0

    if args.png:
        import cv2
        os.makedirs(args.output, exist_ok=True)
        t0 = time.time()
        for i in range(dec.nframes):
            cv2.imwrite(os.path.join(args.output, f"frame_{i:05d}.png"),
                        dec.decode_frame(i)[:, :, ::-1])
        print(f"wrote {dec.nframes} PNGs to {args.output} in {time.time()-t0:.1f}s")
        return 0

    p, fsz = frame_writer(args.output, dec.width, dec.height, dec.fps, args.crf)
    t0 = time.time()
    try:
        for i in range(dec.nframes):
            try:
                frame = dec.decode_frame(i)
            except StreamError as e:
                raise SystemExit(
                    f"fatal: frame {i} failed to decode: {e}") from None
            p.stdin.write(frame.tobytes())
    finally:
        p.stdin.close()
        p.wait()
    secs = time.time() - t0
    print(f"wrote {args.output} ({os.path.getsize(args.output)/1024:.1f} KiB) in {secs:.1f}s "
          f"= {dec.nframes/max(secs,1e-9):.2f} fps decode+remux")
    print("  note: the H.264 remux at crf "
          f"{args.crf} is for viewing convenience, not part of the codec's own "
          "loss; use --raw for the decoder's exact output.")
    return 0


def cmd_info(args) -> int:
    with open(args.input, "rb") as fh:
        blob = fh.read()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{args.input} is not a PHOTON-C stream")
    dec = Decoder(blob)
    lsh = luma_shapes(dec.height, dec.width, dec.meta["base_min_side"])
    csh = chroma_shapes(lsh[-1], dec.meta["chroma_min_side"])
    dur = dec.nframes / dec.fps
    raw_kbps = dec.width * dec.height * 3 * 8 * dec.fps / 1000.0
    print(f"PHOTON-C stream {args.input}")
    print(f"  version        {dec.meta.get('version', '?')}")
    print(f"  geometry       {dec.width}x{dec.height} @ {dec.fps:g} fps, {dec.nframes} frames ({dur:.2f}s)")
    print(f"  size           {len(blob)} bytes ({len(blob)/1024:.1f} KiB)")
    print(f"  bitrate        {8*len(blob)/dur/1000:.1f} kbps   "
          f"{8*len(blob)/(dec.nframes*dec.width*dec.height*3):.5f} bpp")
    print(f"  vs raw RGB     {raw_kbps/(8*len(blob)/dur/1000):,.0f}x smaller")
    print(f"  qp / gop       {dec.meta['qp']} / {dec.meta['gop']}")
    print(f"  tx_depth       {dec.meta['tx_depth']}  "
          f"(bands 0..{dec.meta['tx_depth']-1} transmitted, "
          f"{dec.n_levels-1-dec.meta['tx_depth']} synthesised)")
    print(f"  luma pyramid   {len(lsh)} levels, base {lsh[0]}")
    for i, s in enumerate(lsh):
        tag = "base" if i == 0 else f"band {i-1}"
        sent = "TRANSMITTED" if (i == 0 or i - 1 < dec.meta["tx_depth"]) else "synthesised"
        print(f"      level {i}  {str(s):12s} {s[0]*s[1]:9,d} px   {tag:8s} {sent}")
    print(f"  chroma pyramid {len(csh)} levels, base {csh[0]}  (4:2:0-equivalent at full tx_depth)")
    sizes = [dec.frame_bytes(i) for i in range(dec.nframes)]
    print(f"  frame sizes    mean {np.mean(sizes):.0f} B, min {min(sizes)}, max {max(sizes)}")
    print(f"                 first 12: {sizes[:12]}")
    return 0


def cmd_bench(args) -> int:
    import benchlib as B
    w, h = [int(x) for x in args.res.split("x")]
    fps = args.fps
    nf = args.frames
    src_yuv = args.source or f"/home/user/photon/out/work/bench_{w}x{h}_{nf}.yuv"
    os.makedirs(os.path.dirname(src_yuv), exist_ok=True)
    if not (os.path.exists(src_yuv) and os.path.getsize(src_yuv) == nf * w * h * 3 // 2):
        print(f"preparing {nf} frames of {w}x{h} from {args.video} -> {src_yuv}")
        B.make_source_yuv(src_yuv, nf, w, h, fps, kind="video", src_video=args.video)
    src = B.YuvSource(src_yuv, w, h, nf, fps)

    out = []
    if not args.refs_only:
        for name in (args.presets or ["photon-hd-atto", "photon-hd-nano", "photon-hd-lean"]):
            cfg = preset(name, w, h, fps)
            if args.kbps:
                r = B.run_photon(src, cfg, target_kbps=args.kbps)
            else:
                r = B.run_photon(src, cfg)
            r.pop("blob", None)
            r["label"] = name
            out.append(r)
            print(f"  {name:18s} {r['kbps']:8.1f} kbps  PSNR {r['psnr']:5.2f}  "
                  f"SSIM {r['ssim']:.4f}  bpp {r['bpp']:.5f}  enc {r['enc_fps']:.2f} f/s", flush=True)
    for ref in (args.ref_codecs or []):
        import tempfile, shutil
        for tk in (args.ref_targets or [200.0, 400.0, 800.0]):
            tmp = tempfile.mkdtemp(prefix="bench_")
            try:
                info = B.encode_reference(ref, src_yuv, w, h, nf, fps, tk, tmp)
                r = B.measure_reference(ref, src, info, tmp)
                r["label"] = f"{ref}@{tk:.0f}k"
                out.append(r)
                print(f"  {r['label']:18s} {r['kbps']:8.1f} kbps  PSNR {r['psnr']:5.2f}  "
                      f"SSIM {r['ssim']:.4f}  bpp {r['bpp']:.5f}  enc {r['enc_fps']:.2f} f/s", flush=True)
            except Exception as e:
                print(f"  {ref}@{tk:.0f}k FAILED: {str(e)[:120]}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="photon",
        description=f"PHOTON-C v{__version__} -- video compression for extreme low bandwidth")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list presets and what they do").set_defaults(fn=cmd_presets)

    e = sub.add_parser("encode", help="encode a video to a .phtn stream")
    e.add_argument("input"); e.add_argument("-o", "--output", required=True)
    e.add_argument("--preset", help="one of the names from `photon presets`")
    e.add_argument("--kbps", type=float, help="target bitrate; searches qp to hit it")
    e.add_argument("--qp", type=int); e.add_argument("--gop", type=int)
    e.add_argument("--tx-depth", type=int, dest="tx_depth")
    e.add_argument("--base-min-side", type=int, dest="base_min_side")
    e.add_argument("--chroma-min-side", type=int, dest="chroma_min_side")
    e.add_argument("--width", type=int); e.add_argument("--height", type=int)
    e.add_argument("--fps", type=float); e.add_argument("--frames", type=int)
    e.set_defaults(fn=cmd_encode)

    d = sub.add_parser("decode", help="decode a .phtn stream")
    d.add_argument("input"); d.add_argument("-o", "--output", required=True)
    d.add_argument("--raw", action="store_true", help="write raw I420, no remux")
    d.add_argument("--png", action="store_true", help="output is a directory of PNGs")
    d.add_argument("--crf", type=int, default=12)
    d.set_defaults(fn=cmd_decode)

    i = sub.add_parser("info", help="dump a stream's header and pyramid layout")
    i.add_argument("input"); i.set_defaults(fn=cmd_info)

    b = sub.add_parser("bench", help="measure PHOTON-C and reference codecs on a clip")
    b.add_argument("--video", default="/home/user/testvideo/bbb_720p.mp4")
    b.add_argument("--source", help="reuse/overwrite this raw I420 file")
    b.add_argument("--res", default="640x360"); b.add_argument("--fps", type=float, default=30.0)
    b.add_argument("--frames", type=int, default=48)
    b.add_argument("--presets", nargs="*")
    b.add_argument("--kbps", type=float)
    b.add_argument("--ref-codecs", nargs="*", dest="ref_codecs")
    b.add_argument("--ref-targets", nargs="*", type=float, dest="ref_targets")
    b.add_argument("--refs-only", action="store_true")
    b.add_argument("--json")
    b.set_defaults(fn=cmd_bench)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
