#!/usr/bin/env python3
"""Build the best achievable ~200-300 kbps 'clear' demo on the Nikon 4K nature
clip using the winner config found by rd200_probe.py.

Usage:
  PHOTON_F32=1 CLEAR200_CFG=360p10-base136,qp68 N=900 python3 demo_build_clear200.py

Encodes N frames, decodes+PSNRs them, md5-verifies a fresh decoder against the
encode-run decoder (self-consistent PHOTON_F32 env), and writes an x264-wrapped
playable mp4.  All streaming: nothing retains full frames.
"""
import os, sys, json, time, subprocess, hashlib
import numpy as np
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PHOTON_F32", "1")

from photon.codec import preset, Config
from photon.cli import Encoder
from photon.frame import Decoder

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/user/testvideo/nikon4k_50s.webm"
FF = imageio_ffmpeg.get_ffmpeg_exe()
OUTDIR = os.path.join(HERE, "out")
RESULTS = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

N = int(os.environ.get("N", "300"))
RG = os.environ.get("PRE_S", "0")


def parse_spec(spec):
    """'360p10-base136,qp68'  ->  w,h,fps,tx,qp,gop,bms (gop override via g=)"""
    import re
    m = re.match(r"(\d+)p(\d+)-base(\d+),qp(\d+)(?:,g(\d+))?(?:,tx(\d+))?$", spec)
    h_px = int(m.group(1)); fps = int(m.group(2)); bms = int(m.group(3))
    qp = int(m.group(4)); gop = int(m.group(5) or 150); tx = int(m.group(6) or 0)
    w = {360: 640, 420: 746, 480: 854, 540: 960, 720: 1280, 1080: 1920, 2160: 3840}[h_px]
    return w, h_px, fps, tx, qp, gop, bms


def frames_winner(w, h, fps, n):
    p = subprocess.Popen(
        [FF, "-v", "error", "-ss", RG, "-i", SRC,
         "-vf", f"fps={fps},scale={w}:{h}:flags=lanczos",
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    fsz = w * h * 3
    for _ in range(n):
        chunk = p.stdout.read(fsz)
        if len(chunk) < fsz:
            break
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3)
    p.stdout.close(); p.wait()


def main():
    spec = os.environ.get("CLEAR200_CFG", "360p10-base136,qp68")
    w, h, fps, tx, qp, gop, bms = parse_spec(spec)
    tag = spec.replace(",", "_").replace(" ", "")
    cfg = Config(width=w, height=h, fps=fps, gop=gop, qp=qp, tx_depth=tx,
                 base_min_side=bms, chroma_min_side=max(48, bms // 2))

    # ---- encode ----
    enc = Encoder(cfg)
    nf = 0
    t0 = time.perf_counter()
    for fr in frames_winner(w, h, fps, N):
        enc.push_frame(fr)
        nf += 1
    blob = enc.finish()
    t_enc = time.perf_counter() - t0
    kbps = 8 * len(blob) / (max(nf, 1) / fps) / 1000.0
    stream_path = os.path.join(RESULTS, f"clear200_{tag}.phtn")
    open(stream_path, "wb").write(blob)
    print(f"encoded {nf} f in {t_enc:.0f}s ({nf/t_enc:.2f} f/s), "
          f"{len(blob)/1e6:.3f} MB -> {kbps:.0f} kbps", flush=True)

    # ---- decode+PSNR (source re-streamed; nothing retained) ----
    mp4_tmp = os.path.join(OUTDIR, f"clear200_{tag}.mp4")
    wr = subprocess.Popen(
        [FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-vf", "format=yuv420p",
         "-c:v", "libx264", "-preset", "superfast", "-crf", "16",
         "-movflags", "+faststart", mp4_tmp], stdin=subprocess.PIPE)
    dec = Decoder(blob)
    md5s = []
    psnrs = []
    gen = frames_winner(w, h, fps, nf)
    t1 = time.perf_counter()
    for i in range(nf):
        fr = dec.decode_frame(i)
        md5s.append(hashlib.md5(np.ascontiguousarray(fr).tobytes()).hexdigest())
        try:
            ref = next(gen)
        except StopIteration:
            break
        mse = float(np.mean((fr[:ref.shape[0], :ref.shape[1]].astype(np.float64)
                             - ref) ** 2))
        psnrs.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        try:
            wr.stdin.write(fr.tobytes())
        except BrokenPipeError:
            break
    try:
        wr.stdin.close(); wr.wait()
    except BrokenPipeError:
        pass
    t_dec = time.perf_counter() - t1
    print(f"decode+pipe: {t_dec:.0f}s ({nf/max(t_dec,1e-9):.2f} f/s), "
          f"mean PSNR {np.mean(psnrs):.2f} dB, mp4 {os.path.getsize(mp4_tmp)/1e6:.2f} MB",
          flush=True)

    # ---- verify pass: fresh decoder == encode-pass decoder (same env) ----
    dec2 = Decoder(blob)
    bad = 0
    t2 = time.perf_counter()
    for i in range(nf):
        r = dec2.decode_frame(i)
        if hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest() != md5s[i]:
            bad += 1
    print(f"verify pass: {nf/(time.perf_counter()-t2):.2f} f/s decode-only, "
          f"md5 mismatches {bad}/{nf}", flush=True)

    meta = dict(spec=spec, w=w, h=h, fps=fps, n=nf, kbps=round(kbps, 1),
                tx=tx, qp=qp, gop=gop, base_min_side=bms,
                psnr_mean=round(float(np.mean(psnrs)), 2),
                psnr_min=round(float(np.min(psnrs)), 2),
                md5_bad=bad, stream=stream_path, mp4=mp4_tmp,
                dec_fps_no_pipe=round(nf / max(t_dec - (os.path.getsize(mp4_tmp) * 0), 1e-9), 2))
    json.dump(meta, open(os.path.join(RESULTS, f"clear200_{tag}.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
