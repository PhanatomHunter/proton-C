#!/usr/bin/env python3
"""HD/'normal good internet' tier demo: BBB 1080p60 -> photon-lean stream ->
decode + md5 verify + playable mp4 + quality stats.

Runs under PHOTON_F32=1 (fast rebuild mode); the md5 list generated here is
self-consistent (encode and decode under the same flag), so the verify step
checks decode-vs-encode fidelity, not cross-environment bytes.
"""
import hashlib, json, os, resource, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imageio_ffmpeg
from photon.cli import Encoder
from photon.codec import preset
from photon.frame import Decoder

SRC = "/home/user/testvideo/bbb_1080p.mp4"
W, H, FPS = 1920, 1080, 60
N = 300                        # 5 s
PRESET = os.environ.get("HDTIER_PRESET", "photon-lean")
OUTDIR = os.path.join(HERE, "out")
TARGET_FRAMES = int(os.environ.get("HDTIER_FRAMES", str(N)))
TAG = os.environ.get("HDTIER_TAG", PRESET)
RESULTS = os.path.join(HERE, "results")
FF = imageio_ffmpeg.get_ffmpeg_exe()

os.makedirs(OUTDIR, exist_ok=True)

def frames(n):
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-i", SRC,
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         bufsize=W * H * 3 * 4)
    got = 0
    while got < n:
        chunk = p.stdout.read(W * H * 3)
        if not chunk:
            break
        got += 1
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(H, W, 3)
    p.stdout.close(); p.wait()

# ---------------- encode ----------------
cfg = preset(PRESET, W, H, fps=FPS)
enc = Encoder(cfg)
n_enc = 0
t0 = time.perf_counter()
for fr in frames(N):
    enc.push_frame(fr)
    n_enc += 1
blob = enc.finish()
t_enc = time.perf_counter() - t0
kbps = 8 * len(blob) / (n_enc / FPS) / 1000.0
print(f"encoded {n_enc} f in {t_enc:.0f}s ({n_enc/t_enc:.2f} f/s), "
      f"{len(blob)/1e6:.2f} MB -> {kbps:.0f} kbps", flush=True)
open(os.path.join(RESULTS, "hdtier_bbb_" + TAG + ".phtn"), "wb").write(blob)

# ---------------- decode + md5 + PSNR ----------------
d = Decoder(blob)
md5s = []
seg = os.path.join(OUTDIR, "hdtier_bbb_" + TAG + ".mp4")
cmd = [FF, "-hide_banner", "-loglevel", "error", "-y",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(FPS), "-i", "-",
       "-c:v", "libx264", "-preset", "superfast", "-crf", "16",
       "-pix_fmt", "yuv420p", seg]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
import cv2
psnrs = []
t0 = time.perf_counter()
gen = frames(N)
for i in range(n_enc):
    src = next(gen)
    rec = d.decode_frame(i)
    md5s.append(hashlib.md5(np.ascontiguousarray(rec).tobytes()).hexdigest())
    psnrs.append(cv2.PSNR(src, rec))
    p.stdin.write(rec.tobytes())
p.stdin.close()
rc = p.wait()
wall = time.perf_counter() - t0
if rc != 0:
    raise RuntimeError(p.stderr.read().decode()[-1000:])
print(f"decode+pipe: {wall:.0f}s ({n_enc/wall:.2f} f/s), "
      f"mean PSNR {float(np.mean(psnrs)):.2f} dB, "
      f"mp4 {os.path.getsize(seg)/1e6:.1f} MB", flush=True)
json.dump({"preset": PRESET, "frames": n_enc, "kbps": kbps,
           "psnr_mean": float(np.mean(psnrs)), "md5": md5s},
          open(os.path.join(RESULTS, "hdtier_bbb_" + TAG + "_metrics.json"), "w"))

# ---------------- verify second decode matches md5 ----------------
d2 = Decoder(blob)
bad = 0
t0 = time.perf_counter()
for i in range(n_enc):
    h = hashlib.md5(np.ascontiguousarray(d2.decode_frame(i)).tobytes()).hexdigest()
    if h != md5s[i]:
        bad += 1
wall = time.perf_counter() - t0
dec_fps = n_enc / wall
print(f"verify pass2: {wall:.0f}s ({dec_fps:.2f} f/s decode-only), md5 mismatches {bad}/{n_enc}", flush=True)
json.dump({"decode_fps_no_pipe": dec_fps,
           "decode_ms_mean": wall / n_enc * 1000.0},
          open(os.path.join(RESULTS, "hdtier_bbb_" + TAG + "_speed.json"), "w"))
