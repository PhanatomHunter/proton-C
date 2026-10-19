#!/usr/bin/env python3
"""Decode atto_4k50.phtn (208 kbps 4K x 50 s) -> per-chunk mp4 segments, concat.

RAM-tight box (~1.9 GB): x264 lookahead is throttled hard so the mp4 writer
cannot be OOM-killed.  Chunks of 250 frames then a stream-copy concat.
"""
import os, resource, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageio_ffmpeg
from photon.frame import Decoder

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "out")
FF = imageio_ffmpeg.get_ffmpeg_exe()
SEG = 250
t0_wall = time.time()

blob = open(os.path.join(HERE, "results", "atto_4k50.phtn"), "rb").read()
d = Decoder(blob)
n = d.nframes
stills = {}
segments = []

t_dec0 = time.time()
cpu0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
for start in range(0, n, SEG):
    seg = os.path.join(OUTDIR, f"_seg_{start:05d}.mp4")
    segments.append(seg)
    if not os.path.exists(seg):
        cmd = [FF, "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "3840x2160",
               "-r", "30000/1001", "-i", "-",
               "-frames:v", str(min(SEG, n - start)),
               "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-x264-params", "rc-lookahead=8:ref=2",
               "-pix_fmt", "yuv420p", seg]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for k in range(min(SEG, n - start)):
            fr = d.decode_frame(start + k)
            try:
                p.stdin.write(fr.tobytes())
            except BrokenPipeError:
                raise RuntimeError(f"ffmpeg dead: {p.stderr.read().decode()[-1500:]}")
            if (start + k) in (750, 1000, 1400):
                stills[start + k] = fr.copy()
        p.stdin.close()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"seg rc={rc}: {p.stderr.read().decode()[-1500:]}")
    else:
        d.decode_frame(start)   # advance decoder state (stateful)
        for k in range(1, min(SEG, n - start)):
            fr = d.decode_frame(start + k)
            if (start + k) in (750, 1000, 1400):
                stills[start + k] = fr.copy()
    print(f"seg {start}: done", flush=True)

cpu1 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
wall_dec = time.time() - t_dec0
print(f"DECODE+WRITE wall {wall_dec:.1f}s total; python cpu {cpu1-cpu0:.1f}s", flush=True)

for k, fr in stills.items():
    np.save(os.path.join(OUTDIR, f"dec_still_{k}.npy"), fr)

with open(os.path.join(OUTDIR, "_concat.txt"), "w") as fh:
    for s in segments:
        fh.write(f"file '{s}'\n")
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", os.path.join(OUTDIR, "_concat.txt"),
                "-c", "copy", "-movflags", "+faststart",
                os.path.join(OUTDIR, "photon_4k50_200kbps_demo.mp4")], check=True)
sz = os.path.getsize(os.path.join(OUTDIR, "photon_4k50_200kbps_demo.mp4"))
print(f"MP4 ready: {sz/1e6:.1f} MB", flush=True)
with open(os.path.join(OUTDIR, "decode_stats.json"), "w") as fh:
    import json
    json.dump({"frames": n, "wall_total_s": time.time() - t0_wall,
               "decode_plus_mp4_wall_s": wall_dec, "blob_bytes": len(blob),
               "kbps": 8 * len(blob) / (n / 29.97) / 1000}, fh)
