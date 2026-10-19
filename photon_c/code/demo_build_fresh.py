#!/usr/bin/env python3
"""Decode out/fresh_4k50_atto.phtn -> seg mp4s -> concat -> demo + stills + md5 verify."""
import hashlib, json, os, resource, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imageio_ffmpeg
from photon.frame import Decoder

OUTDIR = os.path.join(HERE, "out")
FF = imageio_ffmpeg.get_ffmpeg_exe()
SEG = 250
t0_wall = time.time()

blob = open(os.path.join(HERE, "results", "demo_photon_4k50_207kbps.phtn"), "rb").read()
metrics = json.load(open(os.path.join(HERE, "results", "fresh_4k50_atto_metrics.json")))
d = Decoder(blob)
n = d.nframes
stills = {}
segments = []

t_dec0 = time.time()
cpu0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
md5_bad = 0
for i in range(n):
    fr = None
    seg_i = i // SEG
    if i % SEG == 0:
        seg = os.path.join(OUTDIR, f"_segN_{seg_i:05d}.mp4")
        segments.append(seg)
        cmd = [FF, "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "3840x2160",
               "-r", "30000/1001", "-i", "-",
               "-frames:v", str(min(SEG, n - i)),
               "-c:v", "libx264", "-preset", "superfast", "-crf", "16",
               "-pix_fmt", "yuv420p", seg]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    fr = d.decode_frame(i)
    h = hashlib.md5(np.ascontiguousarray(fr).tobytes()).hexdigest()
    if h != metrics["md5"][i]:
        md5_bad += 1
    try:
        p.stdin.write(fr.tobytes())
    except BrokenPipeError:
        raise RuntimeError(f"ffmpeg dead frame {i}: {p.stderr.read().decode()[-1200:]}")
    if i in (250, 750, 1000, 1250):
        stills[i] = fr.copy()
    if i % SEG == SEG - 1 or i == n - 1:
        p.stdin.close()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"seg rc={rc}: {p.stderr.read().decode()[-1200:]}")
        print(f"seg {i - SEG + 1}..{i} done, md5_bad so far {md5_bad}", flush=True)

wall_dec = time.time() - t_dec0
cpu1 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
for i, fr in stills.items():
    np.save(os.path.join(OUTDIR, f"fresh_dec_still_{i}.npy"), fr)

with open(os.path.join(OUTDIR, "_concatN.txt"), "w") as fh:
    for s in segments:
        fh.write(f"file '{s}'\n")
subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", os.path.join(OUTDIR, "_concatN.txt"),
                "-c", "copy", "-movflags", "+faststart",
                os.path.join(OUTDIR, "photon_fresh_4k50_200kbps_demo.mp4")], check=True)
sz = os.path.getsize(os.path.join(OUTDIR, "photon_fresh_4k50_200kbps_demo.mp4"))
print(f"DECODE DONE wall {wall_dec:.0f}s ({n/wall_dec:.2f} f/s incl mp4 pipe), "
      f"cpu {cpu1-cpu0:.0f}s, md5 mismatches {md5_bad}/{n}, mp4 {sz/1e6:.1f} MB", flush=True)
json.dump({"wall_s": wall_dec, "cpu_s": cpu1 - cpu0, "md5_bad": md5_bad,
           "n": n}, open(os.path.join(OUTDIR, "fresh_decode_stats.json"), "w"))
