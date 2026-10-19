#!/usr/bin/env python3
"""PSNR/SSIM of x264/x265 200k refs vs mezzanine, sampled PSNR on all frames
(median grid), full SSIM on frames (750, 1000, 1250), plus playtime figures."""
import json, os, subprocess, sys
import numpy as np
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
import cv2
from skimage.metrics import structural_similarity as ssim

MEZZ = os.path.join(HERE, "out", "mezz_4k50.mkv")

def stats_for(path):
    # ffmpeg-native PSNR/SSIM both at 960x540 (same as enc-side metrics style)
    p = subprocess.run([FF, "-hide_banner", "-i", MEZZ, "-i", path,
                        "-lavfi", "scale=960:540,psnr=stats_file=/dev/stdout",
                        "-f", "null", "-"], capture_output=True, text=True)
    psnr_lines = [l for l in p.stdout.splitlines() if l.startswith("frame:")]
    vals = [float(l.split("psnr_avg:")[1].split()[0]) for l in psnr_lines]
    psnr_avg = float(np.mean(vals))
    s = subprocess.run([FF, "-hide_banner", "-i", MEZZ, "-i", path,
                        "-lavfi", "scale=960:540,ssim=stats_file=/dev/stdout",
                        "-f", "null", "-"], capture_output=True, text=True)
    svals = [float(l.split("All:")[1].split()[0]) for l in s.stdout.splitlines()
             if l.startswith("n:")]
    kb = os.path.getsize(path) / 1e3
    kbps = 8.0 * os.path.getsize(path) / (len(vals) / 29.97) / 1000.0
    return {"frames": len(vals), "psnr_avg_540p": psnr_avg,
            "ssim_avg_540p": float(np.mean(svals)), "mb": kb / 1024,
            "kbps": kbps, "psnr_min": float(np.min(vals)),
            "psnr_max": float(np.max(vals))}

out = {}
for name in ("x264_200k_4k.mp4", "x265_200k_4k.mp4"):
    p = os.path.join(HERE, "out", name)
    if os.path.exists(p):
        out[name] = stats_for(p)
        print(name, json.dumps(out[name], indent=1), flush=True)
json.dump(out, open(os.path.join(HERE, "out", "ref_200k_metrics.json"), "w"), indent=1)
