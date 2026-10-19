#!/usr/bin/env python3
"""Fresh PHOTON-c atto encode of the *current* mezzanine (rice-paddy 4K, 50 s).

Records measured compute (wall time, process CPU-time via resource) so the
compute/power verdict is honest.  Saves the blob, encoder-side PSNR trace and
per-50-frame RGB recon snapshots for md5-verification during decode.
"""
import hashlib, json, os, resource, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PHOTON_F32", "0")

import cv2
from photon.codec import preset
from photon.frame import Encoder

W, H, N = 3840, 2160, 1500
MEZZ = os.path.join(HERE, "out", "mezz_4k50.mkv")
OUT  = os.path.join(HERE, "out")

cfg = preset("photon-atto", W, H, 30000.0 / 1001.0)
enc = Encoder(cfg)

t0 = time.time()
c0 = (resource.getrusage(resource.RUSAGE_SELF).ru_utime
      + resource.getrusage(resource.RUSAGE_SELF).ru_stime)
psnrs, hashes = [], []
cap = cv2.VideoCapture(MEZZ)
i = 0
while i < N:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = bgr[:, :, ::-1]
    enc.push_frame(rgb)
    rec = enc.last_rgb
    mse = float(((rgb.astype(np.float64) - rec.astype(np.float64)) ** 2).mean())
    psnrs.append(20 * np.log10(255.0) - 10 * np.log10(max(mse, 1e-12)))
    hashes.append(hashlib.md5(np.ascontiguousarray(rec).tobytes()).hexdigest())
    if i % 100 == 0:
        print(f"enc {i}/{N} {i/max(time.time()-t0, 1e-9):.2f} f/s "
              f"psnr={np.mean(psnrs):.2f}", flush=True)
    i += 1
cap.release()
blob = enc.finish()
dt = time.time() - t0
cpu = (resource.getrusage(resource.RUSAGE_SELF).ru_utime
       + resource.getrusage(resource.RUSAGE_SELF).ru_stime) - c0

open(os.path.join(OUT, "fresh_4k50_atto.phtn"), "wb").write(blob)
json.dump({"nframes": i, "blob_bytes": len(blob),
           "kbps": 8 * len(blob) / (i / 29.97) / 1000,
           "psnr_encoder_mean": float(np.mean(psnrs)),
           "psnr_min": float(np.min(psnrs)), "enc_wall_s": dt,
           "enc_cpu_s": cpu, "enc_fps": i / dt, "md5": hashes},
          open(os.path.join(OUT, "fresh_4k50_atto_metrics.json"), "w"))
print(f"ATTO-ENC DONE: {i} frames, {len(blob)/1e6:.2f} MB, "
      f"{8*len(blob)/(i/29.97)/1000:.1f} kbps, psnr {np.mean(psnrs):.2f} dB, "
      f"wall {dt:.0f}s cpu {cpu:.0f}s", flush=True)
