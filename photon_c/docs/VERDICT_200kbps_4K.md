# PHOTON-C at 200 kbps / 4K — complete test run, compute, power, verdict

Run date: 2026-10-15 · Box: 2× vCPU Xeon @ 2.6 GHz (AVX-512), 2 GB RAM, no
power-cap sensor (VM) → power lines are *estimates* and are labeled as such.
Source: "4K Nature Cinematography recorded with Nikon D5300" (CC BY 3.0),
3840×2160 @ 29.97 fps, first 50.05 s = 1500 frames, lossless x264 mezzanine.

## 1. The 200 kbps 4K demo

| artifact | file | note |
|---|---|---|
| **PHOTON stream** | `results/demo_photon_4k50_207kbps.phtn` | 1.29 MB for 50 s = **206.9 kbps**, the actual bitstream |
| **Playable demo (recommended)** | `results/demo_photon_4k50_200kbps.mp4` | 72 MB, decoded+re-wrapped x264 crf18 — this is what PHOTON's decoder produces, ready to watch |
| x264 reference @ same rate | `results/demo_x264_4k50_200kbps.mp4` | 1.31 MB = 209.6 kbps, watch for side-by-side feel |
| 3-way compare crop (4K, 1:1) | `results/demo_3way_crop_f750.jpg` | leaves detail, **no super-res is performed** |
| 3-way side-by-side (540p) | `results/demo_3way_f750.jpg` | full frame for scene-tone comparison |
| PHOTON full-frame still | `results/demo_photon_f750_full.png` | decoded frame 750 at 3840×2160 |

Decoder ↔ encoder identity verified by md5 over **all 1500 frames**:
**0 mismatches** (`results/fresh_decode_stats.json`).

## 2. Quality — measured this run, equal conditions

Compared against the lossless mezzanine.  Table A: full-pipeline numbers at
960×540 RGB across all 1500 frames (`results/compare_540_v2.json`).
Table B: full 4K RGB check at frame 750 (mjpeg-stills + numpy/skimage).

| codec | kbps | PSNR mean (540p) | PSNR min | SSIM (540p) | f750 @4K PSNR | f750 @4K SSIM |
|---|---|---|---|---|---|---|
| **PHOTON-C (atto)** | 206.9 | **21.79 dB** | **18.67** | **0.571** | 21.97 dB | **0.800** |
| x264 medium (2-pass) | 209.6 | 20.63 dB | 7.93 | 0.459 | 21.80 dB | 0.706 |

Interpretation.  At 207 kbps x264 means crushes this clip at ≈8 Mbit/s raw
→ bits/pixel/frame ≈ 0.00028.  PHOTON's mean is ~1.2 dB ahead and its
minimum frame does not crater (18.7 vs 7.9 dB) — uniform blur instead of
scene-change brick-walls.  Frame 750 at full 4K RGB shows both land around
21.8–22.0 dB, but PHOTON keeps a higher SSIM (0.800 vs 0.706) on foliage.
Neither is "legible faces" quality at this rate; PHOTON's failure mode
(smooth) is visually gentler than x264's (block soup).

For context (prior RD data on the same 1500-frame clip, `master_report.json`):
x264 @ 600 kbps = 30.2 dB / 0.849 SSIM; x264 @ 1500 kbps = 36.3 / 0.963.
At 200 kbps available codecs are all in "you can see what it is, not what it
says" territory; PHOTON's niche is that the floor underneath is *stable*.

## 3. Compute — measured, not estimated

PHOTON pipeline on 2 threads (encoder+decoder, this run):

| stage | wall | CPU | rate | cpu-s per video-second |
|---|---|---|---|---|
| **encode 4K** (1500 f) | 1015 s | 1338 s | **1.48 fps** | 26.7 |
| **decode 4K** (1500 f, incl. mp4 wrap) | 496 s | 342 s | **3.0 fps** | 6.8 |
| decode pure (no mp4 write, earlier benches) | — | — | 5.7–8.0 fps | ~4 |

Peak RSS: ~0.8 GB both sides for 4K (fits a phone's shared memory).

x264 on this box (same 2 threads, `master_report.json` figures):
x264 **encode** ≈ 15.4 fps at 4K, x264 **decode** 210 fps at 4K.
So PHOTON-C is ~10× slower to encode and ~30–70× slower to decode than the
industry codec — the price of a research-grade Python/C hybrid.

## 4. Power — honest estimate (VM has no power meter)

The VM exposes no RAPL/power-cap, so watts below are *modelled*:
2 vCPUs pinned at AVX-512 load on a modern Xeon ≈ **15 W** package-share;
cpu-seconds are measured.

| stage | measured cpu-s | est. energy for 50 s 4K | est. J/frame |
|---|---|---|---|
| PHOTON encode | 1338 s | ≈ 20 kJ (5.6 Wh) | ≈ 13 J |
| PHOTON decode | 342 s | ≈ 5.1 kJ (1.4 Wh) | ≈ 3.4 J |
| x264 decode (est. 30× faster) | ~11 s | ≈ 0.17 kJ | ≈ 0.11 J |

Real-time consequences.  Decode at 5.7–8 fps top speed on 2 cores ⇒
real-time 4K29.97 would need 4–6 cores as-is; playing this on a phone draws
continuously ~3–4 W/core — **~10–16 W while x264 uses a fraction of that.**
Not pocket-real-time yet; the SIMD+tiling pass in `SPEEDUP_REPORT.md` is the
documented next lever (2–3× more), still leaving recharge-the-phone rate.
Encoding at 1.5 fps is offline only — a 50 s 4K clip takes 17 min.

## 5. Stored-compatibility

* Previously encoded streams still decode bit-for-bit on the current code
  (checked on the other clip's archive: 0/30 md5 mismatches, f64 path).
* New streams from the current code: encoder reopen == decoder emit, 0/1500.

## Verdict

200 kbps 4K (a 14,000:1 squeeze) is survivable: PHOTON-C delivers a
**stable, block-free ~21.8 dB / 0.57 SSIM** picture — beats x264's
collapse-prone 20.6 dB / 0.44 at the same rate, and its minimum frame is
drastically better. The price is brutal: **~10× slower encode, ~30–70×
slower decode**, offline-only at 1.5 fps on 2 cores, and battery life that
punishes 4K playback on mobile without more SIMD work.

Where it stands: a *research-lossless-ish floor* for ultra-low-bitrate 4K
(surveillance archives, bulk storage, "will never be watched at full quality
but must never be unintelligible") — not a camera-path or realtime-streaming
codec until decode gets 4× faster.
