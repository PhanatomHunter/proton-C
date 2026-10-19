# PHOTON-C

A working research video codec built for one regime: **extremely low bandwidth**.
It streams real 4K UHD at **~200–1600 kilobits per second** — where the raw signal
needs **5,972 Mbit/s** — using a Laplacian-pyramid decomposition plus a decoder-
side texture synthesis engine, instead of committing the entire bitrate budget to
block residuals the way hybrid codecs do.

It is delivered as a real, running system: an encoder, a decoder, a CLI, a
benchmark harness, and a live web demo (**PHOTON Studio**, port 3000) that encodes
clips on demand and plays back the codec's own output.

---

## The design in two measurements

Both of these were measured on real footage before a line of the encoder was
finalised:

1. **Detail is temporally predictable.** Correlation between consecutive frames'
   detail bands is **0.967–0.999**; a single gain applied to the previous band
   predicts the current one with **+11.9 to +29.4 dB** less MSE than the band
   itself carries. High-frequency content in natural video is overwhelmingly
   static texture that only *appears* to move.

2. **Detail is NOT spatially predictable.** An upsampled lower band predicts the
   band above it at correlation **0.03–0.13**, worth at most **+0.12 dB**. A
   two-term predictor trained on the base adds no more. Sharpening a coarse
   image does not invent texture.

Conclusion: **detail can only ever come from a frame that actually transmitted
it.** That dictates the architecture:

* Each frame is decomposed into a **Laplacian pyramid**: one small base + N
  detail bands.
* Only the base and a configurable number of the lowest bands are transmitted
  (`tx_depth`). For 4K the extreme preset sends a **72×120-pixel base** and
  nothing else.
* Missing bands are **synthesised** on both sides from the *previous frame's
  own band in that position*, warped by a motion field both sides **derive**
  from their reconstructions (never transmitted — at 4K a transmitted flow
  field would cost ~184 kB/frame, more than all coefficients put together),
  boosted by a saved per-band energy map, and dithered with seeded noise.
* A **progressive refresh** instead of a keyframe: frame 0 sends the base
  intra, frame 1 adds band 0, frame 2 adds band 1, …  Peak frame size stays
  near steady state — a 4K frame is ~3.5 KB — so a fixed-bandwidth stream never
  sees a megabyte spike.

## Results (measured, two-pass ABR, identical frames & GOP for all codecs)

**The headline: 50 seconds of REAL 4K from the internet** — 1500 frames of
3840×2160 @ 29.97 nature footage (Nikon D5300, CC BY 3.0, Wikimedia Commons),
fully streamed through encoder and decoder on this box (`drive4k_real.py`,
checkpoints in `results/real4k_*.json`):

| codec | kbps (measured) | PSNR | SSIM |
|---|---|---|---|
| **PHOTON-C atto** | **207.8** | **21.58** | 0.635 |
| x264 (slow, 2-pass target 200k) | 209.6 | 19.23 | 0.521 |
| **PHOTON-C femto** | **319.0** | **23.63** | 0.687 |
| x264 (slow, target 600k) | 621.5 | 30.24 | 0.849 |
| **PHOTON-C nano** | **748.7** | **26.04** | 0.744 |
| **PHOTON-C lean** | **2213.7** | **28.27** | 0.798 |
| x264 (slow, target 1500k) | 1565.3 | 36.28 | 0.963 |

At matched ~208 kbps: **PHOTON-C 21.58 dB vs x264 19.23 dB (+2.35 dB)** — raw
4K RGB is 5.97 Gbit/s, so atto runs **28,700× smaller**. The crossover is
~500–600 kbps: above it x264's coefficient coding wins decisively. PHOTON-C's
curve is nearly flat because quality comes from synthesis, not payload.
End-to-end proof: the atto stream was re-decoded through the real `Decoder`
path — **1500/1500 frames md5-identical** to the encoder's closed loop
(4.22 f/s decode on 2 cores). Figures: `results/real4k_rd.png`,
`real4k_traces.png`, `real4k_crops.jpg`.

**4K UHD 3840×2160, 24-frame stream** (synthetic 1/f²-spectrum source calibrated
to real footage's downsampling ceiling; x264 at default `preset slow`):

| codec | kbps (actual) | PSNR | SSIM |
|---|---|---|---|
| PHOTON-C atto48 | 214.9 | 24.52 | 0.548 |
| PHOTON-C atto96 | 441.7 | 26.91 | 0.568 |
| PHOTON-C nano136 | 1011.8 | 28.75 | 0.605 |
| PHOTON-C lean136 | 1625.3 | 28.39 | 0.609 |
| x264 (slow) | 406.4 | 23.74 | 0.521 |
| x264 (slow) | 478.4 | 25.15 | 0.539 |
| x264 (slow) | 926.5 | 28.50 | 0.628 |
| x265 (medium, ref≤2) | 2659–2702 *(overshot 3–9× targets)* | 30.66–32.62 | 0.81–0.88 |
| AV1 (libaom cpu-8) | — | **OOM-killed at 4K in this 2 GB sandbox** | — |

Matchet rate at ~430 kbps: **PHOTON-C 26.91 dB vs x264 ≈ 24.5 dB** (+2.4 dB).
At ~950 kbps the two are within 0.25 dB. PHOTON-C wins the sub-1 Mbps regime it
was built for; ≥ ~2.6 Mbps HEVC's stronger coefficient coding dominates.

**1080p real footage (Big Buck Bunny, 60 frames):** PHOTON-C trails x264 by
~4 dB and AV1 by ~6 dB at matched bitrate (23.99 vs 28.37 dB @ ~500–620 kbps).
See `results/rd_curves.png` and the studio's *Verdict* tab for the full honest
position. The gap is a coefficient-coding problem — the base-layer entropy
coder is a flat rANS vs CABAC — and it is the most conventional, least funded
part of the design.

## Repo layout

```
photon/
  photon/rans.py        byte-oriented rANS entropy engine (+ why not CABAC, measured)
  photon/transform.py   8×8 DCT · JPEG quants · BT.601 · deblock
  photon/motion.py      DIS flow · block SAD · skip decisions
  photon/synthesis.py   pyramid · guided detail · band synthesis (the engine)
  photon/codec.py       plane-level coder (bit-exact), Config + recalibrated presets
  photon/frame.py       3-plane frame layer, progressive refresh, closed-loop RDO
  photon/cli.py         photon encode / decode / info / presets / bench
  tests/test_rans.py    entropy engine suite (6000 tables, adversarial dists, 4K maps)
  benchlib.py           disk-streamed benchmark harness (metrics, yuv sources, refs)
  results/              measured JSON + rd_curves.png + comparison imagery
  studio/server.py      PHOTON Studio — live encoder console on 0.0.0.0:3000 (stdlib only)
```

## Use it

```bash
pip install imageio-ffmpeg          # whole-clip IO; the only dependency beyond numpy/opencv/scipy
python3 -m photon presets
python3 -m photon encode in.mp4  -o out.phtn --preset photon-hd-nano
python3 -m photon encode in.mp4  -o out.phtn --kbps 400
python3 -m photon decode out.phtn -o recon.mp4
python3 -m photon info   out.phtn
python3 -m photon bench  --video in.mp4 --res 640x360 --ref-codecs libx264

python3 studio/server.py            # -> http://localhost:3000  live console + benchmarks
```

**Compute & power audit** (same 1500 4K frames, 2-core sandbox, measured by
`drive_eff.py` → `results/eff4k.json`; CPU-time exact via `getrusage`, energy
estimated at 10 W sustained as no power counter exists here):

| pass | wall | f/s | CPU-s per video-s | peak RAM | note |
|---|---|---|---|---|---|
| PHOTON-C encode (atto) | 549 s | 2.73 | 16.1 | **0.81 GB** | pure Python, no SIMD/threads |
| PHOTON-C decode (atto) | 264 s | 5.69 | 7.9 | **0.81 GB** | synthesis runs at full res |
| x264 slow encode, 1-pass | 98 s | 15.4 | 3.8 | 1.05 GB | quality table used 2-pass (≈×2) |
| x264 decode | 7 s | 210.2 | 0.28 | 1.05 GB | SIMD, 25 years of tuning |

Honest read: PHOTON-C wins the wire (28,700× below raw) and loses the meter —
encoders ~4× x264's CPU, decoders ~28×. Memory is the exception (lower than
the reference). Energy ≈ 9.7 kWh/h-of-4K encode and 4.7 kWh/h decode at 10 W
vs 2.2 and 0.17 for x264. The synthesis path has no vectorisation or threading
yet; those are engineering, not architecture, and they multiply downward on
real silicon.

### Native port: what actually happened (measured, not promised)

Following the standard "port to C with SIMD and threads" recipe on the
measured #1 decode hotspot (colour conversion, ~30% of frame time), we
shipped `native/` (C + OpenMP, 9-buffer LUT/BT.601-exact kernels loaded via
ctypes with silent NumPy fallback) and verified it the way production codecs
must be verified:

* **Bit-exactness proven exhaustively** — every possible `YCrCb→RGB` uint8
  input (2^24 points) matches cv2's transform exactly; plus md5-verification
  shows all decoded frames of the real 4K stream are unchanged (`tests/test_native.py`).
* **Speed, op-level:** 54 ms → 24 ms/frame at 4K micro-bench (2 threads).
* **Speed, end-to-end on this 2-core noisy VM:** bounded by measurement
  noise (±35% run-to-run); the real wins here were architectural fixes the
  port exposed: three 66 MB f64→f32 conversions eliminated per frame, a
  25 MB page-fault-per-frame output alloc replaced by buffer rotation,
  and OpenMP/cv2 thread-pool contention diagnosed and contained.

The honest read on "20–50× by porting": that applies to scalar loops.
NumPy's full-frame ops are already vectorised C; the residual Python tax is
*inter-plane orchestration*, which the native pass removes per-kernel but
which buys single-digit percentages each, not orders of magnitude. The four
big levers that remain, in order: (1) float32 planes end-to-end (halves all
full-res traffic; needs precision drift re-validation through the DC chains),
(2) fusing the pyramid rebuild's Lanczos resize chain (39% of decode), (3)
porting `synthesise_band`/`warp_band`, (4) thread-tiling those fused stages.
Build: `sh native/build.sh`.

## Known limitations

* **Encoder is offline**, ~2 f/s at 4K single-core NumPy (the decoder ~0.8 f/s).
* The base-layer coder is deliberately simple; closing the 1080p gap means a
  context-adaptive entropy coder and proper RDO, both well-understood.
* Synthesis reproduces *texture*, not *structure*: text, logos and faces at
  distance must be refreshed as NC coefficients, which they are — periodically.

## Security hardening (decoder fuzz battery)

The decoder consumes attacker-controlled bytes, so `photon/guard.py` wraps every
untrusted read: magic/version/fps/header-field validation *before any
allocation*, payload count/length caps and bounds (MAX_PACKET 1 GB, rANS
MAX_SYMS 4M), level geometry cross-checked against the trusted size chain
(this killed a `65535^2`-block numpy allocation bomb), `n_coded <= blocks`
(killed a 4-billion-iteration decode loop), zero-frequency rANS tables rejected.
`tests/fuzz_decoder.py` runs 831 cases — 800 random byte flips, truncations,
garbage, hostile headers, length-field bombs, deleted sections:

    cases: 831   clean StreamError: 226   decoded-partial-payload: 605
    unexpected exceptions: 0   hangs: 0   memory bombs: 0

Everything surfaces as `StreamError`; the CLI reports it and exits non-zero.
Known remaining gap (by design, roadmap): the bitstream carries no integrity
tag, so a corrupt payload can decode to *wrong pixels without error* — a
frame-level CRC32 is a cheap future add for transport.

## Good-internet tier

Three presets for pipes that can carry it (`photon-good`, `photon-good-4k`,
`photon-good-sd`): same architecture, synthesis off, base + both detail bands
transmitted at qp 88 with a short GOP.  Measured: 4K at 5.5 Mbit/s -> 38.2 dB
(natural-ish content; harder 4K textures drive 15-25 Mbit/s), 360p at 1085
kbit/s -> 31.7 dB, encode 12-15 f/s @360p and decode 43-52 f/s on 2 cores.
The "ceiling" is the existing rate control: `--kbps N` walks the preset ladder
until PHOTON lands under N (e.g. target 500 -> settled 356 kbit/s, ladder moved
from photon-good-sd to photon-hd-femto automatically).  Use it like:

    python3 -m photon encode in.mp4 -o out.phtn --preset photon-good-4k --kbps 20000  # <= 20 Mbit/s
