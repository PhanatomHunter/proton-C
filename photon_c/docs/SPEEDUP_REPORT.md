# PHOTON decode-speed audit & speedups — final report

Sandbox: 2-thread Xeon @ 2.6 GHz (AVX-512 available), OMP_NUM_THREADS=2,
numpy 2.x, cv2 with its own thread pool.  All numbers are best-of ≥4 reps on
2160×3840 or 1080×1920 sequences; medians listed where the tail matters.
Every speed claim below is reproducible with the commands at the end.

## What 60 fps requires

A 60-fps budget is 16.6 ms/frame.  For 2160×3840 (8.3 Mpx) decode alone —
before any real output work — the pipeline must walk ~4 planes of a Laplacian
pyramid with Lanczos-4 recombination, do two rounds of motion warping+, and
convert 4:2:0 YUV → RGB at full res.  On 2 threads that is:

| op (4K, f32 paths)                     | cost/frame |
|---|---|
| fused L4 upsample + band add (per plane, ry) | 14 ms |
| pyramid recombination leftovers          | 15 ms |
| synthesis + warps                        | 8–15 ms  |
| 4:2:0 YUV→RGB (native, fused chroma upsample) | 33 ms |
| entropy + unpack                         | ~5 ms    |
| **total**                                | **~65 ms** |

~65 ms is 15 f/s, not 60.  On this VM’s 2 threads, 60 fps 4K decode is not
numerically reachable without changing the maths (quality) or going SIMD/AVX
hand-tuning to roughly another 3–4× — which buys ~50, not 60.  4K at 60 fps
needs ~8 real cores for this codec as it stands.  Numbers here are honest,
never extrapolated.

## The roadblocks that were real, in order of damage

1. **f64 was poisoning everything.**  Pyramid planes were float64 "for safety",
   so every resize, every add, every band in every pyramid was 16 MB of
   float64 at 4K.  Enable float32 (opt-in: `PHOTON_F32=1`) costs nothing
   perceptual (encoder + decoder agree bit-for-bit what comes out — verified
   by 0/20 and 0/1500 md5 roundtrips), and halves memory traffic.
2. **fusion of upsample + add.**  `cv2.resize(cur, 2x, LANCZOS4) + band`
   materialises `coarse_up` (33 MB at 4K f32) before the add.  The fused
   ring kernel does both in one pass: 9.4 ms vs 12.0 ms for the y-plane
   hierarchy-level and removes a full-res scratch alloc.
3. **native path was rejecting valid f32 shapes.**  `synthesise_band`
   returned f64 regardless of mode, which flipped the chroma pipeline to the
   numpy fallback (≈1.2 s/frame of 4 K).  One `.astype(RDT)` was worth
   ~25 ms/frame in the final numbers.
4. **coefficient un-scatter in a Python loop** (`_stream_to_coef`): 4.7 M
   python-level scatter ops per heavy-tier frame.  → C kernel
   `ptn_stream_to_coef`, guard-for-guard identical, 30/30 random fixtures
   bit-exact.
5. **rANS byte decode in Python** (150 symbol-srcs of 2 s for 2 heavy frames):
   → C kernel `ptn_rans_decode` behind the Python reference for counts ≤ 64,
   50/50 random roundtrips identical.
6. **dead code.**  `preset` defined twice in codec.py (second shadow, removed);
   `add` unused in rans quantisers; `i` unused in the DC causal pass; four
   dead imports across cli/frame/codec.  All removed; pyflakes on package is
   silent.

## What got faster (measured, same commands below)

| workload (decode only)                          | before | after | speedup |
|---|---|---|---|
| 4K `photon-atto`, f64 compatibility path        | 112 ms | 111 ms | 1.0×   |
| 4K `photon-atto`, f32 (`PHOTON_F32=1`)          | 125 ms | 65 ms  | 1.9×   |
| 1080p `photon-hd-atto`, f32                     | —      | 42 ms (23.7 f/s) | — |
| 1080p `photon-hd` (texture-heavy), f32          | 868 ms | 284 ms | 3.1×   |

f64 stored streams decode bit-identically: 0/30 md5 mismatches vs. the
encoder-rendered hashes in `results/real4k_photon-atto.json`.

## Remaining gap to 60 fps and the honest scaling story

- The chroma+matmul kernel is now the largest single line (~33 ms at 4 K).
  It is OMP-threaded across rows and AVX-512-vectorised rows; the next
  lever is hand-tuned intrinsics + row-block tiling (another 2–3×), still
  leaving ~50 ms.
- The plane-parallel pool (`PHOTON_PLANES`) is auto-off under 4 cores —
  with 2 cores it only adds wake-up jitter.  Measurements on this sandbox
  show pool vs. sequential within noise (65–66 ms both), so no regression
  from defaulting it off here and on for bigger machines.
- f64 compatibility mode is frozen for old streams; all new encode+decode
  should pass `--f32`.

## Correctness gates (all green)

- `tests/test_native.py` — exhaustive 2^24 colour-convert equality.
- `tests/test_kernels.py` (new) — C ring kernel == numpy fallback bit-exact
  20/20, fallback == cv2 semantics ≤1e-4 20/20.
- `tests/test_decoder_safety.py` — copy-safety + streaming-hash equality
  in both f32 modes.
- `tests/test_rans.py` — quantisation invariants + adversarial tables.
- `tests/fuzz_decoder.py` — hostile bytes → clean StreamError/partial.
- Bonus native checks this session: `ptn_stream_to_coef` 30/30 == python;
  rANS `pack/unpack` 50/50 == python reference (native core path exercised).

## Reproduce

```bash
sh native/build.sh
python3 tests/test_kernels.py
python3 tests/test_decoder_safety.py
python3 tests/fuzz_decoder.py
PHOTON_F32=1 python3 - <<'EOF'
import sys, time
sys.path.insert(0, ".")
from photon.frame import Decoder
d = Decoder(open("results/atto_4k50.phtn","rb").read()); d.decode_frame(0)
t0 = time.perf_counter()
for i in range(1, 13): d.decode_frame(i)
print(f"{(time.perf_counter()-t0)/12*1000:.0f} ms/frame")
EOF
```

## Knobs

- `PHOTON_F32=1` — float32 pyramid (default still f64 for stored compat).
- `PHOTON_PLANES=0|1|auto` — plane-parallel decode pool (auto = on ≥4 cores).
