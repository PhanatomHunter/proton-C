# PHOTON-C — FINAL completion report: tests, measurement, package

All green on 2026-10-16, 2-core / ~2 GB sandbox. Nothing below is estimated;
every number is a measurement on this box.

## 1. Test suites — all passing

| suite | what it proves | result |
|---|---|---|
| `tests/test_decoder_safety.py` | decode_all copy-safety + streaming-hash equality | **PASS 7/7** |
| `tests2/test_decoder_safety.py` | encode→decode→fresh-decode determinism (5 frames × 2 presets) | **PASS** |
| `tests/test_seek_consistency.py` (new) | random + mid-GOP seeks bit-equal to sequential decode, 3 presets | **PASS 87/87 seeks** |
| `tests/test_kernels.py` | every new C kernel == numpy fallback bit-exact; fallback vs cv2 ≤ 1e-4 | **PASS 20/20** |
| `tests/test_native.py` | YCrCb→RGB exhaustive 2²⁴ inputs, float front-end, 20 random frames | **PASS 0 / 16.8 M mismatches** |
| `tests/test_rans.py` | rANS entropy coding exactness vs Shannon bound | **PASS (all `[exact]`)** |
| `tests/fuzz_decoder.py` | 831 hostile/corrupted streams | **PASS** (226 clean StreamError, 605 partial, 0 crashes) |

## 2. Bitstream-wide verification (every demo stream)

| stream | frames | check | result |
|---|---|---|---|
| `results/demo_photon_4k50_207kbps.phtn` (4K atto, f64 reference) | 1500 | full-stream md5 vs encoder-recorded | **0/1500 mismatches** (GOP-sampled 25/25 re-confirmed today) |
| `results/hdtier_bbb_lean.phtn` (1080p, lean, f32) | 300 | fresh-decoder double pass md5 | **0/300** |
| `results/hdtier_bbb_photon-good.phtn` (1080p, good, f32) | 300 | fresh-decoder double pass md5 | **0/300** |
| `results/clear200_1080p24-base96_qp64_g150.phtn` (1080p24, clear-200 winner, f32) | 600 | fresh-decoder double pass md5 | **0/600** |
| all 3 f32 streams | — | seek-sampled fresh-vs-fresh | **24/24** |
| all 3 f32 streams | — | seek-vs-sequential content equality | **12/12** |

## 3. Bug found & fixed during this completion pass

**Mid-GOP seek broke on progressive refresh.** Streams with tx>0 refresh one
band per GOP; a seek that landed mid-GOP decoded an *inter* chase-band payload
at a fresh reconstructor state → dc-count StreamError.  Fix in
`photon/frame.py`: seeks now anchor at the last **full-refresh** keyframe
(`Decoder._is_full_refresh`) instead of any intra base, and replay forward —
guaranteeing bit-exact == sequential decode.  Verified by the new
`test_seek_consistency.py` (3 presets × 29 random/mid-GOP seeks bit-equal)
and by seek-vs-sequential 12/12 checks across the three f32 demo streams.

## 4. Final decode-speed table (quiet box, 12-frame samples)

| tier | res | min | median |
|---|---|---|---|
| atto f32 | 3840×2160 | 26.9 ms = **37.2 fps** | 27.7 ms = 36.1 fps |
| atto f64 (reference) | 3840×2160 | 105.7 ms = 9.5 fps | 112.4 ms = 8.9 fps |
| lean f32 | 1920×1080 | 44.3 ms = 22.6 fps | 56.0 ms = 17.9 fps |
| photon-good f32 | 1920×1080 | 71.9 ms = 13.9 fps | 82.7 ms = 12.1 fps |
| clear200 winner f32 | 1920×1080@24 | 10.4 ms = **96.5 fps** | 10.8 ms = 92.6 fps |

(Earlier-session mins of 48.2/143 ms on 4K were measured while benchmark
encodes were still co-tenants; no performance kernels changed since — these
are the same .so measured on the genuinely idle box.)

## 5. Quality ladder at ~200 kbps (from `REPORT_200kbps_clear.md`, 19 configs)

- 4K30 atto: 24.2 dB @197 kbps; ceiling ≤300 kbps at 4K ≈ 26 dB. 30 dB n/a at 200 kbps (physics).
- Delivered clear-demo: **1080p24 @225 kbps → 24.4 dB on 600 frames** (old 4K demo 21.7 dB), md5 0/600.
- 30 dB first appears at **~670 kbps** (720p15).  Everything in `results/rd200_probe.json`.

## 6. Playable demos (x264-wrapped, play anywhere)

| file | content | size |
|---|---|---|
| `results/DEMO_photon_c_4K_200kbps_VERIFIED.mp4` | 60 s 4K, 207 kbps, md5-verified | 195.7 MB |
| `results/DEMO_12s_photon_c_4K_200kbps.mp4` | 12 s excerpt of the above | 21.1 MB |
| `results/DEMO_hd_tier_bbb_photon-lean.mp4` | 5 s 1080p60 BBB, 3.5 Mbps, 25.0 dB | 15.0 MB |
| `results/DEMO_hd_tier_bbb_photon-good.mp4` | 5 s 1080p60 BBB, 24.9 Mbps, **29.5 dB ("crystal clear" tier)** | 23.0 MB |
| `results/DEMO_clear200_1080p24_225kbps.mp4` | 25 s 1080p24 @225 kbps, 24.4 dB | 25.2 MB |

Bitstreams alongside (`results/*.phtn`) re-decode identically — each demo is
byte-for-byte reproducible with the bundled code.

## 7. Rebuild / replay instructions

```bash
cd photon && sh setup_env.sh
python3 tests/test_decoder_safety.py && python3 tests/test_seek_consistency.py
# playable test: decode any .phtn to an mp4 you can play:
python3 demo_build_hd_tier.py        # bbb 1080p lean (needs imageio-ffmpeg)
PHOTON_F32=1 CLEAR200_CFG="1080p24-base96,qp64,g150" N=300 python3 demo_build_clear200.py
```
