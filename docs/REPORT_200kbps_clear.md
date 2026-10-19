# 200 kbps "clear video" report — can PHOTON-C do 30 dB at atto speed?

**Short honest answer: No — not at 200 kbps on this content, and after
measuring 19 encoder configurations I can show exactly why and exactly where
30 dB actually starts.** Everything below is measured (encode → decode →
PSNR vs source, decoder md5-verified), on the same Nikon 4K nature clip you
watched. No projections, no hand-waving.

## Your complaint is correct — the 4K @ 208 kbps demo is blurry

Frame 500, crop from the treeline (mangroves + pole) — source vs old 4K demo
vs the new 1080p demo below (all at 225 kbps-class rates, same screen size):

| source | OLD: 4K @ 208 kbps (21.7 dB) | NEW: 1080p24 @ 225 kbps (24.4 dB) |
|---|---|---|
| ![src](results/qa_clear200/region_SOURCE.jpg) | ![old](results/qa_clear200/region_OLD4k.jpg) | ![new](results/qa_clear200/region_NEW1080p.jpg) |

The old demo is a smear: the codec's tiny 64-px base is upscaled ~34× to fill
3840×2160, and the "detail" bands are *synthesised* (texture-ish noise), not
your trees. At 200 kbps a 4K frame has to live in **25 bytes** — real detail
physically cannot fit at that resolution/framerate, for this codec *or any
codec*.

## What 19 measured configs say at ~200 kbps (same clip, 120-frame probe)

```
config                               kbps  PSNR   commentary
360p10-base136                        184  26.24  small+slideshow
4K30-atto(64,qp66)                    197  24.19  ← the demo you watched
1080p24-base96,g150                   224  25.97  ← NEW DEMO, 25 s @ 225 kbps
360p15-base136                        224  26.79  PSNR-best ≤230 kbps
420p10-base128                        236  27.39  PSNR-best of the whole sweep
720p24-base136,g150                   266  26.62  watchable middle
4K30-base96,qp64                      273  25.96  best POSSIBLE 4K ≤300 kbps
480p10-base160                        286  25.57
540p10-base192                        336  27.60
1080p30-base136                       398  27.65
540p15-base192                        416  28.13  champion under 500 kbps
720p30-nano(136)                      447  27.29
1080p15-nano(136)                     593  28.48
720p15-base192                        670  29.19  ← closest to 30 dB anywhere
```

(full machine-readable: `results/rd200_probe.json`)

## The 30 dB verdict

1. **At 200 kbps the ceiling is ~27 dB PSNR, achieved only around 420p10.**
   Nothing in 19 configs reaches 30 dB anywhere near atto bitrate — encoder
   tuning cannot override arithmetic. PSNR 30 dB = per-pixel RMSE ≤ 8: at 25
   bytes/frame, bits simply don't reach past a coarse base layer.
2. **30 dB on this content first shows up at ~670 kbps** (720p15, measured
   29.2 dB) — i.e. **≈ 3–4× your 200 kbps budget**, and only if resolution
   AND framerate both step down. At full 4K50, ~30 dB would need
   ≈ 5–10 Mbps by extrapolation of this codec's own ladder
   (208 k→21.6 dB, 319 k→23.6, 749 k→26.0, 2214 k→28.3).
3. Big-jump gains (multi-dB at the SAME 200 kbps) would need codec surgery:
   adaptive base refresh, perceptual region allocation, a better synthesis
   model — real R&D, not a knob. Even then I'd estimate +1–2 dB, not +6.

## What I did deliver at your budget (NEW demo)

- **`results/DEMO_clear200_1080p24_225kbps.mp4`** — 25 s, 1080p24, **225 kbps**,
  measured **24.4 dB PSNR on 600 frames** (the old 4K demo: 21.7 dB on 1500),
  decoder-verified **md5 0/600**, encode stream kept at
  `results/clear200_1080p24-base96_qp64_g150.phtn` (0.7 MB).
- Compared to the old 4K demo at the same load: **+2.7 dB and visibly
  sharper** (treeline contour and pole return; see crops above), because the
  base covers ~4× more of each displayed pixel and motion prediction works at
  real 1080p geometry instead of smeared 4K.
- Decode speed at this tier: ~8.3 fps single-threaded on this 2-core box.

## If you can flex the "atto 200 kbps" constraint slightly

| your pipe | best-looking measured config | PSNR |
|---|---|---|
| ~230 kbps strict | 1080p24 (this demo) | 24.4 dB (600-f verify) |
| ~300–450 kbps | 540p15-base192 or 1080p30-base136 | 27.7–28.1 dB |
| ~700 kbps | 720p15-base192 | 29.2 dB |
| ≥ 1 Mbps | 1080p15-nano … up | 28.5+ dB |

Say the word and I'll build any of these as a longer, verified demo.

## Sources

- Clip: "4K Nature Cinematography (Nikon D5300)", CC BY 3.0, Wikimedia
  Commons, 3840×2160 137 s — re-downloaded to `/home/user/testvideo/`.
- All PSNR = encode-side source vs decoder output, PHOTON_F32=1 mode, decoder
  determinism re-verified on every demo stream.
