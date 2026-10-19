"""
PHOTON-C -- a compression engine for video at bandwidths where conventional
codecs stop working.

The design in one paragraph
---------------------------
Every hybrid codec spends its bits the same way: predict each block from
neighbours, transform the residual, quantise, entropy-code.  That works down to
roughly 0.05 bits per pixel and then collapses, because at 0.001 bpp there is no
residual left to code -- the prediction itself is the entire budget.

PHOTON-C spends the budget differently.  A frame is decomposed into a Laplacian
pyramid: one small base layer plus a stack of detail bands.  Only the base and a
configurable number of the lowest detail bands are ever transmitted.  The rest are
*synthesised* on both sides from the same deterministic recipe -- borrowed forward
in time from the last frame that did transmit them, warped by a motion field both
sides derive independently, and topped up with seeded noise so flat areas do not
look painted on.  Because the recipe is deterministic and its inputs are the
reconstructions both sides already hold, nothing about it has to be signalled.

Two measurements decided this design, both on real footage:

* Consecutive frames' detail bands correlate 0.967-0.999, and the previous frame's
  band predicts the current one with +29.4 dB of MSE reduction.  Temporal transfer
  of detail is close to free.
* Upsampling band k-1 to predict band k correlates 0.025-0.131 and is worth at most
  +0.12 dB.  Spatial extrapolation of detail is worthless.

So detail can only come from a frame that actually transmitted it.  That is why
the codec uses a progressive refresh -- frame 0 sends the base, frame 1 adds band
0, frame 2 adds band 1 -- instead of a conventional keyframe, which at 4K would be
megabytes in a single frame and would wreck any streaming buffer.

Module map
----------
    rans        byte-oriented rANS entropy coder (the arithmetic-coder attempt is
                documented in rans.py's header as a dead end, not hidden)
    transform   8x8 DCT, JPEG quantisation tables, BT.601 colour, deblocking
    motion      DIS optical flow, block motion search, skip decisions
    synthesis   the Laplacian pyramid, band synthesis, guided detail, seeded noise
    codec       plane-level coder: one pyramid band, intra or inter, bit-exact
    frame       frame layer: three planes, progressive refresh, closed-loop encode
    cli         command line entry point

Everything that crosses the bitstream boundary is reproduced exactly on both
sides; the test suite in tests/ checks that, because a codec that is merely
*approximately* symmetric is not a codec.
"""

__version__ = "1.0.0"
__all__ = ["rans", "transform", "motion", "synthesis", "codec", "frame", "cli"]
