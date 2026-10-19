"""REVERIFIED test suite (step 8): decodes a freshly-encoded short stream with
the current .so and verifies every decoded frame against a second, independent
decode pass (self-consistency + determinism)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import photon


def _gen_frames(seed, n, w, h, fps):
    frames = []
    for t in range(n):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = 0.4 * t
        xx = xx - (200 + 8 * t)
        xxr = (xx * np.cos(a) - (yy - h / 2) * 0.01 * t + 30 * t / w) % w
        img = (np.sin(xx / (w / 6.28) + t * 0.3) * 0.5 + 0.5)
        img = (img * (yy / h * 0.7 + 0.3) + 0.25 * np.sin((yy + 3 * t) / 9.0))
        img = np.clip(img, 0, 1)
        frame = np.empty((h, w, 3), np.uint8)
        frame[..., 0] = (img * 255).astype(np.uint8)
        frame[..., 1] = ((1 - img) * 255 * 0.8 + 20).astype(np.uint8)
        frame[..., 2] = ((xxr / w) * 255 * 0.9 + 10).astype(np.uint8)
        frames.append(frame)
    return frames, fps


def _encode(frames, w, h, fps, preset_name):
    from photon.codec import preset
    from photon.cli import Encoder
    cfg = preset(preset_name, w, h, fps)
    enc = Encoder(cfg)
    for f in frames:
        enc.push_frame(f)
    return enc.finish()


def run(w, h, n, preset_name):
    from photon.frame import Decoder
    frames, fps = _gen_frames(3, n, w, h, fps=30)
    t0 = time.perf_counter()
    blob = _encode(frames, w, h, fps, preset_name)
    t_enc = time.perf_counter() - t0
    dec = Decoder(blob)
    outs = []
    t1 = time.perf_counter()
    for i in range(n):
        fr = dec.decode_frame(i).copy()        # global 4-buf rotation: copy out
        assert fr.shape == (h, w, 3)
        outs.append(fr)
    t_dec = time.perf_counter() - t1
    dec2 = Decoder(blob)
    for i in range(n):
        r = dec2.decode_frame(i)
        assert (r == outs[i]).all(), f"frame {i} self-decode mismatch"
    print(f"PASS {w}x{h} {preset_name}: {n} fr, enc {t_enc:.1f}s,"
          f" dec {1000*t_dec/n:.2f} ms/fr, {len(blob)} B")
    return t_dec / n


def main():
    per = run(320, 240, 5, "photon-good")
    per2 = run(320, 240, 5, "photon-lean")
    print(f"avg decode at 320x240: good {per*1000:.2f} ms, lean {per2*1000:.2f} ms")


if __name__ == "__main__":
    main()
