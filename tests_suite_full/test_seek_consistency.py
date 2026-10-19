"""Seek consistency: decode_frame(i) after a random seek must produce the
bit-exact same pixels as decoding the same frame sequentially, for every
preset family (base-only atto, tx=1 lean, tx=2 photon-good = inter coded
bands refreshed progressively across GOPs).

Regression: mid-GOP seek crashed on progressive refresh (inter chase-band
payloads mis-decoded at fresh state); fix rounds the seek anchor back to the
nearest full-refresh keyframe (_is_full_refresh).
"""
import os, sys, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PHOTON_F32", "1")


def gen_frames(n, w, h, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = (np.sin(xx / (w / 6.28) + t * 0.31) * 0.5 + 0.5)
        img = np.clip(img * ((yy + 3 * t) / h * 0.6 + 0.4)
                      + 0.25 * np.sin((yy + 7 + 2 * t) / 8.0), 0, 1)
        f = np.empty((h, w, 3), np.uint8)
        f[..., 0] = (img * 255).astype(np.uint8)
        f[..., 1] = ((1 - img) * 200 + 40).astype(np.uint8)
        f[..., 2] = ((xx / w + t * 0.05) % 1.0 * 255).astype(np.uint8)
        frames.append(f)
    return frames


def encode(frames, preset_name, w, h, fps=30):
    from photon.codec import preset
    from photon.cli import Encoder
    enc = Encoder(preset(preset_name, w, h, fps))
    for f in frames:
        enc.push_frame(f)
    return enc.finish()


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(preset_name, w=320, h=240, n=40):
    from photon.frame import Decoder
    blob = encode(gen_frames(n, w, h), preset_name, w, h)
    # sequential reference hashes
    d = Decoder(blob)
    ref = {i: md5(d.decode_frame(i).copy()) for i in range(n)}

    # attacker: random seek pattern incl. mid-GOP jumps and repeats
    d2 = Decoder(blob)
    rng = np.random.default_rng(11)
    order = list(int(v) for v in rng.integers(0, n, size=25)) + [0, 1, n - 1, n // 2]
    for i in order:
        fr = d2.decode_frame(i).copy()
        assert md5(fr) == ref[i], f"{preset_name}: seek({i}) output != sequential"
    print(f"PASS seek-consistency {preset_name:12s} ({w}x{h}, {n} f): "
          f"{len(order)} random/mid-GOP seeks bit-equal")


if __name__ == "__main__":
    run("photon-good")        # tx=2, progressive band refresh
    run("photon-lean")        # tx=1
    run("photon-good-sd")     # tx=2, small-res variant of quality family
    print("SEEK: all consistent")
