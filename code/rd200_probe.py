#!/usr/bin/env python3
"""R-D probe around 200 kbps on the Nikon 4K nature clip (CC BY 3.0).

Question: can PHOTON-C reach 30 dB PSNR at ~200 kbps on this content, and if
not at 4K, at which (resolution, fps, config) does the picture stop being
blurry?  Every config is: stream frames from source (no retention in RAM),
encode, stream-decode, PSNR vs the same source frames, kbps from blob size.

Run:  cd /home/user/photon && PHOTON_F32=1 python3 rd200_probe.py
"""
import os, sys, json, time, subprocess
import numpy as np
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PHOTON_F32", "1")

from photon.codec import preset, Config
from photon.cli import Encoder
from photon.frame import Decoder

SRC = "/home/user/testvideo/nikon4k_50s.webm"
FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

N = 120          # probe frames per config
PRE = 0.0        # lead-in seconds to skip into the clip


def frames(out_w, out_h, out_fps, n=N):
    """Stream n frames from the 4K source, fps-converted+scaled on the fly."""
    p = subprocess.Popen(
        [FF, "-v", "error", "-ss", f"{PRE:.2f}", "-i", SRC,
         "-vf", f"fps={out_fps},scale={out_w}:{out_h}:flags=lanczos",
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    fsz = out_w * out_h * 3
    for _ in range(n):
        chunk = p.stdout.read(fsz)
        if len(chunk) < fsz:
            break
        yield np.frombuffer(chunk, dtype=np.uint8).reshape(out_h, out_w, 3)
    p.stdout.close(); p.wait()


def probe(name, w, h, fps, cfg):
    # ---- encode ----
    enc = Encoder(cfg)
    nf = 0
    t0 = time.perf_counter()
    for fr in frames(w, h, fps):
        enc.push_frame(fr)
        nf += 1
    blob = enc.finish()
    t_enc = time.perf_counter() - t0
    kbps = 8 * len(blob) / (nf / fps) / 1000.0

    # ---- decode + PSNR (stream source again as reference) ----
    dec = Decoder(blob)
    psnrs = []
    t1 = time.perf_counter()
    for i, ref in zip(range(nf), frames(w, h, fps, nf)):
        fr = dec.decode_frame(i)
        if fr.shape != ref.shape:
            fr = fr[:ref.shape[0], :ref.shape[1]]
        mse = float(np.mean((fr.astype(np.float64) - ref) ** 2))
        psnrs.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    t_dec = time.perf_counter() - t1

    rec = dict(name=name, w=w, h=h, fps=fps, n=nf, kbps=round(kbps, 1),
               psnr=round(float(np.mean(psnrs)), 2),
               psnr_p05=round(float(np.percentile(psnrs, 5)), 2),
               psnr_min=round(min(psnrs), 2),
               dec_fps=round(nf / t_dec, 1), enc_fps=round(nf / t_enc, 2))
    print(json.dumps(rec), flush=True)
    return blob, rec


def custom(w, h, fps, tx, qp, gop, bms, cms=None):
    return Config(width=w, height=h, fps=fps, gop=gop, qp=qp,
                  tx_depth=tx, base_min_side=bms,
                  chroma_min_side=cms if cms is not None else max(48, bms // 2))


def main():
    jobs = []
    only = sys.argv[1] if len(sys.argv) > 1 else None

    def add(name, w, h, fps, *pcert):
        jobs.append((name, w, h, fps, pcert[0]))

    # -- same-clip baseline at 4K: the deliverable's own tier ----------------
    add("4K30-atto(64,qp66,tx0)", 3840, 2160, 30,
        preset("photon-atto", 3840, 2160, 30))
    # -- tuned base size at 4K ~200 kbps --------------------------------------
    add("4K30-base96,qp64,tx0", 3840, 2160, 30,
        custom(3840, 2160, 30, 0, 64, 60, 96))
    # -- resolution ladder at ~200 kbps ---------------------------------------
    add("1080p30-femto(96)", 1920, 1080, 30, preset("photon-hd-femto", 1920, 1080, 30))
    add("1080p30-base136,qp66", 1920, 1080, 30, custom(1920, 1080, 30, 0, 66, 60, 136))
    add("1080p15-nano(136)", 1920, 1080, 15, preset("photon-hd-nano", 1920, 1080, 15))
    add("720p30-nano(136)", 1280, 720, 30, preset("photon-hd-nano", 1280, 720, 30))
    add("720p15-base192,qp68", 1280, 720, 15, custom(1280, 720, 15, 0, 68, 60, 192))
    add("540p15-base192,qp70", 960, 540, 15, custom(960, 540, 15, 0, 70, 60, 192))
    add("540p30-sd(80,qp70)", 960, 540, 30, preset("photon-sd", 960, 540, 30))
    add("480p15-sd-lean(96,tx1)", 854, 480, 15, preset("photon-sd-lean", 854, 480, 15))

    # -- round 2 --------------------------------------------------------------
    # GOP impact (P-frame bits are almost free vs I-frames at these rates):
    add("4K30-atto-gop150", 3840, 2160, 30, custom(3840, 2160, 30, 0, 66, 150, 64))
    add("1080p30-femto-gop150", 1920, 1080, 30, custom(1920, 1080, 30, 0, 70, 150, 96))
    # 30-dB hunting at small res / low fps:
    add("360p15-base136,qp70", 640, 360, 15, custom(640, 360, 15, 0, 70, 150, 136))
    add("360p10-base136,qp68", 640, 360, 10, custom(640, 360, 10, 0, 68, 150, 136))
    add("420p10-base128,qp70", 746, 420, 10, custom(746, 420, 10, 0, 70, 150, 128))
    add("480p10-base160,qp70", 854, 480, 10, custom(854, 480, 10, 0, 70, 150, 160))
    add("540p10-base192,qp70", 960, 540, 10, custom(960, 540, 10, 0, 70, 150, 192))
    # watchable middle ground near 200-300 kbps:
    add("720p24-base136,qp66,g150", 1280, 720, 24, custom(1280, 720, 24, 0, 66, 150, 136))
    add("1080p24-base96,qp64,g150", 1920, 1080, 24, custom(1920, 1080, 24, 0, 64, 150, 96))
    # 4K at ~200 with longer GOP + slightly bigger base:
    add("4K30-base80,qp64,g150", 3840, 2160, 30, custom(3840, 2160, 30, 0, 64, 150, 80))

    allrec = []
    fn_done = os.path.join(RESULTS, "rd200_probe.json")
    done = set()
    if only in (None, "all") and os.path.exists(fn_done) and os.environ.get("REDO", "0") != "1":
        done = {r["name"] for r in json.load(open(fn_done))}
    for name, w, h, fps, cfg in jobs:
        if only == "new" and name in done:
            continue
        if only and only not in ("all", "new") and only != name:
            continue
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] probe {name} ({w}x{h}@{fps}) ...", flush=True)
        _, rec = probe(name, w, h, fps, cfg)
        allrec.append(rec)
        print(f"    -> {rec['kbps']} kbps  PSNR {rec['psnr']} (min {rec['psnr_min']})"
              f"  dec {rec['dec_fps']} fps", flush=True)
    if allrec:
        fn = os.path.join(RESULTS, "rd200_probe.json")
        old = json.load(open(fn)) if os.path.exists(fn) else []
        old = [r for r in old if all(r["name"] != a["name"] for a in allrec)]
        json.dump(old + allrec, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
