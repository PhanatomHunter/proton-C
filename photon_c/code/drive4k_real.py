#!/usr/bin/env python3
"""Complete 50-second REAL 4K test: PHOTON-C vs x264 on internet footage.

Source: "4K Nature Cinematography recorded with Nikon D5300" (Wikimedia
Commons, CC BY 3.0), 3840x2160 @ 29.97 fps, VP9.  We take 50 s = 1500 frames.

Design constraints of the box (2 CPUs, ~1.6 GB usable RAM):
  * No raw 4K frames are held in lists - everything streams through pipes.
  * The source is transcoded ONCE to a lossless x264 mezzanine (~4 GB on
    disk in out/, excluded from snapshots) because cv2's VP9 decoder runs at
    1.1 f/s while the static ffmpeg build hits ~78 f/s.
  * PHOTON metrics are taken from the ENCODER's closed-loop reconstruction
    (mathematically identical to what the decoder emits: both share
    _Reconstructor).  Identity is *proven*, not assumed: the atto blob is
    then fully decoded 1500 frames and every pixel md5-verified against the
    encoder-side hashes.
  * SSIM is sampled every 6th frame (0.94 s/frame at 4K); PSNR on every frame.
  * Every stage checkpoints a JSON so the run is resumable.
"""
import hashlib
import resource
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
os.environ["TMPDIR"] = OUT                      # /tmp is a small RAM tmpfs

import benchlib                                  # noqa: E402
from benchlib import ffmpeg_exe, run, psnr, ssim  # noqa: E402
from photon.codec import preset                   # noqa: E402
from photon.frame import Encoder, Decoder         # noqa: E402

W, H = 3840, 2160
FPS = 30000.0 / 1001.0          # 29.97
N = int(os.environ.get("PHOTON_BENCH_N", 1500))  # 50.05 s
SRC = "/home/user/testvideo/nikon4k_50s.webm"
MEZZ = os.path.join(OUT, "mezz_4k50.mkv" if N == 1500
                    else f"mezz_4k50_smoke{N}.mkv")
RES = os.path.join(HERE, "results")
PHOTON_PRESETS = ["photon-atto", "photon-femto", "photon-nano", "photon-lean"]
X264_TARGETS = [200, 600, 1500]                 # kbps targets; measured rate reported
SSIM_EVERY = 6

os.makedirs(RES, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(obj: dict, name: str) -> None:
    path = os.path.join(RES, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
    log(f"saved {name}")


def done(name: str):
    path = os.path.join(RES, name)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def bgr_frames(path: str, nmax: int = N):
    """Yield HxWx3 uint8 BGR frames from `path`.

    cv2 keeps the decoder inside this process (one small buffer), which is the
    difference between fitting in this box's ~1.6 GB next to a 650 MB encoder
    and getting the OOM killer's attention with a second live ffmpeg process.
    VP9 over cv2 was unusably slow (1.1 f/s) -- that is exactly why the source
    was first transcoded to the lossless-h264 mezzanine, which cv2 chews fine.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = 0
    while n < nmax:
        ok, f = cap.read()
        if not ok:
            break
        yield f
        n += 1
    cap.release()


# --------------------------------------------------------------------------- #
# 0. mezzanine
# --------------------------------------------------------------------------- #
def make_mezz():
    if os.path.exists(MEZZ):
        log("mezzanine exists, skipping transcode")
        return
    log("transcoding VP9 -> lossless x264 mezzanine (one-time) ...")
    t0 = time.time()
    rc, msg = run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-i", SRC, "-frames:v", str(N),
                   "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                   "-x264-params", "ref=1", "-an", MEZZ], timeout=3600)
    if rc != 0:
        raise RuntimeError(f"mezzanine transcode failed: {msg[-400:]}")
    sz = os.path.getsize(MEZZ) / 1e9
    log(f"mezzanine: {sz:.2f} GB in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- #
# 1. PHOTON presets (metrics from encoder closed loop)
# --------------------------------------------------------------------------- #
def run_photon_preset(name: str) -> dict:
    ck = done(f"real4k_{name}.json")
    if ck:
        log(f"{name}: checkpoint found, skipping")
        return ck
    cfg = preset(name, W, H, FPS)
    enc = Encoder(cfg)
    psnrs, byte_trace, ssim_vals, hashes = [], [], [], []
    t0 = time.time()
    for i, bgr in enumerate(bgr_frames(MEZZ)):
        rgb = bgr[:, :, ::-1]
        st = enc.push_frame(rgb)
        rec = np.ascontiguousarray(enc.last_rgb[:H, :W])
        psnrs.append(float(psnr(rgb, rec)))
        byte_trace.append(st["bytes"])
        if i % SSIM_EVERY == 0:
            ssim_vals.append([i, float(ssim(rgb, rec))])
        if name == "photon-atto":
            hashes.append(hashlib.md5(rec.tobytes()).hexdigest())
        if i % 100 == 0:
            dt = time.time() - t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            log(f"{name}: frame {i}/{N}  {i/max(dt,1e-9):.2f} f/s  "
                f"psnr={np.mean(psnrs):.2f} dB  "
                f"kbps={8*sum(byte_trace)*FPS/(i+1)/1000:.0f}  rss {rss:.2f} GB")
        del bgr, rgb, rec
    enc_time = time.time() - t0
    blob = enc.finish()
    nframes = enc.frame_idx
    dur = nframes / FPS
    res = dict(
        codec=f"photon-c ({name})", preset=name, nframes=nframes,
        bytes=len(blob), kbps=8.0 * len(blob) / dur / 1000.0,
        bpp=8.0 * len(blob) / (nframes * H * W * 3),
        psnr=float(np.mean(psnrs)), psnr_min=float(np.min(psnrs)),
        ssim=float(np.mean([v for _, v in ssim_vals])),
        ssim_every=SSIM_EVERY,
        enc_s=enc_time, enc_fps=nframes / enc_time,
        peak_note="metrics from encoder closed-loop recon; decoder-identical "
                  "(proven on photon-atto by md5 of all decoded frames)",
        psnr_trace=psnrs, byte_trace=byte_trace, ssim_samples=ssim_vals,
    )
    if name == "photon-atto":
        res["md5_encoder"] = hashes
        with open(os.path.join(OUT, "atto_blob.phtn"), "wb") as fh:
            fh.write(blob)
    save(res, f"real4k_{name}.json")
    del blob
    return res


# --------------------------------------------------------------------------- #
# 2. End-to-end Decoder proof on the atto stream (all 1500 frames)
# --------------------------------------------------------------------------- #
def validate_atto_decoder(enc_res: dict) -> dict:
    ck = done("real4k_atto_decoder_check.json")
    if ck:
        log("atto decoder check: checkpoint found, skipping")
        return ck
    log("atto: full 1500-frame Decoder pass, md5-verifying every pixel ...")
    with open(os.path.join(OUT, "atto_blob.phtn"), "rb") as fh:
        dec = Decoder(fh.read())
    t0 = time.time()
    mism = 0
    for i in range(dec.nframes):
        h = hashlib.md5(dec.decode_frame(i).tobytes()).hexdigest()
        if h != enc_res["md5_encoder"][i]:
            mism += 1
        if i % 250 == 0:
            log(f"atto decode verify {i}/{dec.nframes}")
    dt = time.time() - t0
    res = dict(nframes=dec.nframes, md5_mismatches=mism,
               decode_s=dt, decode_fps=dec.nframes / dt,
               identical_to_encoder=(mism == 0))
    save(res, "real4k_atto_decoder_check.json")
    log(f"atto decoder check: mismatches={mism}, {dec.nframes/dt:.2f} f/s")
    return res


# --------------------------------------------------------------------------- #
# 3. x264 reference, two-pass ABR
# --------------------------------------------------------------------------- #
def run_x264(target: int) -> dict:
    key = f"real4k_x264_{target}.json"
    ck = done(key)
    if ck:
        log(f"x264 {target}k: checkpoint found, skipping")
        return ck
    out = os.path.join(OUT, f"x264_{target}.mkv")
    passlog = os.path.join(OUT, f"x264_{target}_pass")
    t0 = time.time()
    threads = 2
    for attempt in range(2):
        extra = ["-preset", "slow",
                 "-x264-params", "ref=2:rc-lookahead=10:bframes=2",
                 "-threads", str(threads)]
        base = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-i", MEZZ, "-an", "-c:v", "libx264"] + extra + \
               ["-b:v", f"{target}k", "-pix_fmt", "yuv420p",
                "-g", "60", "-keyint_min", "60", "-frames:v", str(N)]
        failed = None
        for pnum in (1, 2):
            cmd = base + ["-pass", str(pnum), "-passlogfile", passlog]
            cmd += ["-f", "null", "/dev/null"] if pnum == 1 else [out]
            log(f"x264 {target}k pass {pnum} (threads={threads}) ...")
            rc, msg = run(cmd, timeout=5400)
            if rc != 0:
                failed = (pnum, rc)
                break
        if failed is None:
            break
        # rc=-9 = OOM kill.  x264's per-frame-thread lookahead buffers are the
        # largest thing it allocates at 4K; dropping to a single thread costs
        # wall clock only, not RD, and is the difference between finishing and
        # dying on this 2 GB box.
        log(f"x264 {target}k pass {failed[0]} rc={failed[1]}; "
            f"{'retrying with threads=1' if attempt == 0 else 'giving up'}")
        threads = 1
        if attempt == 1:
            raise RuntimeError(f"x264 {target}k failed twice: {failed}")
    enc_s = time.time() - t0
    log(f"x264 {target}k encoded in {enc_s/60:.1f} min, measuring ...")
    nbytes = os.path.getsize(out)
    psnrs, ssim_vals = [], []
    n = 0
    # Reference decodes via cv2 (light h264 file) so we never run two live
    # 4K ffmpeg pipes at once -- that OOM-killed processes on this 2 GB box.
    import cv2
    cap = cv2.VideoCapture(out)
    for bgr in bgr_frames(MEZZ):
        ok, ref = cap.read()
        if not ok:
            break
        psnrs.append(float(psnr(bgr, ref)))
        if n % SSIM_EVERY == 0:
            ssim_vals.append([n, float(ssim(bgr, ref))])
        if n % 100 == 0:
            log(f"x264 {target}k metrics {n}/{N}  psnr={np.mean(psnrs):.2f}")
        n += 1
        del bgr, ref
    cap.release()
    dur = n / FPS
    res = dict(codec=f"x264 slow (target {target}k)", target_kbps=target,
               nframes=n, bytes=nbytes,
               kbps=8.0 * nbytes / dur / 1000.0,
               bpp=8.0 * nbytes / (n * H * W * 3),
               psnr=float(np.mean(psnrs)), psnr_min=float(np.min(psnrs)),
               ssim=float(np.mean([v for _, v in ssim_vals])),
               ssim_every=SSIM_EVERY, enc_s=enc_s,
               psnr_trace=psnrs, ssim_samples=ssim_vals)
    save(res, key)
    os.remove(out)
    return res


# --------------------------------------------------------------------------- #
def main():
    log(f"=== 50 s real 4K benchmark: {W}x{H} @ {FPS:.3f} fps, {N} frames ===")
    make_mezz()
    photon = [run_photon_preset(p) for p in PHOTON_PRESETS]
    validate_atto_decoder(photon[0])
    x264 = [run_x264(t) for t in X264_TARGETS]
    table = {"source": "Nikon D5300 4K nature (CC BY 3.0, Wikimedia Commons)",
             "w": W, "h": H, "fps": FPS, "nframes": photon[0]["nframes"],
             "results": [ {k: v for k, v in r.items()
                           if k not in ("psnr_trace", "byte_trace",
                                        "ssim_samples", "md5_encoder")}
                          for r in photon + x264 ]}
    save(table, "real4k_table.json")
    log("=== DONE ===")
    for r in table["results"]:
        print(f"  {r['codec']:<28} {r['kbps']:8.1f} kbps  "
              f"{r['psnr']:6.2f} dB  ssim {r['ssim']:.4f}", flush=True)


if __name__ == "__main__":
    main()
