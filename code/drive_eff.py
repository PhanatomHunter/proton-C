#!/usr/bin/env python3
"""Efficiency / compute audit of PHOTON-C vs x264 on the real 50 s 4K clip.

Measures, per codec pass, on the identical 1500 4K frames:
  * wall seconds and fps
  * CPU seconds (utime+stime of self, or RUSAGE_CHILDREN for ffmpeg)
  * peak RSS (VmHWM for self; ru_maxrss per child, proc-isolated)
  * output bytes

No hardware power meter exists in this sandbox (no RAPL, no powertop/perf),
so energy is reported as an *estimate from CPU-time* with the assumption
printed next to it, and a 5/10/15 W sensitivity row — the honest way to talk
about "power" without a watt-meter.  CPU-time is the quantity hardware
actually charges you for on a 2-core box.

Every stage writes into results/eff4k.json atomically, so the run is
resumable (re-running just skips finished stages).
"""
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
os.environ["TMPDIR"] = OUT

import benchlib
from benchlib import ffmpeg_exe
from photon.codec import preset
from photon.frame import Encoder, Decoder

# Installed packages do not survive between conversation turns here.  If the
# imageio-ffmpeg binary is unavailable, try setup_env.sh first.
if ffmpeg_exe() is None:
    _ff = "/usr/local/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    if os.path.exists(_ff):
        benchlib.FFMPEG = _ff
    else:
        raise SystemExit("ffmpeg missing -- run: sh setup_env.sh")

W, H = 3840, 2160
FPS = 30000.0 / 1001.0
N = int(os.environ.get("PHOTON_EFF_N", 1500))
SRC = os.path.join(OUT, "nikon4k_50s.webm")
SRC_URL = ("https://upload.wikimedia.org/wikipedia/commons/4/46/"
           "4K_Nature_Cinematography_recorded_with_Nikon_D5300.webm")
MEZZ = os.path.join(OUT, "mezz_4k50.mkv" if N == 1500 else f"mezz_4k50_smoke{N}.mkv")
BLOB = os.path.join(HERE, "results", "atto_4k50.phtn")   # persists (workspace)
RESJ = os.path.join(HERE, "results", "eff4k.json")
P_ASSUME_W = 10.0   # W, sustained package of a 2-core mobile-class CPU at full tilt


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    try:
        with open(RESJ) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save(d):
    tmp = RESJ + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, RESJ)
    log("saved results/eff4k.json")


def vmhwm_gb() -> float:
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM"):
            return float(ln.split()[1]) / 1048576.0
    return 0.0


def bgr_frames(path, nmax=N):
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


def mezz():
    if os.path.exists(MEZZ):
        return
    # Big downloads (>128 MB snapshot cap) do not survive between turns; fetch.
    if not os.path.exists(SRC):
        raw = os.path.join(OUT, "nikon4k_full.webm")
        log("re-downloading 4K source from Wikimedia (~144 MB) ...")
        r = subprocess.run(["curl", "-sL", "--max-time", "900", "-o", raw, SRC_URL])
        if r.returncode != 0 or not os.path.exists(raw):
            raise RuntimeError("source download failed")
        rc, msg = benchlib.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                                "-y", "-ss", "10", "-t", "50.05", "-i", raw,
                                "-c", "copy", SRC], timeout=600)
        os.remove(raw)
        if rc != 0:
            raise RuntimeError(msg[-400:])
    log("transcoding mezzanine (out/ does not persist between turns) ...")
    rc, msg = benchlib.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                            "-i", SRC, "-frames:v", str(N),
                            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                            "-x264-params", "ref=1", "-an", MEZZ], timeout=3600)
    if rc != 0:
        raise RuntimeError(msg[-400:])


def enc_photon(d):
    key = "photon_enc_atto"
    if key in d:
        log("photon enc: checkpoint exists, skipping"); return
    log("PHOTON encode atto, 1500 frames, no metrics (pure pipeline) ...")
    cfg = preset("photon-atto", W, H, FPS)
    enc = Encoder(cfg)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = r0.ru_utime + r0.ru_stime
    t0 = time.time()
    n = 0
    for bgr in bgr_frames(MEZZ):
        enc.push_frame(bgr[:, :, ::-1])
        n += 1
        if n % 200 == 0:
            log(f"  enc {n}/{N}  rss {vmhwm_gb():.2f} GB")
        del bgr
    blob = enc.finish()
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = r1.ru_utime + r1.ru_stime - cpu0
    with open(BLOB, "wb") as fh:
        fh.write(blob)
    dur = n / FPS
    d[key] = dict(
        frames=n, wall_s=round(wall, 1), cpu_s=round(cpu, 1),
        fps_wall=round(n / wall, 3),
        cpu_s_per_video_s=round(cpu / dur, 3),
        peak_rss_gb=round(vmhwm_gb(), 3),
        bytes=len(blob), kbps=round(8.0 * len(blob) / dur / 1000.0, 1),
        note="includes cv2 reading the lossless mezzanine (source ingest); "
             "kbps should match the real4k benchmark exactly (deterministic)")
    save(d)
    log(f"PHOTON enc: {n/wall:.2f} f/s wall, {cpu:.0f}s CPU, "
        f"{vmhwm_gb():.2f} GB peak, {8.0*len(blob)/dur/1000:.1f} kbps")


def dec_photon(d):
    key = "photon_dec_atto"
    if key in d:
        log("photon dec: checkpoint exists, skipping"); return
    log("PHOTON decode atto blob, 1500 frames ...")
    with open(BLOB, "rb") as fh:
        dec = Decoder(fh.read())
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = r0.ru_utime + r0.ru_stime
    t0 = time.time()
    for i in range(dec.nframes):
        fr = dec.decode_frame(i)
        if i % 300 == 0:
            log(f"  dec {i}/{dec.nframes}")
        del fr
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = r1.ru_utime + r1.ru_stime - cpu0
    dur = dec.nframes / FPS
    d[key] = dict(frames=dec.nframes, wall_s=round(wall, 1), cpu_s=round(cpu, 1),
                  fps_wall=round(dec.nframes / wall, 3),
                  cpu_s_per_video_s=round(cpu / dur, 3),
                  peak_rss_gb=round(vmhwm_gb(), 3))
    save(d)
    log(f"PHOTON dec: {dec.nframes/wall:.2f} f/s wall, {cpu:.0f}s CPU, "
        f"{vmhwm_gb():.2f} GB peak")


def ffmpeg_child(cmd):
    """Run an ffmpeg command, return (rc, wall, cpu_child, peak_rss_child_gb)."""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu0 = r0.ru_utime + r0.ru_stime
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    wall = time.time() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (p.returncode, wall, r1.ru_utime + r1.ru_stime - cpu0,
            r1.ru_maxrss / 1048576.0)


def enc_x264(d):
    for threads in (2, 1):
        key = f"x264_enc_200k_1pass_t{threads}"
        if key in d:
            log(f"x264 enc t{threads}: checkpoint exists, skipping"); continue
        out = os.path.join(OUT, f"eff_x264_t{threads}.mkv")
        log(f"x264 single-pass 200k, threads={threads}, 1500 4K frames ...")
        rc, wall, cpu, rss = ffmpeg_child([
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", MEZZ, "-an", "-c:v", "libx264", "-preset", "slow",
            "-x264-params", "ref=2:rc-lookahead=10:bframes=2",
            "-threads", str(threads), "-b:v", "200k", "-pix_fmt", "yuv420p",
            "-g", "60", "-keyint_min", "60", "-frames:v", str(N),
            "-f", "matroska", out])
        if rc != 0:
            raise RuntimeError(f"x264 t{threads} rc={rc}")
        nbytes = os.path.getsize(out)
        dur = N / FPS
        d[key] = dict(frames=N, wall_s=round(wall, 1), cpu_s=round(cpu, 1),
                      fps_wall=round(N / wall, 3),
                      cpu_s_per_video_s=round(cpu / dur, 3),
                      peak_rss_gb=round(rss, 3), bytes=nbytes,
                      kbps=round(8.0 * nbytes / dur / 1000.0, 1),
                      note="single-pass ABR here: measures encode compute; the "
                           "quality table uses two-pass (same compute x2)")
        save(d)
        log(f"x264 enc t{threads}: {N/wall:.2f} f/s wall, {cpu:.0f}s CPU, {rss:.2f} GB")


def dec_x264(d):
    key = "x264_dec_200k"
    if key in d:
        log("x264 dec: checkpoint exists, skipping"); return
    src = os.path.join(OUT, "eff_x264_t2.mkv")
    log("x264 decode 4K 200k file, 1500 frames ...")
    rc, wall, cpu, rss = ffmpeg_child([
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-frames:v", str(N), "-f", "null", "/dev/null"])
    if rc != 0:
        raise RuntimeError(f"x264 dec rc={rc}")
    dur = N / FPS
    d[key] = dict(frames=N, wall_s=round(wall, 1), cpu_s=round(cpu, 1),
                  fps_wall=round(N / wall, 3),
                  cpu_s_per_video_s=round(cpu / dur, 3),
                  peak_rss_gb=round(rss, 3))
    save(d)
    log(f"x264 dec: {N/wall:.2f} f/s wall, {cpu:.0f}s CPU, {rss:.2f} GB")


def summarize(d):
    dur = N / FPS
    e = {}
    for k, v in d.items():
        if not isinstance(v, dict) or "cpu_s_per_video_s" not in v:
            continue
        cpu_min = v["cpu_s_per_video_s"] * 60.0
        e[k] = dict(
            cpu_minutes_per_video_minute=round(cpu_min, 2),
            est_joules_per_video_minute_at_10W=round(cpu_min * 60 * P_ASSUME_W, 0),
            est_Wh_per_video_hour_at_10W=round(cpu_min * P_ASSUME_W, 1),
            sensitivity_Wh_per_video_hour=dict(
                [("5W", round(cpu_min * 5, 1)), ("10W", round(cpu_min * 10, 1)),
                 ("15W", round(cpu_min * 15, 1))]))
    d["_energy_model"] = dict(
        basis="No hardware power counter exists in this sandbox (no RAPL, no "
              "powertop/perf). Energy = CPU-seconds x assumed sustained package "
              "power; the 5/15 W bracket covers phone-SoC to hot laptop. "
              "CPU-time itself is exact and is the real number to compare.",
        assumed_watts=P_ASSUME_W, video="1500 f 4K29.97 = 50.05 s",
        summary=e)
    save(d)
    log("summary written")


def main():
    d = load()
    mezz()
    enc_photon(d); dec_photon(d); enc_x264(d); dec_x264(d); summarize(d)
    log("=== efficiency audit done ===")


if __name__ == "__main__":
    main()
