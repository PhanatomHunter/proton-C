"""
PHOTON Studio -- interactive demo of the codec, served on 0.0.0.0:3000.

Written against the Python standard library only.  Flask is cleaner, but this
sandbox has no network and the demo has to start first time, every time, so the
whole thing is an http.server with a small routing layer.  Everything the page
needs -- markup, styles, scripts, the RD chart, the architecture diagram -- is
inline or served from this one process, because the preview iframe is sandboxed
and cannot load anything from the outside network.

What the studio does
--------------------
* Lets the visitor encode any of the bundled test clips at any resolution up to
  4K with any preset (or an explicit target bitrate), watches per-frame bitrate
  and PSNR as the job runs, and then plays the decoded result back.  The player
  gets H.264 -- the browser cannot decode a .phtn stream -- re-muxed at CRF 12,
  so what is on screen is the codec's own output, not a dressing-up.
* Shows the honest benchmark position: where PHOTON-C wins, and where it does
  not.  Both curves are in this process, from results/*.json measured by
  benchlib -- served, not hard-coded, so re-running a benchmark updates the page.

One job at a time.  A 4K encode needs most of the box's memory, and two at once
is how the sandbox starts returning OOM-killed processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np  # noqa: E402  -- used by the job worker

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
sys.path.insert(0, PROJ)

from photon.codec import Config, PRESETS, preset          # noqa: E402
from photon.frame import Encoder, Decoder                  # noqa: E402

PORT = 3000
JOBDIR = os.path.join(ROOT, "jobs")
os.makedirs(JOBDIR, exist_ok=True)

SOURCES = {
    "bbb_720":    dict(path="/home/user/testvideo/bbb_720p.mp4",    label="Big Buck Bunny (720p)",  fps=30.0),
    "bbb_1080":   dict(path="/home/user/testvideo/bbb_1080p.mp4",   label="Big Buck Bunny (1080p)", fps=30.0),
    "sintel_1080": dict(path="/home/user/testvideo/sintel_1080p.mp4", label="Sintel (1080p)",       fps=30.0),
}
MAX_FRAMES = [(3840 * 2160, 24), (1920 * 1080, 60), (1280 * 720, 90), (0, 120)]


def max_frames_for(w: int, h: int) -> int:
    """Keep jobs inside memory and nobody waiting three minutes for a demo."""
    px = w * h
    for cap, n in MAX_FRAMES:
        if px >= cap:
            return n
    return MAX_FRAMES[-1][1]


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- #
#  Encode jobs
# --------------------------------------------------------------------------- #
_job_lock = threading.Lock()
_current = None          # the Job currently running
_jobs: dict = {}         # id -> Job


class Job:
    def __init__(self, jid: str, spec: dict):
        self.id = jid
        self.spec = spec
        self.status = "queued"
        self.stage = ""
        self.t0 = None
        self.error = None
        self.frames_done = 0
        self.frames_total = 0
        self.frame_stats = []        # per-frame {bytes, psnr}
        self.result = None

    def to_dict(self) -> dict:
        return dict(id=self.id, spec=self.spec, status=self.status, stage=self.stage,
                    error=self.error, frames_done=self.frames_done,
                    frames_total=self.frames_total, frame_stats=self.frame_stats,
                    result=self.result,
                    elapsed=(time.time() - self.t0) if self.t0 else 0)


def _reader_frames(path: str, w: int, h: int, limit: int):
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += ["-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsz = w * h * 3
    n = 0
    try:
        while True:
            buf = p.stdout.read(fsz)
            if len(buf) < fsz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            n += 1
            if limit and n >= limit:
                break
    finally:
        p.stdout.close()
        p.wait()


def _psnr(a, b) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else float(10.0 * np.log10(255.0 ** 2 / mse))


def run_job(job: Job) -> None:
    """The whole lifecycle, executed on a worker thread.

    Encode stores its own per-frame stats directly into the job, then decode
    re-derives each frame from the stream and measures it against the source
    before remuxing to H.264 for the player.  Decoded frames are never kept in
    memory beyond the one being written -- a 4K frame is 25 MB, and holding 24 of
    them is the difference between a demo and an OOM.
    """
    global _current
    spec = job.spec
    w, h = spec["w"], spec["h"]
    fps = float(SOURCES[spec["source"]]["fps"])
    path = SOURCES[spec["source"]]["path"]
    limit = max_frames_for(w, h)
    job.frames_total = limit
    job.t0 = time.time()

    try:
        job.stage = "encoding"
        if spec.get("preset"):
            cfg = preset(spec["preset"], w, h, fps, target_kbps=spec.get("kbps"))
        else:
            cfg = Config(width=w, height=h, fps=fps,
                         gop=spec.get("gop", 60), qp=spec.get("qp", 70),
                         tx_depth=spec.get("tx_depth", 0),
                         base_min_side=spec.get("bms", 96),
                         chroma_min_side=spec.get("cms", 96),
                         target_kbps=spec.get("kbps"))

        if spec.get("kbps"):
            # exact rate control: walk the preset ladder with two-pass qp search
            # per rung.  The earlier probe-based estimator routinely missed by
            # 2-3x, which is precisely the failure mode a demo UI can't hide.
            from photon.cli import encode_rate_controlled
            make_iter = lambda: _reader_frames(path, w, h, limit)
            blob, enc, rc = encode_rate_controlled(make_iter, w, h, fps,
                                                   spec["kbps"],
                                                   verbose=False)
            qp = enc.cfg.qp
            srcs = limit
            job.frames_done = srcs
        else:
            qp = cfg.qp
            cfg = Config(**{**cfg.__dict__, "qp": qp, "target_kbps": None})
            enc = Encoder(cfg)
            srcs = 0
            for f in _reader_frames(path, w, h, limit):
                enc.push_frame(f)
                srcs += 1
                job.frames_done = srcs
            blob = enc.finish()
        phtn = os.path.join(JOBDIR, f"{job.id}.phtn")
        with open(phtn, "wb") as fh:
            fh.write(blob)

        job.stage = "decoding + measuring"
        job.frames_done = 0
        dec = Decoder(blob)
        n = dec.nframes
        psnrs, sizes = [], []
        mp4 = os.path.join(JOBDIR, f"{job.id}.mp4")
        wp = subprocess.Popen(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "fast", "-crf", "12",
             "-pix_fmt", "yuv420p", mp4],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        first_png = os.path.join(JOBDIR, f"{job.id}_f0.png")
        import cv2
        for i, sf in enumerate(_reader_frames(path, w, h, n)):
            rf = dec.decode_frame(i)
            psnrs.append(_psnr(sf, rf))
            sizes.append(dec.frame_bytes(i))
            job.frame_stats.append(dict(i=i, bytes=dec.frame_bytes(i),
                                        psnr=round(psnrs[-1], 2)))
            wp.stdin.write(rf.tobytes())
            if i == n // 2:
                cv2.imwrite(first_png, rf[:, :, ::-1])
                # The source frame at the same index, for the side-by-side panel:
                # there is no honest way to show a codec's output without its input.
                cv2.imwrite(os.path.join(JOBDIR, f"{job.id}_src.png"), sf[:, :, ::-1])
            job.frames_done = i + 1
        wp.stdin.close(); wp.wait()

        dur = n / fps
        job.result = dict(
            bytes=len(blob), kbps=8.0 * len(blob) / dur / 1000.0,
            bpp=8.0 * len(blob) / (n * w * h * 3),
            raw_mbps=w * h * 3 * 8 * fps / 1e6,
            compression=w * h * 3 * 8 * fps / max(8.0 * len(blob) / dur, 1),
            psnr=float(np.mean(psnrs)), psnr_min=float(np.min(psnrs)),
            qp=qp, nframes=n, w=w, h=h, fps=fps, dur=dur,
            keyframes=int(sum(1 for s in enc.stats if s["keybase"])),
            delta_frames=int(sum(1 for s in enc.stats if not s["keybase"])),
            base=[int(x) for x in enc.shapes[0]],
            enc_fps=n / max((enc.stats and sum(s["enc_time"] for s in enc.stats)) or 1, 1e-9),
            mp4=os.path.getsize(mp4),
        )
        job.stage = "done"
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        if _current is job:
            _current = None


def submit(spec: dict) -> Job:
    global _current
    with _job_lock:
        busy = _current is not None
        jid = f"j{int(time.time()*1000)%10**9}"
        job = Job(jid, spec)
        _jobs[jid] = job
        if not busy:
            _current = job
            job.status = "running"
            threading.Thread(target=run_job, args=(job,), daemon=True).start()
        return job


# --------------------------------------------------------------------------- #
#  Results data (served from the files the benchmarks wrote, kept honest)
# --------------------------------------------------------------------------- #
def load_results() -> dict:
    out = {"4k": {"photon": [], "h264": []}, "1080p": {"photon": [], "h264": [], "h265": [], "av1": []},
           "real4k": {}}
    def j(name):
        try:
            with open(os.path.join(PROJ, "results", name)) as fh:
                return json.load(fh)
        except Exception:
            return []
    eff = j("eff4k.json") or {}
    if isinstance(eff, dict) and eff:
        out["eff4k"] = eff
    tbl = j("real4k_table.json") or {}
    if tbl:
        out["real4k"] = dict(
            source=tbl.get("source", ""), nframes=tbl.get("nframes", 0),
            w=tbl.get("w"), h=tbl.get("h"), fps=round(tbl.get("fps", 0), 3),
            results=[dict(codec=r["codec"], kbps=round(r["kbps"], 1),
                          psnr=round(r["psnr"], 2), ssim=round(r["ssim"], 4),
                          bpp=r.get("bpp", 0)) for r in tbl.get("results", [])])
        chk = j("real4k_atto_decoder_check.json") or {}
        if isinstance(chk, dict):
            out["real4k"]["decoder_check"] = chk
    a = j("long4k_photon.json") or []
    for r in a:
        out["4k"]["photon"].append(dict(kbps=round(r["kbps"], 1), psnr=round(r["psnr"], 2),
                                        ssim=round(r["ssim"], 4), label=r.get("label", "photon")))
    for r in j("long4k_refs.json") or []:
        k = "h264" if r["codec"] == "libx264" else ("h265" if r["codec"] == "libx265" else "av1")
        out["4k"].setdefault(k, []).append(dict(kbps=round(r["kbps"], 1), psnr=round(r["psnr"], 2),
                                                ssim=round(r["ssim"], 4), target=r.get("target_kbps")))
    # short-clip 4K table as a secondary view
    short = []
    for src, c in ((j("photon_4k.json"), "photon"),
                   (j("ref4k.json"), "libx264"),
                   (j("ref_4k_libx264.json"), "libx264")):
        for r in src:
            short.append(dict(codec=c, label=r.get("label", c), kbps=round(r["kbps"], 1),
                              psnr=round(r["psnr"], 2), ssim=round(r.get("ssim") or 0, 4)))
    out["4k_short"] = short
    for name, key in (("photon_1080p.json", "photon"), ("ref_1080p_libx264-full.json", "h264"),
                      ("ref_1080p_libx265-full.json", "h265"), ("ref_1080p_libaom-av1.json", "av1")):
        for r in j(name):
            out["1080p"][key].append(dict(kbps=round(r["kbps"], 1), psnr=round(r["psnr"], 2),
                                          ssim=round(r.get("ssim") or 0, 4)))
    return out


# --------------------------------------------------------------------------- #
#  HTTP
# --------------------------------------------------------------------------- #
INDEX = os.path.join(ROOT, "index.html")
RD_PNG = os.path.join(PROJ, "results", "rd_curves.png")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            self._send(open(INDEX, "rb").read(), "text/html")
        elif p == "/rd.png":
            self._send(open(RD_PNG, "rb").read(), "image/png")
        elif p.startswith("/results/"):
            # benchmark figures produced by results/ generation (png/jpg only)
            name = os.path.basename(p)
            if not name.endswith((".png", ".jpg")) or ".." in name:
                self._json(dict(error="nope"), 404); return
            full = os.path.join(PROJ, "results", name)
            ctype = "image/png" if name.endswith(".png") else "image/jpeg"
            self._send(open(full, "rb").read(), ctype)
        elif p == "/api/config":
            self._json(dict(
                sources={k: dict(label=v["label"], fps=v["fps"]) for k, v in SOURCES.items()},
                presets={k: dict(tx=v[0], qp=v[1], gop=v[2], bms=v[3], cms=v[4])
                         for k, v in PRESETS.items()},
                max_frames=dict((f"{w}", max_frames_for(w, w * 9 // 16)) for w in
                                (3840, 1920, 1280, 640)),
                busy=_current is not None,
            ))
        elif p == "/api/results":
            self._json(load_results())
        elif p == "/api/job":
            q = parse_qs(u.query)
            jid = q.get("id", [""])[0]
            job = _jobs.get(jid)
            if not job:
                self._json(dict(error="no such job"), 404)
            else:
                d = job.to_dict()
                if job is not _jobs.get(last_id()):
                    pass
                self._json(d)
        elif p.startswith("/job/"):
            # /job/<id>.mp4 or /job/<id>.phtn or /job/<id>_f0.png
            fname = os.path.basename(p[len("/job/"):])
            if not all(c.isalnum() or c in "._-" for c in fname):
                self._json(dict(error="bad name"), 400)
                return
            full = os.path.join(JOBDIR, fname)
            if not os.path.exists(full):
                self._json(dict(error="not ready"), 404)
                return
            ctype = ("video/mp4" if fname.endswith(".mp4")
                     else "image/png" if fname.endswith(".png")
                     else "application/octet-stream")
            self._send(open(full, "rb").read(), ctype)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/encode":
            n = int(self.headers.get("Content-Length", 0))
            try:
                spec = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._json(dict(error="bad json"), 400); return
            try:
                spec["w"] = int(spec.get("w", 3840)); spec["h"] = int(spec.get("h", 2160))
                assert spec["source"] in SOURCES
                assert 320 <= spec["w"] <= 3840 and 180 <= spec["h"] <= 2160
                if spec.get("kbps") is not None:
                    spec["kbps"] = min(max(float(spec["kbps"]), 8.0), 50000.0)
                if spec.get("qp") is not None:
                    spec["qp"] = int(min(max(float(spec["qp"]), 30), 97))
                if spec.get("preset") is not None and spec["preset"] not in PRESETS:
                    spec["preset"] = None
            except Exception as e:
                self._json(dict(error=f"bad spec: {e}"), 400); return
            job = submit(spec)
            self._json(dict(id=job.id, queued=job.status == "queued"))
        else:
            self._send(b"not found", "text/plain", 404)


def last_id():
    return max(_jobs, default="")


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.daemon_threads = True
    print(f"PHOTON Studio listening on 0.0.0.0:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
