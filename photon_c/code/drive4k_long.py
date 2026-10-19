"""
Final controlled 4K experiment, designed against the failure modes of the quick
runs:

* 24 frames.  The 8-frame runs were dominated by a single keyframe, so rate
  control never converged -- x265 ignoring a 350k target and producing 10949
  kb/s is what that looks like.  With a 60-frame GOP, 24 frames is still
  keyframe-heavy, but the steady state (as reported by the per-frame sizes) is
  visible in both codecs.

* One codec at a time, kid-glove settings, and memory checked inside the loop.
  x264's default slow at 4K has *sometimes* survived in this sandbox (~1.3 GB)
  and sometimes been SIGKILLed; an OOM shows up as rc=-9 with no output, so the
  run records it and moves on rather than killing the whole experiment.

* Report actual measured kbps from the produced file, never the target.  At
  these rates a codec can be an order of magnitude off its target and that fact
  is a result, not a bug in the harness.
"""
import sys, os, time, gc, json, tempfile, shutil, resource
sys.path.insert(0, '/home/user/photon')
import benchlib as B
from photon.codec import Config

RES  = "/home/user/photon/results"
WORK = "/home/user/photon/out/work"
H, W = 2160, 3840
NF   = int(os.environ.get("NF", "24"))
FPS  = 30.0
SRC  = os.path.join(WORK, f"src4k_{NF}.yuv")
os.makedirs(RES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

# ---- 1. source ---------------------------------------------------------- #
if not (os.path.exists(SRC) and os.path.getsize(SRC) == NF*H*W*3//2):
    t0 = time.time()
    B.make_source_yuv(SRC, NF, W, H, FPS, kind="synthetic4k")
    print(f"[source] wrote {SRC} ({os.path.getsize(SRC)/1e6:.0f} MB) in {time.time()-t0:.1f}s", flush=True)
else:
    print(f"[source] reuse {SRC}", flush=True)
src = B.YuvSource(SRC, W, H, NF, FPS)

# ---- 2. PHOTON-C  (same settings as before so tables stay comparable) ----- #
photon_cfgs = [
    ("atto48",  Config(width=W,height=H,fps=FPS,gop=60,tx_depth=0,qp=66, base_min_side=48, chroma_min_side=48)),
    ("atto96",  Config(width=W,height=H,fps=FPS,gop=60,tx_depth=0,qp=78, base_min_side=96, chroma_min_side=96)),
    ("nano136", Config(width=W,height=H,fps=FPS,gop=60,tx_depth=0,qp=78, base_min_side=136,chroma_min_side=136)),
    ("lean136", Config(width=W,height=H,fps=FPS,gop=60,tx_depth=1,qp=76, base_min_side=136,chroma_min_side=96)),
]
rows = []
for lbl, cfg in photon_cfgs:
    t0 = time.time()
    try:
        r = B.run_photon(src, cfg)
        blob = r.pop("blob")
        r.update(label="photon-"+lbl, bytes_frame=len(blob)/NF)
        rows.append(r)
        print(f"[photon-{lbl}] {r['kbps']:9.1f} kbps  PSNR {r['psnr']:5.2f} (min {r['psnr_min']:.2f})  "
              f"SSIM {r['ssim']:.4f}  bpp {r['bpp']:.6f}  enc {r['enc_fps']:.2f} f/s dec {r['dec_fps']:.2f} f/s  "
              f"[{time.time()-t0:.0f}s rss {rss():.0f}MB]", flush=True)
        del blob
    except Exception as e:
        print(f"[photon-{lbl}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    gc.collect()
    json.dump(rows, open(f"{RES}/long4k_photon.json","w"), indent=2, default=str)

# ---- 3. references, targets measured against what they really deliver ----- #
REF = {
    "libx264": dict(codec="libx264", extra=["-preset","slow",
                     "-x264-params","keyint=60:min-keyint=60"]),
    "libx265": dict(codec="libx265", extra=["-preset","medium",
                     "-x265-params","pools=1:frame-threads=1:ref=2:rc-lookahead=10:"
                                    "bframes=3:keyint=60:min-keyint=60:log-level=error"]),
    "libaom-av1": dict(codec="libaom-av1", extra=["-cpu-used","8","-row-mt","1",
                                                   "-tiles","2x2","-g","60"]),
}
ff = B.ffmpeg_exe()
ref_rows = []
for name, spec in REF.items():
    for tk in (300.0, 500.0, 800.0):
        tmp = tempfile.mkdtemp(prefix="l4k_")
        out = os.path.join(tmp, f"{name}_{int(tk)}.mkv")
        passlog = os.path.join(tmp, name)
        t0 = time.time()
        try:
            base = [ff,"-hide_banner","-loglevel","error","-y",
                    "-f","rawvideo","-pix_fmt","yuv420p","-s",f"{W}x{H}",
                    "-r",str(FPS),"-i",SRC]
            for pn in (1,2):
                cmd = base + ["-c:v",spec["codec"]] + spec["extra"] + [
                        "-b:v",f"{int(tk)}k","-pass",str(pn),"-passlogfile",passlog,
                        "-pix_fmt","yuv420p"]
                cmd += ["-f","null","/dev/null"] if pn==1 else ["-an",out]
                rc, msg = B.run(cmd, timeout=1200)
                if rc != 0:
                    raise RuntimeError(f"pass {pn} rc={rc}: {msg[-400:]}")
            nbytes = os.path.getsize(out)
            actual = 8.0*nbytes/(NF/FPS)/1000.0
            dec_yuv = os.path.join(tmp, "dec.yuv")
            B.decode_to_yuv(out, dec_yuv)
            dec = B.YuvSource(dec_yuv, W, H, NF)
            acc = B.MetricAcc()
            for i in range(NF): acc.add(src[i], dec[i])
            q = acc.result()
            r = dict(codec=name, target_kbps=tk, kbps=actual, bytes=nbytes,
                     bpp=8.0*nbytes/(NF*W*H*3), psnr=q["psnr"], psnr_min=q["psnr_min"],
                     ssim=q["ssim"], enc_s=time.time()-t0)
            ref_rows.append(r)
            print(f"[{name} @{tk:.0f}k] actual {actual:9.1f} kbps  PSNR {q['psnr']:5.2f} "
                  f"(min {q['psnr_min']:.2f})  SSIM {q['ssim']:.4f}  [{time.time()-t0:.0f}s rss {rss():.0f}MB]",
                  flush=True)
        except Exception as e:
            print(f"[{name} @{tk:.0f}k] FAILED {type(e).__name__}: {str(e)[:200]}  [{time.time()-t0:.0f}s]",
                  flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()
        json.dump(ref_rows, open(f"{RES}/long4k_refs.json","w"), indent=2, default=str)

print("4K LONG EXPERIMENT DONE", flush=True)
