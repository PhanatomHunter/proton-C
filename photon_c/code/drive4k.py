"""Driver: generate the 4K source once to disk, then benchmark each codec against
the very same file.  Run as  python3 drive4k.py <stage>  where stage is one of
source | photon | libx264 | libx265 | libaom-av1."""
import sys, os, time, gc, json, resource, tempfile, shutil
sys.path.insert(0, '/home/user/photon')
import numpy as np
import benchlib as B
from photon.codec import Config

RES   = "/home/user/photon/results"
# Scratch space.  A 48-frame raw 4K I420 file is ~600 MB, so it cannot live in
# /tmp -- in this sandbox /tmp is a 992 MB RAM-backed tmpfs, and filling it both
# fails the write and eats memory the codec needs.  It goes under out/ instead,
# which is on the real disk and is excluded from workspace snapshots.
WORK = "/home/user/photon/out/work"
os.environ.setdefault("TMPDIR", "/home/user/photon/out/tmp")
H, W  = 2160, 3840
NF    = int(os.environ.get("NF", "48"))
FPS   = 30.0
SRC   = os.path.join(WORK, f"src4k_{NF}.yuv")
TARGETS = [160.0, 210.0, 320.0, 430.0, 575.0]   # bracket PHOTON-C measured points
os.makedirs(RES, exist_ok=True); os.makedirs(WORK, exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

stage = sys.argv[1] if len(sys.argv) > 1 else "source"

if stage == "source":
    if os.path.exists(SRC) and os.path.getsize(SRC) == NF*H*W*3//2:
        print(f"source already present: {SRC} ({os.path.getsize(SRC)/1e6:.0f} MB)")
    else:
        t0=time.time()
        B.make_source_yuv(SRC, NF, W, H, FPS, kind="synthetic4k")
        print(f"wrote {SRC}  {os.path.getsize(SRC)/1e6:.0f} MB in {time.time()-t0:.1f}s")
    # sanity: what does the source look like?
    src=B.YuvSource(SRC,W,H,NF,FPS)
    f0=src[0]
    print("frame0", f0.shape, f0.dtype, "mean %.1f std %.1f"%(f0.mean(),f0.std()))
    from photon.transform import rgb_to_yuv
    from photon import synthesis as synth
    from photon.frame import luma_shapes
    y,_,_=rgb_to_yuv(f0)
    for bms in (540,270,96):
        sh=luma_shapes(H,W,bms); base,d=synth.build_pyramid(y,sh)
        import cv2
        cur=base.astype(np.float64)
        for k in range(len(d)): cur=cv2.resize(cur,(sh[k+1][1],sh[k+1][0]),interpolation=cv2.INTER_LANCZOS4)
        print(f"  base-only ceiling 1/{H/sh[0][0]:.1f} (base {sh[0]}): {B.psnr(y,cur):.2f} dB")
        del base,d,cur
    print("RSS %.0f MB"%rss())

elif stage == "photon":
    src=B.YuvSource(SRC,W,H,NF,FPS)
    rows=[]
    cfgs=[
        dict(bms=32, cms=32, tx=0, qp=60, gop=60),
        dict(bms=32, cms=32, tx=0, qp=72, gop=60),
        dict(bms=48, cms=48, tx=0, qp=66, gop=60),
        dict(bms=72, cms=72, tx=0, qp=66, gop=60),
        dict(bms=72, cms=72, tx=0, qp=78, gop=60),
        dict(bms=136,cms=96, tx=1, qp=72, gop=60),
        dict(bms=136,cms=96, tx=0, qp=72, gop=60),
        dict(bms=270,cms=136,tx=1, qp=78, gop=60),
        dict(bms=540,cms=272,tx=2, qp=80, gop=60),
    ]
    print(f"=== PHOTON-C  {W}x{H}  {NF} frames @ {FPS} fps ===", flush=True)
    print(" bms  cms tx  qp nlev base         bytes/fr      kbps        bpp   PSNR  PSNRmin  SSIM   enc f/s  dec f/s", flush=True)
    for c in cfgs:
        cfg=Config(width=W,height=H,fps=FPS,gop=c['gop'],tx_depth=c['tx'],qp=c['qp'],
                   base_min_side=c['bms'],chroma_min_side=c['cms'])
        t0=time.time()
        try:
            r=B.run_photon(src,cfg)
        except Exception as e:
            print(f" {c['bms']:4d} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True); continue
        blob=r.pop('blob')
        r.update(label=f"photon bms{c['bms']} cms{c['cms']} tx{c['tx']} qp{c['qp']}",
                 bytes_frame=len(blob)/NF)
        rows.append(r)
        print(f" {c['bms']:4d} {c['cms']:4d} {c['tx']:2d} {c['qp']:3d} {r['n_levels']:5d} {r['base_shape']:12s}"
              f" {r['bytes_frame']:9.0f} {r['kbps']:9.1f} {r['bpp']:11.6f} {r['psnr']:6.2f} {r['psnr_min']:8.2f}"
              f" {r['ssim']:6.4f} {r['enc_fps']:8.2f} {r['dec_fps']:8.2f}  [{time.time()-t0:.0f}s RSS {rss():.0f}MB]", flush=True)
        del blob, r; gc.collect()
    json.dump(rows,open(os.path.join(RES,"photon_4k.json"),"w"),indent=2,default=str)
    print(f"\nsaved results/photon_4k.json  peak RSS {rss():.0f} MB")

else:
    src=B.YuvSource(SRC,W,H,NF,FPS)
    rows=[]
    print(f"=== {stage}  {W}x{H}  {NF} frames @ {FPS} fps ===", flush=True)
    for tk in TARGETS:
        tmp=tempfile.mkdtemp(prefix="ref_")
        t0=time.time()
        try:
            info=B.encode_reference(stage,SRC,W,H,NF,FPS,tk,tmp)
            r=B.measure_reference(stage,src,info,tmp)
            r["target_kbps"]=tk; r["label"]=f"{stage} @{tk:.0f}k"
            rows.append(r)
            print(f"  target {tk:6.0f}k -> actual {r['kbps']:8.1f} kbps  PSNR {r['psnr']:5.2f} "
                  f"(min {r['psnr_min']:.2f})  SSIM {r['ssim']:.4f}  bpp {r['bpp']:.6f}  "
                  f"enc {r['enc_fps']:.2f} f/s  [{time.time()-t0:.0f}s RSS {rss():.0f}MB]", flush=True)
        except Exception as e:
            print(f"  target {tk:6.0f}k FAILED {type(e).__name__}: {str(e)[:260]}", flush=True)
        shutil.rmtree(tmp,ignore_errors=True); gc.collect()
    json.dump(rows,open(os.path.join(RES,f"ref_4k_{stage}.json"),"w"),indent=2,default=str)
    print(f"saved results/ref_4k_{stage}.json  peak RSS {rss():.0f} MB")
