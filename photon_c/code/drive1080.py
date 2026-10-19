"""1080p benchmark on REAL footage (Big Buck Bunny), all codecs at full presets.

This is the table to trust for codec-vs-codec efficiency: real content, enough
frames for every rate controller to converge (90 available, 60 used), identical
GOP, and each reference at its own good settings rather than a memory-crippled
one.  At 4K this box cannot run x265 or AV1 at all, so the 4K table is narrower.
"""
import sys, os, time, gc, json, tempfile, shutil, resource
sys.path.insert(0,'/home/user/photon')
import numpy as np
import benchlib as B
from photon.codec import Config
from photon.frame import luma_shapes

W,H,NF,FPS = 1920,1080,60,30.0
SRC="/home/user/photon/out/work/src1080.yuv"
RES="/home/user/photon/results"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
src=B.YuvSource(SRC,W,H,NF,FPS)
stage=sys.argv[1] if len(sys.argv)>1 else "all"

if stage in ("all","photon"):
    rows=[]
    print(f"=== PHOTON-C {W}x{H} {NF} real frames ===",flush=True)
    print(" bms  cms tx  qp nlev base        bytes/fr     kbps       bpp   PSNR PSNRmin  SSIM  enc f/s dec f/s",flush=True)
    for c in [dict(bms=48,cms=48,tx=0,qp=60),dict(bms=64,cms=64,tx=0,qp=66),
              dict(bms=96,cms=96,tx=0,qp=70),dict(bms=136,cms=96,tx=0,qp=74),
              dict(bms=136,cms=136,tx=1,qp=74),dict(bms=270,cms=136,tx=1,qp=78),
              dict(bms=540,cms=270,tx=2,qp=80)]:
        cfg=Config(width=W,height=H,fps=FPS,gop=60,tx_depth=c['tx'],qp=c['qp'],
                   base_min_side=c['bms'],chroma_min_side=c['cms'])
        t0=time.time()
        try: r=B.run_photon(src,cfg)
        except Exception as e:
            print(f" {c['bms']:4d} FAILED {type(e).__name__}: {str(e)[:150]}",flush=True); continue
        blob=r.pop('blob'); r.update(bytes_frame=len(blob)/NF,
            label=f"photon bms{c['bms']} tx{c['tx']} qp{c['qp']}")
        rows.append(r)
        print(f" {c['bms']:4d} {c['cms']:4d} {c['tx']:2d} {c['qp']:3d} {r['n_levels']:5d} {r['base_shape']:11s}"
              f" {r['bytes_frame']:9.0f} {r['kbps']:9.1f} {r['bpp']:9.5f} {r['psnr']:6.2f} {r['psnr_min']:7.2f}"
              f" {r['ssim']:6.4f} {r['enc_fps']:7.2f} {r['dec_fps']:7.2f}  [{time.time()-t0:.0f}s]",flush=True)
        del blob,r; gc.collect()
    json.dump(rows,open(f"{RES}/photon_1080p.json","w"),indent=2,default=str)
    print("saved results/photon_1080p.json",flush=True)

for name in ([stage] if stage not in ("all","photon") else ["libx264-full","libx265-full"]):
    rows=[]
    print(f"\n=== {name} {W}x{H} {NF} real frames ===",flush=True)
    for tk in (100.0,200.0,400.0,800.0):
        tmp=tempfile.mkdtemp(prefix="r10_"); t0=time.time()
        try:
            info=B.encode_reference(name,SRC,W,H,NF,FPS,tk,tmp)
            r=B.measure_reference(name,src,info,tmp)
            r["target_kbps"]=tk; rows.append(r)
            print(f"  target {tk:6.0f}k -> actual {r['kbps']:8.1f} ({r['kbps']/tk:.2f}x)  PSNR {r['psnr']:5.2f} "
                  f"(min {r['psnr_min']:5.2f})  SSIM {r['ssim']:.4f}  bpp {r['bpp']:.6f}  enc {r['enc_fps']:.2f} f/s  [{time.time()-t0:.0f}s RSS {rss():.0f}MB]",flush=True)
        except Exception as e:
            print(f"  target {tk:6.0f}k FAILED {type(e).__name__}: {str(e)[:150]}  [{time.time()-t0:.0f}s]",flush=True)
        shutil.rmtree(tmp,ignore_errors=True); gc.collect()
    json.dump(rows,open(f"{RES}/ref_1080p_{name}.json","w"),indent=2,default=str)
    print(f"saved results/ref_1080p_{name}.json",flush=True)
