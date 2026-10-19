"""4K reference RD points.  Reports ACTUAL rate, since on this box the crippled
memory-safe settings overshoot their targets by 1.2x-3.4x."""
import sys, os, time, gc, json, resource, tempfile, shutil
sys.path.insert(0,'/home/user/photon')
import benchlib as B
os.environ.setdefault("TMPDIR","/home/user/photon/out/tmp"); os.makedirs(os.environ["TMPDIR"],exist_ok=True)
RES="/home/user/photon/results"; WORK="/home/user/photon/out/work"
H,W,NF,FPS=2160,3840,48,30.0
SRC=os.path.join(WORK,f"src4k_{NF}.yuv")
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
names=sys.argv[1:] or ["libx264","libx265"]
TARGETS=[40.0,70.0,110.0,160.0,240.0,340.0,500.0]
src=B.YuvSource(SRC,W,H,NF,FPS)
for name in names:
    rows=[]
    print(f"=== {name} @ 4K, {NF} frames ===",flush=True)
    for tk in TARGETS:
        tmp=tempfile.mkdtemp(prefix="ref_")
        t0=time.time()
        try:
            info=B.encode_reference(name,SRC,W,H,NF,FPS,tk,tmp)
            r=B.measure_reference(name,src,info,tmp)
            r["target_kbps"]=tk
            rows.append(r)
            print(f"  target {tk:6.0f}k -> actual {r['kbps']:8.1f} kbps  PSNR {r['psnr']:5.2f} (min {r['psnr_min']:5.2f})  SSIM {r['ssim']:.4f}  bpp {r['bpp']:.6f}  enc {r['enc_fps']:.2f} f/s  [{time.time()-t0:.0f}s]",flush=True)
        except Exception as e:
            print(f"  target {tk:6.0f}k FAILED {type(e).__name__}: {str(e)[:200]}",flush=True)
        shutil.rmtree(tmp,ignore_errors=True); gc.collect()
    json.dump(rows,open(os.path.join(RES,f"ref_4k_{name}.json"),"w"),indent=2,default=str)
    print(f"  saved results/ref_4k_{name}.json  peak RSS {rss():.0f} MB",flush=True)
