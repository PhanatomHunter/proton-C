"""AV1 at 1080p on real footage, in the background.  cpu-used 8 to stay inside the
time budget; still a genuine modern reference codec at its normal 1080p settings."""
import sys, os, time, gc, json, tempfile, shutil
sys.path.insert(0,'/home/user/photon')
import benchlib as B
W,H,NF,FPS=1920,1080,60,30.0
SRC="/home/user/photon/out/work/src1080.yuv"
src=B.YuvSource(SRC,W,H,NF,FPS)
rows=[]
name=sys.argv[1] if len(sys.argv)>1 else "libaom-av1"
targets=[float(x) for x in (sys.argv[2:] or ["100","200","400","800"])]
print(f"=== {name} {W}x{H} {NF} real frames ===",flush=True)
for tk in targets:
    tmp=tempfile.mkdtemp(prefix="rav1_"); t0=time.time()
    try:
        info=B.encode_reference(name,SRC,W,H,NF,FPS,tk,tmp)
        r=B.measure_reference(name,src,info,tmp)
        r["target_kbps"]=tk; rows.append(r)
        print(f"  target {tk:6.0f}k -> actual {r['kbps']:8.1f} ({r['kbps']/tk:.2f}x)  PSNR {r['psnr']:5.2f} (min {r['psnr_min']:5.2f})  SSIM {r['ssim']:.4f}  bpp {r['bpp']:.6f}  enc {r['enc_fps']:.2f} f/s  [{time.time()-t0:.0f}s]",flush=True)
        json.dump(rows,open(f"/home/user/photon/results/ref_1080p_{name}.json","w"),indent=2,default=str)
    except Exception as e:
        print(f"  target {tk:6.0f}k FAILED {type(e).__name__}: {str(e)[:200]}  [{time.time()-t0:.0f}s]",flush=True)
    shutil.rmtree(tmp,ignore_errors=True); gc.collect()
print("DONE",flush=True)
