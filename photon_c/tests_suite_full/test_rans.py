import sys; sys.path.insert(0,'/home/user/photon')
import random, math
from photon.rans import *
fails=[]
def check(n,c,x=""):
    print(("  PASS  " if c else "  FAIL  ")+n+(("  "+x) if x else ""))
    if not c: fails.append(n)

print("[1] FreqTable quantisation invariants")
ok=True; worst=0
for trial in range(6000):
    nsym=random.randint(1,400)
    counts={s:random.randint(1,10**random.randint(1,7)) for s in range(nsym)}
    r=random.random()
    if r<0.25: counts[0]=sum(counts.values())*1000      # extreme skew
    elif r<0.35: counts={0:10**7,1:1}                    # 1-in-10^7 rare symbol
    elif r<0.40: counts={s:1 for s in range(nsym)}       # perfectly uniform
    t=FreqTable(counts)
    if sum(t.freq.values())!=PROB_SCALE or min(t.freq.values())<1: ok=False;print("  invariant",trial);break
    if t.cum[t.symbols[-1]]+t.freq[t.symbols[-1]]!=PROB_SCALE: ok=False;print("  cum",trial);break
    d,_=FreqTable.deserialize(bytes(t.serialize()))
    if (d.freq,d.cum,d.symbols)!=(t.freq,t.cum,t.symbols): ok=False;print("  ser/deser",trial);break
check("6000 random tables: sum==M, min>=1, cum contiguous, serialize roundtrip", ok)

print("[2] rANS exact roundtrip, randomised")
ok=True
for trial in range(400):
    n=random.randint(0,6000)
    nsym=random.randint(1,80)
    p=[random.random()**random.choice([1,1,3,8,20]) for _ in range(nsym)]
    tot=sum(p); p=[x/tot for x in p]
    syms=[]
    for _ in range(n):
        r=random.random(); acc=0
        for k,pk in enumerate(p):
            acc+=pk
            if r<acc: syms.append(k); break
        else: syms.append(nsym-1)
    blob=pack(syms); got,used=unpack(blob)
    if got!=syms: ok=False;print("  MISMATCH trial",trial,"n",n);break
    if used!=len(blob): ok=False;print("  consumed",used,"of",len(blob));break
check("400 random sources (n up to 6000, heavy skew), exact roundtrip", ok)

print("[3] adversarial distributions")
for name,syms in {
 "single symbol x5000":[7]*5000,
 "99.9% / 0.1% two symbols":[0]*9990+[1]*10,
 "one rare symbol in 100k":[0]*99999+[1],
 "uniform 256 symbols":list(range(256))*20,
 "geometric run lengths":[min(int(-math.log(random.random())*3),50) for _ in range(5000)],
 "empty":[],
 "20k zeros then five ones":[0]*20000+[1]*5,
 "alternating":[i%2 for i in range(10000)],
 "sorted ascending":sorted(random.randint(0,50) for _ in range(8000)),
}.items():
    blob=pack(syms); got,_=unpack(blob)
    extra=f"{len(blob)} B total"
    if syms and len(set(syms))>1:
        t=FreqTable({s:syms.count(s) for s in set(syms)})
        payload_only=len(blob)-19-len(t.symbols)*5     # strip header+table
        ideal=ideal_bits(syms,t)/8
        extra+=f", payload {payload_only} B vs Shannon ideal {ideal:.0f} B ({payload_only/max(ideal,1e-9):.3f}x)"
    check(name, got==syms, extra)

print("[4] the number that matters: skip-map cost for a static 4K frame")
rng=random.Random(5)
for rate,label in ((0.80,"busy scene"),(0.93,"typical streaming"),(0.99,"talking head / CCTV"),(0.999,"near-static")):
    nblk=(3840//16)*(2160//16)          # 32400 blocks of 16x16 at 4K
    skip=[1 if rng.random()<rate else 0 for _ in range(nblk)]
    blob=pack(skip); got,_=unpack(blob)
    t=FreqTable({0:skip.count(0),1:skip.count(1)})
    H=-(rate*math.log2(rate)+(1-rate)*math.log2(1-rate))
    print(f"   {label:22s} {100*rate:5.1f}% skip -> {len(blob):6d} bytes = {8*len(blob)/nblk:.4f} bits/block "
          f"(Shannon {H:.4f})  [{'exact' if got==skip else 'MISMATCH'}]")
    if got!=skip: fails.append("skipmap "+label)
check("all skip maps exact", not any(f.startswith("skipmap") for f in fails))
check("99.9%-skip 4K map under 200 bytes", len(pack([1 if rng.random()<0.999 else 0 for _ in range(32400)]))<200)
print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
