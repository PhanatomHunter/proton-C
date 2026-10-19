#!/usr/bin/env python3
"""Plots + report for the 50 s real 4K test (results/real4k_*.json)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FPS = 30000.0 / 1001.0

PH_COLORS = {"photon-atto": "#22d3ee", "photon-femto": "#34d399",
             "photon-nano": "#a3e635", "photon-lean": "#fbbf24"}


def load():
    rows, traces = [], {}
    for p in sorted(glob.glob(os.path.join(RES, "real4k_*.json"))):
        d = json.load(open(p))
        n = os.path.basename(p)
        if "decoder_check" in n or "table" in n:
            continue
        rows.append(d)
        if "psnr_trace" in d:
            traces[d["preset"] if "preset" in d else f"x264-{d['target_kbps']}"] = d
    return rows, traces


def fig_rd(rows):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.6), dpi=130)
    fig.patch.set_facecolor("#0b1220"); ax.set_facecolor("#0b1220")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("#0b1220")
    ax.tick_params(colors="#9aa7bd"); ax.xaxis.label.set_color("#cdd7e5")
    ax.yaxis.label.set_color("#cdd7e5"); ax.title.set_color("#e8eef7")
    ph = [r for r in rows if r["codec"].startswith("photon")]
    ph.sort(key=lambda r: r["kbps"])
    xs = [r["kbps"] for r in ph]; ys = [r["psnr"] for r in ph]
    ax.plot(xs, ys, "-o", color="#38bdf8", lw=2.2, ms=7,
            label="PHOTON-C (synthesis codec)")
    for r in ph:
        ax.annotate(r["preset"].replace("photon-", ""), (r["kbps"], r["psnr"]),
                    textcoords="offset points", xytext=(6, -14),
                    color=PH_COLORS.get(r["preset"], "#93c5fd"), fontsize=9)
    x2 = [r for r in rows if r["codec"].startswith("x264")]
    x2.sort(key=lambda r: r["kbps"])
    ax.plot([r["kbps"] for r in x2], [r["psnr"] for r in x2], "-s",
            color="#f87171", lw=2, ms=6, label="x264 slow (two-pass ABR)")
    for r in x2:
        ax.annotate(f"{r['target_kbps']}k→{r['kbps']:.0f}",
                    (r["kbps"], r["psnr"]), textcoords="offset points",
                    xytext=(8, 6), color="#fca5a5", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("measured bitrate (kbps, log scale) — 4K 3840×2160, 50 s, H.264-decode-free")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PHOTON-C vs x264 — REAL 4K, 50 s nature footage (Nikon D5300, CC BY 3.0)")
    ax.grid(True, color="#1e293b", lw=0.6)
    lg = ax.legend(loc="upper left", framealpha=0.2)
    for t in lg.get_texts():
        t.set_color("#dbe4f0")
    out = os.path.join(RES, "real4k_rd.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_traces(traces):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=130, sharex=True)
    fig.patch.set_facecolor("#0b1220")
    for ax in (a1, a2):
        ax.set_facecolor("#0b1220")
        ax.tick_params(colors="#9aa7bd")
        ax.grid(True, color="#1e293b", lw=0.5)
        a = ax.yaxis.label; a.set_color("#cdd7e5")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for k, d in traces.items():
        col = PH_COLORS.get(k, "#f87171")
        ps = np.asarray(d["psnr_trace"])
        m = 25 if len(ps) > 400 else 5
        ker = np.ones(m) / m
        sm = np.convolve(ps, ker, mode="same")
        a1.plot(sm, color=col, lw=1.4, label=k, alpha=0.95)
        if "byte_trace" in d:
            kbps = np.asarray(d["byte_trace"]) * 8.0 * FPS / 1000.0
            ksm = np.convolve(kbps, ker, mode="same")
            a2.plot(ksm, color=col, lw=1.4, label=k, alpha=0.95)
    a1.set_ylabel("PSNR (dB, 25-frame avg)")
    a2.set_ylabel("kbps (25-frame avg)")
    a2.set_xlabel("frame"); a2.xaxis.label.set_color("#cdd7e5")
    a1.set_title("Per-frame traces — 50 s real 4K", color="#e8eef7")
    for ax in (a1, a2):
        lg = ax.legend(loc="lower left", framealpha=0.2, ncol=2, fontsize=8)
        for t in lg.get_texts():
            t.set_color("#dbe4f0")
    out = os.path.join(RES, "real4k_traces.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def main():
    rows, traces = load()
    chk = None
    p = os.path.join(RES, "real4k_atto_decoder_check.json")
    if os.path.exists(p):
        chk = json.load(open(p))
    p1 = fig_rd(rows)
    p2 = fig_traces(traces)
    # console table
    rows.sort(key=lambda r: r["kbps"])
    print(f"\n{'codec':<30}{'kbps':>10}{'PSNR dB':>10}{'SSIM':>8}{'bpp':>10}")
    for r in rows:
        print(f"{r['codec']:<30}{r['kbps']:>10.1f}{r['psnr']:>10.2f}"
              f"{r['ssim']:>8.4f}{r['bpp']:>10.5f}")
    comp = 5971776.0  # raw 4K RGB kbps at 29.97 fps
    best = min(rows, key=lambda r: r["kbps"])
    print(f"\nraw 4K RGB = {comp/1e6:.2f} Gbit/s; atto = {best['kbps']:.1f} kbps "
          f"-> {comp/best['kbps']:,.0f}x smaller at {best['psnr']:.2f} dB")
    if chk:
        print(f"decoder end-to-end check: {chk['md5_mismatches']} mismatched "
              f"frames of {chk['nframes']}, decode {chk['decode_fps']:.2f} f/s")
    print("wrote", p1, p2)


if __name__ == "__main__":
    main()
