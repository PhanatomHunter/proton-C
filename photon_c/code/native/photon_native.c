/*  PHOTON-C native kernels — v2: BT.601 full-range colour conversion,
 *  put through an exhaustive proof rather than written on faith.
 *
 *  Bit-exactness contract, verified by tests/test_native.py over EVERY
 *  possible input (2^24 points -- a 4096x4096 image covers all of them, so
 *  the equivalence test is a theorem, not a sample):
 *
 *   inverse (decoder hot path):
 *     y_f32 -> rintf(nearest-even) -> clip[0,255] == np.rint + np.clip
 *     R = sat(Y + LUT_R[Cr])          == cv2 YCrCb2RGB, proven on all (Y,Cr)
 *     G = sat((Y<<13 -2818*(Cb-128) -5849*(Cr-128) +4096)>>13)
 *                                     == cv2 YCrCb2RGB, proven on all (Y,Cr,Cb)
 *     B = sat(Y + LUT_B[Cb])          == cv2 YCrCb2RGB, proven on all (Y,Cb)
 *
 *   The LUTs in luts.h are not derived from any standard's decimal
 *   coefficients: they were extracted from cv2's own observed behaviour
 *   (fit_color2.py) because its 8-bit tables round per-entry and no closed
 *   form reproduces that.  This is the same class of trick production codecs
 *   use to stay interoperable.
 *
 *   forward path (encoder): kept in the Python/cv2 pipeline for now; the
 *   decoder side was the measured hotspot (61 ms/frame at 4K colour convert
 *   alone, of ~176 ms/frame total).
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "luts.h"

static inline uint8_t sat8(int32_t v)
{
    return (uint8_t)((uint32_t)v <= 255u ? v : ~(v >> 31));
}

static inline int32_t f32_to_u8(float x)
{
    /* __builtin_rintf honours the default rounding mode (nearest-even),
     * which is exactly np.rint semantics. */
    float r = __builtin_rintf(x);
    r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
    return (int32_t)r;
}

/* -------- planar Y, Cb, Cr (float32 OR float64) -> interleaved RGB u8 ---
 *
 * Two semantic pitfalls, both caught by md5-checking real streams rather
 * than by unit tests looking plausible:
 *
 *  1. np.rint on the Python path runs on float32 (the planes are cast first);
 *     calling rint() on the original f64 rounds differently at .5 borders.
 *     So the f64 path below must cast to float FIRST, then rintf -- order is
 *     part of the algorithm.
 *
 *  2. Function pointers per element (a "generic" loop over dtypes) defeat
 *     vectorisation and cost more than the LUT lookups; the dtype branch is
 *     hoisted out and each specialization gets its own straight-line loop.
 */
static inline int32_t f32_path(float x)
{
    x = __builtin_rintf(x);
    x = x < 0.f ? 0.f : (x > 255.f ? 255.f : x);
    return (int32_t)x;
}
static inline int32_t f64_path(double x)
{
    /* python chain: f64 plane --(copyto)--> f32 --(np.rint)--> .. order matters */
    float xf = (float)x;
    xf = __builtin_rintf(xf);
    xf = xf < 0.f ? 0.f : (xf > 255.f ? 255.f : xf);
    return (int32_t)xf;
}

#define YUV2RGB_LOOP(TYPE, CONV)                                        \
    static void yuv2rgb_##TYPE(const TYPE* Y, const TYPE* U, const TYPE* V, \
                               uint8_t* rgb, long npix)                 \
    {                                                                   \
        long i;                                                         \
        _Pragma("omp parallel for schedule(static) if(npix > (1<<23))") \
        for (i = 0; i < npix; ++i) {                                    \
            int32_t y  = CONV(Y[i]);                                    \
            int32_t cb = CONV(U[i]);                                    \
            int32_t cr = CONV(V[i]);                                    \
            int32_t g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128); \
            rgb[3*i+0] = sat8(y + LUT_R[cr]);                           \
            rgb[3*i+1] = sat8((g + 4096) >> 13);                        \
            rgb[3*i+2] = sat8(y + LUT_B[cb]);                           \
        }                                                               \
    }
YUV2RGB_LOOP(float, f32_path)
YUV2RGB_LOOP(double, f64_path)

void ptn_yuv2rgb(const void* Y, const void* U, const void* V, int is_f64,
                 uint8_t* rgb, long npix)
{
    if (is_f64)
        yuv2rgb_double((const double*)Y, (const double*)U, (const double*)V, rgb, npix);
    else
        yuv2rgb_float((const float*)Y, (const float*)U, (const float*)V, rgb, npix);
}

/* back-compat entry (older native.py) */
void ptn_convert_grid_ycrcb2rgb(const uint8_t* ycrcb, uint8_t* rgb, long npix)
{
    long i;
    for (i = 0; i < npix; ++i) {
        int32_t y = ycrcb[3*i], cr = ycrcb[3*i+1], cb = ycrcb[3*i+2];
        int32_t g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
        rgb[3*i+0] = sat8(y + LUT_R[cr]);
        rgb[3*i+1] = sat8((g + 4096) >> 13);
        rgb[3*i+2] = sat8(y + LUT_B[cb]);
    }
}

int ptn_num_threads(void)
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* -------------------- v6: fused chroma-2x colour conversion --------------
 *
 * Decode-side fused kernel: luma at full res (W x H, f64), chroma at half
 * res (W/2 x H/2, f64), RGB out.  The chroma 2x upsample uses separable
 * dyadic taps (0.75, 0.25) with edge replication -- cv2 INTER_LINEAR's
 * geometry at exactly 2x -- but computed in plain IEEE f64 in a fixed op
 * order, mirroring the numpy fallback in transform.py:_chroma2x_to_full
 * term-for-term (build disables FMA contraction for that reason).
 *
 * cv2's own f64 linear path differs from these exact taps by <= 5.7e-14
 * (its internal coefficient arithmetic rounds differently); after rint/clip
 * to u8 the two agree except on exact .5 rounding borders -- measure-zero
 * input.  Both encoder and decoder route through this one code path, so
 * they stay locked regardless.
 *
 * Ops eliminated vs the old path: one cv2.resize over two full-res f64
 * chroma planes (~23 ms/frame at 4K) plus ~115 MB/frame of f64 write+read
 * traffic between resize and conversion.
 */
/* Fused chroma-2x colour kernels (restored inline-pair form after the
 * row-pipelined experiment measured SLOWER -- the int32 chroma row buffers
 * add 66 MB/frame of traffic, more than the branches cost).  OMP threshold
 * dropped to 1<<21: 4K (8.29 M px) was falling just below 1<<23 and staying
 * serial on this 2-core box. */
void ptn_yuv2rgb_c2x(const double* Y, const double* U, const double* V,
                     uint8_t* rgb, int W, int H)
{
    const int cw = W >> 1;
    const long npix = (long)W * H;
    _Pragma("omp parallel if(npix > (1<<19))")
    {
        double* bu = (double*)malloc((size_t)cw * sizeof(double));
        double* bv = (double*)malloc((size_t)cw * sizeof(double));
        #pragma omp for schedule(static)
        for (int r = 0; r < H; ++r) {
            const int k  = r >> 1;
            const int ch = H >> 1;
            const int kv = (r & 1) ? (k + 1 < ch ? k + 1 : ch - 1)
                                   : (k - 1 >= 0 ? k - 1 : 0);
            const double* ur  = U + (long)k  * cw;
            const double* ur2 = U + (long)kv * cw;
            const double* vr  = V + (long)k  * cw;
            const double* vr2 = V + (long)kv * cw;
            int m;
            for (m = 0; m < cw; ++m) {
                bu[m] = 0.75 * ur[m] + 0.25 * ur2[m];
                bv[m] = 0.75 * vr[m] + 0.25 * vr2[m];
            }
            const double* yr = Y + (long)r * W;
            uint8_t* o = rgb + (long)r * W * 3;
            int j = 0;
            while (j < W) {
                const int m  = j >> 1;
                const int mm = m - 1 >= 0 ? m - 1 : 0;
                const int mp = m + 1 < cw ? m + 1 : cw - 1;
                double cbe = 0.75 * bu[m] + 0.25 * bu[mm];
                double cre = 0.75 * bv[m] + 0.25 * bv[mm];
                double cbo = 0.75 * bu[m] + 0.25 * bu[mp];
                double cro = 0.75 * bv[m] + 0.25 * bv[mp];

                int32_t y, cb, cr, g;
                y  = f64_path(yr[j]);
                cb = f64_path(cbe);
                cr = f64_path(cre);
                g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                o[3*j+0] = sat8(y + LUT_R[cr]);
                o[3*j+1] = sat8((g + 4096) >> 13);
                o[3*j+2] = sat8(y + LUT_B[cb]);
                if (j + 1 < W) {
                    y  = f64_path(yr[j+1]);
                    cb = f64_path(cbo);
                    cr = f64_path(cro);
                    g  = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                    o[3*(j+1)+0] = sat8(y + LUT_R[cr]);
                    o[3*(j+1)+1] = sat8((g + 4096) >> 13);
                    o[3*(j+1)+2] = sat8(y + LUT_B[cb]);
                }
                j += 2;
            }
        }
        free(bu);
        free(bv);
    }
}

void ptn_yuv2rgb_c2x_f32(const float* Y, const float* U, const float* V,
                         uint8_t* rgb, int W, int H)
{
    const int cw = W >> 1;
    const long npix = (long)W * H;
    _Pragma("omp parallel if(npix > (1<<19))")
    {
        float* bu = (float*)malloc((size_t)cw * sizeof(float));
        float* bv = (float*)malloc((size_t)cw * sizeof(float));
        #pragma omp for schedule(static)
        for (int r = 0; r < H; ++r) {
            const int k  = r >> 1;
            const int ch = H >> 1;
            const int kv = (r & 1) ? (k + 1 < ch ? k + 1 : ch - 1)
                                   : (k - 1 >= 0 ? k - 1 : 0);
            const float* ur  = U + (long)k  * cw;
            const float* ur2 = U + (long)kv * cw;
            const float* vr  = V + (long)k  * cw;
            const float* vr2 = V + (long)kv * cw;
            int m;
            for (m = 0; m < cw; ++m) {
                bu[m] = 0.75f * ur[m] + 0.25f * ur2[m];
                bv[m] = 0.75f * vr[m] + 0.25f * vr2[m];
            }
            const float* yr = Y + (long)r * W;
            uint8_t* o = rgb + (long)r * W * 3;
            int j = 0;
            while (j < W) {
                const int m  = j >> 1;
                const int mm = m - 1 >= 0 ? m - 1 : 0;
                const int mp = m + 1 < cw ? m + 1 : cw - 1;
                float cbe = 0.75f * bu[m] + 0.25f * bu[mm];
                float cre = 0.75f * bv[m] + 0.25f * bv[mm];
                float cbo = 0.75f * bu[m] + 0.25f * bu[mp];
                float cro = 0.75f * bv[m] + 0.25f * bv[mp];

                int32_t y, cb, cr, g;
                y  = f32_path(yr[j]);
                cb = f32_path(cbe);
                cr = f32_path(cre);
                g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                o[3*j+0] = sat8(y + LUT_R[cr]);
                o[3*j+1] = sat8((g + 4096) >> 13);
                o[3*j+2] = sat8(y + LUT_B[cb]);
                if (j + 1 < W) {
                    y  = f32_path(yr[j+1]);
                    cb = f32_path(cbo);
                    cr = f32_path(cro);
                    g  = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                    o[3*(j+1)+0] = sat8(y + LUT_R[cr]);
                    o[3*(j+1)+1] = sat8((g + 4096) >> 13);
                    o[3*(j+1)+2] = sat8(y + LUT_B[cb]);
                }
                j += 2;
            }
        }
        free(bu);
        free(bv);
    }
}

/* ==================== v7: fused Lanczos-4 2x upsample + band add =========
 *
 * The pyramid sweep was the measured #1 decode cost (~47 ms/frame at 4K):
 *   for each level:  cur = cv2.resize(cur, 2x, INTER_LANCZOS4) + band[k]
 * cv2's L4 kernel was extracted empirically (impulse response) and is
 * EXACTLY a textbook Lanczos-4 sinc product normalised per phase, separable,
 * with BORDER_REPLICATE edges.  The 8 taps below are that kernel, phase
 * "-0.25" (output column 2m) -- and its mirror "+0.25" (column 2m+1), which
 * is the same array read backwards.
 *
 * Semantics contract (checked in tests): identical formula in the numpy
 * fallback (transform.py:_l4_up2), computed in the same op order in float32,
 * so C and fallback agree bit-for-bit; and equal to cv2.resize INTER_LANCZOS4
 * to float32 rounding (verified, see tests).
 *
 * The "+ band" fusion kills one full-res write+read per level (66 MB at 4K
 * f32), and the whole thing is OMP row parallel.
 */

static const float L4E[8] = {   /* phase -0.25, j from i-4 .. i+3 */
    -0.00397062f, 0.03146775f, -0.09166057f, 0.28268394f,
     0.89338857f, -0.1523039f,  0.05544899f, -0.01505417f
};

#define CLAMP_IMP(x, n) ((x) < 0 ? 0 : ((x) >= (n) ? (n) - 1 : (x)))

void ptn_l4_up2_add_f32(const float* src, const float* band, float* out,
                        int h, int w)
{
    /* ring design: each OUTPUT row needs 8 horizontally-convolved src rows,
     * and consecutive out rows share 7 of them by pawing the ring.  Src row q
     * in the ring always holds src row (q mod RINGBASE) currently --
     * tracked by absolute row index for clamped OOB rows (-1, h, ...). */
    const int ow = 2 * w, oh = 2 * h;
    const long npix = (long)ow * oh;
    _Pragma("omp parallel if(npix > (1<<19))")
    {
        float* tmp[8];
        long row_in_ring[8];
        for (int q = 0; q < 8; ++q) {
            tmp[q] = (float*)malloc((size_t)ow * sizeof(float));
            row_in_ring[q] = -9999;
        }
        #pragma omp for schedule(static)
        for (int r = 0; r < oh; ++r) {
            const int k = r >> 1;
            const int phase_off = (r & 1) ? -3 : -4;
            const int base = k + phase_off;      /* src rows base..base+7 */
            /* make sure rows base..base+7 sit in tmp[0..7] in order */
            for (int q = 0; q < 8; ++q) {
                if (row_in_ring[q] == base + q) continue;
                /* find it elsewhere and swap, else compute */
                int found = -1;
                for (int q2 = q + 1; q2 < 8; ++q2)
                    if (row_in_ring[q2] == base + q) { found = q2; break; }
                if (found >= 0) {
                    float* tt = tmp[q]; tmp[q] = tmp[found]; tmp[found] = tt;
                    long tr = row_in_ring[q];
                    row_in_ring[q] = row_in_ring[found]; row_in_ring[found] = tr;
                    continue;
                }
                const int sr = CLAMP_IMP(base + q, h);
                const float* sp = src + (long)sr * w;
                float* tq = tmp[q];
                for (int m = 0; m < w; ++m) {
                    float acc_e = 0.f, acc_o = 0.f;
                    for (int q2 = 0; q2 < 8; ++q2) {
                        const int ce = CLAMP_IMP(m - 4 + q2, w);
                        const int co = CLAMP_IMP(m - 3 + q2, w);
                        acc_e += L4E[q2] * sp[ce];
                        acc_o += L4E[7 - q2] * sp[co];
                    }
                    tq[2*m] = acc_e;
                    tq[2*m+1] = acc_o;
                }
                row_in_ring[q] = base + q;
            }
            float* o = out + (long)r * ow;
            const float* b = band + (long)r * ow;
            const int mir = (r & 1);
            if (band) {
                for (int j = 0; j < ow; ++j) {
                    float acc = b[j];
                    acc += (mir ? L4E[7-0] : L4E[0]) * tmp[0][j];
                    acc += (mir ? L4E[7-1] : L4E[1]) * tmp[1][j];
                    acc += (mir ? L4E[7-2] : L4E[2]) * tmp[2][j];
                    acc += (mir ? L4E[7-3] : L4E[3]) * tmp[3][j];
                    acc += (mir ? L4E[7-4] : L4E[4]) * tmp[4][j];
                    acc += (mir ? L4E[7-5] : L4E[5]) * tmp[5][j];
                    acc += (mir ? L4E[7-6] : L4E[6]) * tmp[6][j];
                    acc += (mir ? L4E[7-7] : L4E[7]) * tmp[7][j];
                    o[j] = acc;
                }
            } else {
                for (int j = 0; j < ow; ++j) {
                    float acc = 0.f;
                    acc += (mir ? L4E[7-0] : L4E[0]) * tmp[0][j];
                    acc += (mir ? L4E[7-1] : L4E[1]) * tmp[1][j];
                    acc += (mir ? L4E[7-2] : L4E[2]) * tmp[2][j];
                    acc += (mir ? L4E[7-3] : L4E[3]) * tmp[3][j];
                    acc += (mir ? L4E[7-4] : L4E[4]) * tmp[4][j];
                    acc += (mir ? L4E[7-5] : L4E[5]) * tmp[5][j];
                    acc += (mir ? L4E[7-6] : L4E[6]) * tmp[6][j];
                    acc += (mir ? L4E[7-7] : L4E[7]) * tmp[7][j];
                    o[j] = acc;
                }
            }
        }
        for (int q = 0; q < 8; ++q)
            free(tmp[q]);
    }
}

/* v8-rd-coef: serial un-scatter of the zig-zag coefficient stream.
 *
 * Inputs are already ints (decoded rANS symbols); the loop is exactly the
 * Python one in codec.py:_stream_to_coef, guard for guard, so results are
 * bit-exact.  cq_zz must come in zeroed, nb*64 int64. */
void ptn_stream_to_coef(const long long* counts, const long long* positions,
                        const long long* levels, const long long* coded_idx,
                        long long n_coded, long long nb,
                        long long n_pos, long long n_lev,
                        long long* cq_zz)
{
    long long pi = 0, li = 0;
    for (long long j = 0; j < n_coded; ++j) {
        const long long n = (j < n_coded) ? counts[j] : 0;
        long long pos = 0;
        const long long b = coded_idx[j];
        for (long long k = 0; k < n; ++k) {
            if (pi >= n_pos || li >= n_lev)
                break;
            pos += positions[pi];
            pi += 1;
            if (pos >= 64)
                break;
            const long long u = levels[li];
            cq_zz[b * 64 + pos] = (u & 1) ? -((u + 1) >> 1) : (u >> 1);
            li += 1;
            pos += 1;
        }
    }
}

/* v9-rans: serial rANS byte-stream decoder, exact port of rans.rans_decode.
 * Constants match rans.py: PROB_BITS=12, L=1<<23, state from first 4 bytes BE. */
void ptn_rans_decode(const unsigned char* data, long long n,
                     const int* lut, const int* freq_syms, const long long* cum_syms,
                     int n_syms, long long count, int* out)
{
    (void)n_syms;
    long long x = ((long long)data[0] << 24) | ((long long)data[1] << 16)
                | ((long long)data[2] << 8) | (long long)data[3];
    long long pos = 4;
    const long long RANS_L = 1LL << 23;
    for (long long i = 0; i < count; ++i) {
        const long long slot = x & 4095LL;
        const int s = lut[slot];
        const long long f = (long long)freq_syms[s];
        const long long c = cum_syms[s];
        x = f * (x >> 12) + slot - c;
        while (x < RANS_L && pos < n) {
            x = (x << 8) | (long long)data[pos];
            pos += 1;
        }
        out[i] = s;
    }
}

/* v10-rlut: fill the PROB_SCALE=4096 slot->symbol table in C instead of a
 * Python per-symbol loop (was ~1.4 ms/call on hd-tier frames with a large
 * alphabet).  Returns 0 on success. */
int ptn_rans_lut(const int* syms, const int* freq, const long long* cum,
                 int n_syms, int* lut4096)
{
    if (n_syms <= 0 || n_syms > 4096)
        return -1;
    for (int i = 0; i < n_syms; ++i) {
        const long long c = cum[i];
        const long long f = freq[i];
        const int s = syms[i];
        if (c < 0 || f < 0 || c + f > 4096)
            return -2;
        for (long long j = c; j < c + f; ++j)
            lut4096[j] = s;
    }
    return 0;
}
/* ================= v12: AVX-512 fused chroma-2x colour kernel ============
 *
 * Bit-exact twin of ptn_yuv2rgb_c2x_f32.  Per row, per 16 output px:
 *   - bu/bv built by streamed FMAs
 *   - one y16 vector -> rint(roundscale) -> clamp -> int
 *   - three index vectors gather bu/bv at strided taps; even lanes blend
 *     (m, mm), odd lanes (m, mp) via one mask-mix
 *   - G fully arithmetic, R/B through LUT_R/LUT_B gather
 *   - pack to 48 RGB bytes via cvtusepi32_epi8 + unrolled scatter store
 * Scalar head (first 32 px) and tail (last 32 px) keep the original edge
 * clamps, so the vector section never touches an out-of-range chroma tap.
 */
#if defined(__AVX512F__)
#include <immintrin.h>

void ptn_yuv2rgb_c2x_f32_v512(const float* Y, const float* U, const float* V,
                              uint8_t* rgb, int W, int H)
{
    const int cw = W >> 1;
    const long npix = (long)W * H;
    /* permute patterns for three contiguous loads (replaces gathers) */
    const __m512i idx_m  = _mm512_set_epi32(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0);
    const __m512i idx_mm = _mm512_set_epi32(8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0);
    /* idx_mp reuses idx_m on the +1-shifted load */
    const __m512  f075   = _mm512_set1_ps(0.75f);
    const __m512  f025   = _mm512_set1_ps(0.25f);
    const __m512  f0     = _mm512_setzero_ps();
    const __m512  f255   = _mm512_set1_ps(255.f);
    const __mmask16 odd  = (__mmask16)0xAAAA;

    _Pragma("omp parallel if(npix > (1<<19))")
    {
        float* bu = (float*)malloc(((size_t)cw + 16) * sizeof(float));
        float* bv = (float*)malloc(((size_t)cw + 16) * sizeof(float));
        #pragma omp for schedule(static)
        for (int r = 0; r < H; ++r) {
            const int k  = r >> 1;
            const int ch = H >> 1;
            const int kv = (r & 1) ? (k + 1 < ch ? k + 1 : ch - 1)
                                   : (k - 1 >= 0 ? k - 1 : 0);
            const float* ur  = U + (long)k  * cw;
            const float* ur2 = U + (long)kv * cw;
            const float* vr  = V + (long)k  * cw;
            const float* vr2 = V + (long)kv * cw;
            int m = 0;
            for (; m + 16 <= cw; m += 16) {
                _mm512_storeu_ps(bu + m, _mm512_add_ps(
                    _mm512_mul_ps(f075, _mm512_loadu_ps(ur + m)),
                    _mm512_mul_ps(f025, _mm512_loadu_ps(ur2 + m))));
                _mm512_storeu_ps(bv + m, _mm512_add_ps(
                    _mm512_mul_ps(f075, _mm512_loadu_ps(vr + m)),
                    _mm512_mul_ps(f025, _mm512_loadu_ps(vr2 + m))));
            }
            for (; m < cw; ++m) {
                bu[m] = 0.75f * ur[m] + 0.25f * ur2[m];
                bv[m] = 0.75f * vr[m] + 0.25f * vr2[m];
            }

            const float* yr = Y + (long)r * W;
            uint8_t* o = rgb + (long)r * W * 3;
            int j = 0;

            /* scalar head: first 32 px (covers left-edge mm clamp) */
            for (; j < 32 && j < W; j += 2) {
                const int m  = j >> 1;
                const int mm = m - 1 >= 0 ? m - 1 : 0;
                const int mp = m + 1 < cw ? m + 1 : cw - 1;
                int32_t y, cb, cr, g;
                y  = f32_path(yr[j]);
                cb = f32_path(0.75f * bu[m] + 0.25f * bu[mm]);
                cr = f32_path(0.75f * bv[m] + 0.25f * bv[mm]);
                g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                o[3*j+0] = sat8(y + LUT_R[cr]);
                o[3*j+1] = sat8((g + 4096) >> 13);
                o[3*j+2] = sat8(y + LUT_B[cb]);
                if (j + 1 < W) {
                    y  = f32_path(yr[j+1]);
                    cb = f32_path(0.75f * bu[m] + 0.25f * bu[mp]);
                    cr = f32_path(0.75f * bv[m] + 0.25f * bv[mp]);
                    g  = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                    o[3*(j+1)+0] = sat8(y + LUT_R[cr]);
                    o[3*(j+1)+1] = sat8((g + 4096) >> 13);
                    o[3*(j+1)+2] = sat8(y + LUT_B[cb]);
                }
            }

            /* vector main section */
            for (; j + 16 <= W - 32; j += 16) {
                const int m0 = j >> 1;
                __m512 yf = _mm512_loadu_ps(yr + j);
                yf = _mm512_roundscale_ps(yf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                yf = _mm512_min_ps(_mm512_max_ps(yf, f0), f255);
                const __m512i yi = _mm512_cvtps_epi32(yf);

                /* three contiguous loads + permutes (gathers are slow here) */
                __m512 bum  = _mm512_permutexvar_ps(idx_m,  _mm512_loadu_ps(bu + m0));
                __m512 bumm = _mm512_permutexvar_ps(idx_mm, _mm512_loadu_ps(bu + m0 - 1));
                __m512 bump = _mm512_permutexvar_ps(idx_m,  _mm512_loadu_ps(bu + m0 + 1));
                __m512 bvm  = _mm512_permutexvar_ps(idx_m,  _mm512_loadu_ps(bv + m0));
                __m512 bvmm = _mm512_permutexvar_ps(idx_mm, _mm512_loadu_ps(bv + m0 - 1));
                __m512 bvmp = _mm512_permutexvar_ps(idx_m,  _mm512_loadu_ps(bv + m0 + 1));
                __m512 bu2 = _mm512_mask_blend_ps(odd, bumm, bump);
                __m512 bv2 = _mm512_mask_blend_ps(odd, bvmm, bvmp);
                __m512 cbf = _mm512_add_ps(_mm512_mul_ps(f075, bum), _mm512_mul_ps(f025, bu2));
                __m512 crf = _mm512_add_ps(_mm512_mul_ps(f075, bvm), _mm512_mul_ps(f025, bv2));
                cbf = _mm512_roundscale_ps(cbf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                cbf = _mm512_min_ps(_mm512_max_ps(cbf, f0), f255);
                crf = _mm512_roundscale_ps(crf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                crf = _mm512_min_ps(_mm512_max_ps(crf, f0), f255);
                const __m512i cbi = _mm512_cvtps_epi32(cbf);
                const __m512i cri = _mm512_cvtps_epi32(crf);

                __m512i r32 = _mm512_add_epi32(yi,
                    _mm512_i32gather_epi32(cri, (const int*)LUT_R, 4));
                __m512i b32 = _mm512_add_epi32(yi,
                    _mm512_i32gather_epi32(cbi, (const int*)LUT_B, 4));
                __m512i g32 = _mm512_srai_epi32(
                    _mm512_add_epi32(
                        _mm512_sub_epi32(
                            _mm512_slli_epi32(yi, 13),
                            _mm512_add_epi32(
                                _mm512_mullo_epi32(_mm512_sub_epi32(cbi, _mm512_set1_epi32(128)),
                                                   _mm512_set1_epi32(2818)),
                                _mm512_mullo_epi32(_mm512_sub_epi32(cri, _mm512_set1_epi32(128)),
                                                   _mm512_set1_epi32(5849)))),
                        _mm512_set1_epi32(4096)), 13);
                const __m512i i0 = _mm512_setzero_si512();
                const __m512i i255 = _mm512_set1_epi32(255);
                r32 = _mm512_min_epi32(_mm512_max_epi32(r32, i0), i255);
                g32 = _mm512_min_epi32(_mm512_max_epi32(g32, i0), i255);
                b32 = _mm512_min_epi32(_mm512_max_epi32(b32, i0), i255);

                /* VBMI: one vpermb builds the 48-byte RGB interleave.
                 * src lanes: r in bytes 0..15, g in 16..31, b in 32..47 */
                {
                    static const uint8_t PERM[64] __attribute__((aligned(64))) = {
                        0,16,32,  1,17,33,  2,18,34,  3,19,35,
                        4,20,36,  5,21,37,  6,22,38,  7,23,39,
                        8,24,40,  9,25,41,  10,26,42, 11,27,43,
                        12,28,44, 13,29,45, 14,30,46, 15,31,47,
                        0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
                    };
                    const __m512i perm = _mm512_load_si512((const __m512i*)PERM);
                    __m512i src = _mm512_castsi128_si512(_mm512_cvtusepi32_epi8(r32));
                    src = _mm512_inserti32x4(src, _mm512_cvtusepi32_epi8(g32), 1);
                    src = _mm512_inserti32x4(src, _mm512_cvtusepi32_epi8(b32), 2);
                    __m512i rgb16 = _mm512_permutexvar_epi8(perm, src);
                    _mm512_mask_storeu_epi8(o + 3*(long)j,
                                            (__mmask64)(__UINT64_C(0x0000FFFFFFFFFFFF)),
                                            rgb16);
                }
            }

            /* scalar tail (rest of row incl. right-edge mp clamp) */
            while (j < W) {
                const int m  = j >> 1;
                const int mm = m - 1 >= 0 ? m - 1 : 0;
                const int mp = m + 1 < cw ? m + 1 : cw - 1;
                int32_t y, cb, cr, g;
                y  = f32_path(yr[j]);
                cb = f32_path(0.75f * bu[m] + 0.25f * bu[mm]);
                cr = f32_path(0.75f * bv[m] + 0.25f * bv[mm]);
                g = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                o[3*j+0] = sat8(y + LUT_R[cr]);
                o[3*j+1] = sat8((g + 4096) >> 13);
                o[3*j+2] = sat8(y + LUT_B[cb]);
                if (j + 1 < W) {
                    y  = f32_path(yr[j+1]);
                    cb = f32_path(0.75f * bu[m] + 0.25f * bu[mp]);
                    cr = f32_path(0.75f * bv[m] + 0.25f * bv[mp]);
                    g  = (y << 13) - 2818 * (cb - 128) - 5849 * (cr - 128);
                    o[3*(j+1)+0] = sat8(y + LUT_R[cr]);
                    o[3*(j+1)+1] = sat8((g + 4096) >> 13);
                    o[3*(j+1)+2] = sat8(y + LUT_B[cb]);
                }
                j += 2;
            }
        }
        free(bu); free(bv);
    }
}
#endif

/* ============== v13: AVX-512 fused pyramid-up + band-add kernel ==========
 *
 * Bit-exact twin of ptn_l4_up2_add_f32.  Same structure (8-row ring over
 * src rows with per-row horizontal 8-tap conv, then vertical 8-tap + band
 * add), but:
 *   - horizontal pass is vectorised 16-wide over the row, intermediate
 *     results kept in separate even/odd row buffers (no per-m scatter);
 *     left/right 4-column edge rows stay scalar with the original clamped
 *     path;
 *   - vertical pass processes 16 outputs at once: even lanes from tmp_e,
 *     odd lanes from tmp_o, then one permute interleaves them back to the
 *     output line.  The 8-round accumulation order per lane matches the
 *     scalar kernel exactly (mul then add each round, no FMA contraction).
 * Per-lane arithmetic order equals the scalar kernel, so every output
 * float is bit-identical.
 */
#if defined(__AVX512F__)
void ptn_l4_up2_add_f32_v512(const float* src, const float* band, float* out,
                             int h, int w)
{
    const int ow = 2 * w, oh = 2 * h;
    const long npix = (long)ow * oh;
    /* even/odd -> interleaved lane order (set_epi32 lists lane15 first) */
    const __m512i ip = _mm512_set_epi32(15, 7, 14, 6, 13, 5, 12, 4,
                                        11, 3, 10, 2,  9, 1,  8, 0);

    _Pragma("omp parallel if(npix > (1<<19))")
    {
        float* tmp_e[8];
        float* tmp_o[8];
        long row_in_ring[8];
        for (int q = 0; q < 8; ++q) {
            tmp_e[q] = (float*)malloc(((size_t)w + 32) * sizeof(float));
            tmp_o[q] = (float*)malloc(((size_t)w + 32) * sizeof(float));
            row_in_ring[q] = -9999;
        }
        #pragma omp for schedule(static)
        for (int r = 0; r < oh; ++r) {
            const int k = r >> 1;
            const int phase_off = (r & 1) ? -3 : -4;
            const int base = k + phase_off;
            for (int q = 0; q < 8; ++q) {
                if (row_in_ring[q] == base + q) continue;
                int found = -1;
                for (int q2 = q + 1; q2 < 8; ++q2)
                    if (row_in_ring[q2] == base + q) { found = q2; break; }
                if (found >= 0) {
                    float* tt;
                    tt = tmp_e[q]; tmp_e[q] = tmp_e[found]; tmp_e[found] = tt;
                    tt = tmp_o[q]; tmp_o[q] = tmp_o[found]; tmp_o[found] = tt;
                    long tr = row_in_ring[q];
                    row_in_ring[q] = row_in_ring[found];
                    row_in_ring[found] = tr;
                    continue;
                }
                const int sr = CLAMP_IMP(base + q, h);
                const float* sp = src + (long)sr * w;
                float* te = tmp_e[q];
                float* to = tmp_o[q];
                int m = 0;
                for (; m < 4 && m < w; ++m) {
                    float acc_e = 0.f, acc_o = 0.f;
                    for (int q2 = 0; q2 < 8; ++q2) {
                        acc_e += L4E[q2] * sp[CLAMP_IMP(m - 4 + q2, w)];
                        acc_o += L4E[7 - q2] * sp[CLAMP_IMP(m - 3 + q2, w)];
                    }
                    te[m] = acc_e;
                    to[m] = acc_o;
                }
                for (; m + 16 <= w - 4 + 16 && m + 16 <= w - 4 + 16; m += 16) {
                    if (m + 16 > w - 4) break;
                    __m512 ae = _mm512_mul_ps(_mm512_set1_ps(L4E[0]),
                                              _mm512_loadu_ps(sp + m - 4 + 0));
                    __m512 ao = _mm512_mul_ps(_mm512_set1_ps(L4E[7]),
                                              _mm512_loadu_ps(sp + m - 3 + 0));
                    #pragma GCC unroll 7
                    for (int q2 = 1; q2 < 8; ++q2) {
                        ae = _mm512_add_ps(ae,
                            _mm512_mul_ps(_mm512_set1_ps(L4E[q2]),
                                          _mm512_loadu_ps(sp + m - 4 + q2)));
                        ao = _mm512_add_ps(ao,
                            _mm512_mul_ps(_mm512_set1_ps(L4E[7 - q2]),
                                          _mm512_loadu_ps(sp + m - 3 + q2)));
                    }
                    _mm512_storeu_ps(te + m, ae);
                    _mm512_storeu_ps(to + m, ao);
                }
                for (; m < w; ++m) {
                    float acc_e = 0.f, acc_o = 0.f;
                    for (int q2 = 0; q2 < 8; ++q2) {
                        acc_e += L4E[q2] * sp[CLAMP_IMP(m - 4 + q2, w)];
                        acc_o += L4E[7 - q2] * sp[CLAMP_IMP(m - 3 + q2, w)];
                    }
                    te[m] = acc_e;
                    to[m] = acc_o;
                }
                row_in_ring[q] = base + q;
            }
            float* o = out + (long)r * ow;
            const float* b = band + (long)r * ow;
            const int mir = (r & 1);
            const __m512 wv[8] = {
                _mm512_set1_ps(L4E[mir ? 7-0 : 0]), _mm512_set1_ps(L4E[mir ? 7-1 : 1]),
                _mm512_set1_ps(L4E[mir ? 7-2 : 2]), _mm512_set1_ps(L4E[mir ? 7-3 : 3]),
                _mm512_set1_ps(L4E[mir ? 7-4 : 4]), _mm512_set1_ps(L4E[mir ? 7-5 : 5]),
                _mm512_set1_ps(L4E[mir ? 7-6 : 6]), _mm512_set1_ps(L4E[mir ? 7-7 : 7])};
            int j = 0;
            for (; j + 16 <= ow; j += 16) {
                const int m = j >> 1;
                __m512 acc;
                if (band) {
                    acc = _mm512_loadu_ps(b + j);
                } else {
                    acc = _mm512_setzero_ps();
                }
                /* per q: build 16-lane tap vector [e0..e7 | o0..o7] -> ip interleave */
                for (int q = 0; q < 8; ++q) {
                    __m512 both = _mm512_castps256_ps512(_mm256_loadu_ps(tmp_e[q] + m));
                    both = _mm512_insertf32x8(both, _mm256_loadu_ps(tmp_o[q] + m), 1);
                    acc = _mm512_add_ps(acc,
                        _mm512_mul_ps(wv[q], _mm512_permutexvar_ps(ip, both)));
                }
                _mm512_storeu_ps(o + j, acc);
            }
            for (; j < ow; ++j) {
                const int m = j >> 1;
                float acc = band ? b[j] : 0.f;
                const int odd = j & 1;
                for (int q = 0; q < 8; ++q) {
                    const float* row = odd ? tmp_o[q] : tmp_e[q];
                    acc += (mir ? L4E[7 - q] : L4E[q]) * row[m];
                }
                o[j] = acc;
            }
        }
        for (int q = 0; q < 8; ++q) { free(tmp_e[q]); free(tmp_o[q]); }
    }
}
#endif

/* ============ v14: block-translation warp (integer MVs), f32 ==============
 *
 * Bit-exact twin of motion.warp_plane(plane, mv, bs):
 *   dy = round(mv[:, :, 1]), dx = round(mv[:, :, 0])          (nearest-even)
 *   for block (by,bx), offset (oy,ox):
 *     out[by*bs+oy, bx*bs+ox] =
 *         plane[ clip(by*bs+oy+dy, 0, h-1) * w + clip(bx*bs+ox+dx, 0, w-1) ]
 * Same gather addresses, no edge-pad materialisation and no 4-D transpose:
 * interior blocks are row-memcpys, border blocks per-pixel clipped.
 * math_check: nearbyintf uses the current rounding mode (to-nearest-even by
 * default) == np.round == np.rint.  Any mismatch path longs to python.
 */
void ptn_warp_plane_f32(const float* plane, long h, long w,
                        const float* mv, long nbh, long nbw, int bs,
                        float* out)
{
    _Pragma("omp parallel for schedule(static) if(nbh * nbw > 512)")
    for (long by = 0; by < nbh; ++by) {
        for (long bx = 0; bx < nbw; ++bx) {
            /* np.round -> int32 contract: nearbyint (ties-to-even) then cast,
             * expressed via the same builtin used in f32_path so no libm is
             * required and the emitted vroundps keeps the default rounding. */
            const long dy = (long)__builtin_rintf(mv[(by * nbw + bx) * 2 + 1]);
            const long dx = (long)__builtin_rintf(mv[(by * nbw + bx) * 2 + 0]);
            for (int oy = 0; oy < bs; ++oy) {
                long sy = by * bs + oy + dy;
                if (sy < 0) sy = 0;
                if (sy >= h) sy = h - 1;
                const long x0 = bx * bs + dx;
                float* o = out + (by * bs + oy) * (nbw * bs) + bx * bs;
                /* interior horizontal run? */
                if (x0 >= 0 && x0 + bs <= w) {
                    memcpy(o, plane + sy * w + x0, (size_t)bs * sizeof(float));
                } else {
                    for (int ox = 0; ox < bs; ++ox) {
                        long sx = x0 + ox;
                        if (sx < 0) sx = 0;
                        if (sx >= w) sx = w - 1;
                        o[ox] = plane[sy * w + sx];
                    }
                }
            }
        }
    }
}

/* ============ v15: intra DC wavefront reconstruction (f64) ================
 *
 * Bit-exact twin of the python 2-by-d loop in _decode_intra_level:
 *   has_l, has_t: p = 0.5*(p_l + p_t)  /  p_l  /  p_t  /  global_dc
 *   recon[r][c] = p + res[r][c] * qm00
 * Operand ADD order matches the python expression 0.5*(l + t) + res*qm00
 * evaluated as (0.5*(l+t)) + (res*qm00).  Python evaluates
 * 0.5 * (l + t) as: sum first, then multiply.  C does the same.
 */
void ptn_intra_dc_scan(const double* res, double global_dc, long nbh, long nbw,
                       double qm00, double* recon)
{
    for (long r = 0; r < nbh; ++r) {
        for (long c = 0; c < nbw; ++c) {
            double p;
            if (c > 0 && r > 0)
                p = 0.5 * (recon[r * nbw + c - 1] + recon[(r - 1) * nbw + c]);
            else if (c > 0)
                p = recon[r * nbw + c - 1];
            else if (r > 0)
                p = recon[(r - 1) * nbw + c];
            else
                p = global_dc;
            recon[r * nbw + c] = p + res[r * nbw + c] * qm00;
        }
    }
}
