/* Fused phi-conv — no product tensors (see fconv.h).
 * OpenMP over co (proven pattern); scalar co-worker shared with SIMD tails.
 */
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include "fconv.h"
#include "bridge.h"
#include "luts.h"

static void conv_co(const trip_t *P, int Hp, int Wp, int H, int Wd, int Cin,
                    int ho, int wo, const cW_t *cw, int co,
                    const trip_t *bias_or_null, int m_out, int stride,
                    int64_t base, trip_t *out, int Cout);

void phi_conv_fused(const trip_t *in, int H, int Wd, int Cin, const cW_t *cw,
                    const trip_t *bias_or_null, int m_out, int stride, int pad,
                    trip_t *out, int *Ho, int *Wo) {
    int kh = cw->kh, Cout = cw->Cout;
    int ho = (H + 2 * pad - kh) / stride + 1;
    int wo = (Wd + 2 * pad - kh) / stride + 1;
    *Ho = ho;
    *Wo = wo;
    int Hp = H + 2 * pad, Wp = Wd + 2 * pad;
    int64_t base = LUT_FRAC[0];
    trip_t *P = (trip_t *)calloc((size_t)Hp * Wp * Cin, sizeof(trip_t));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < Wd; x++)
            for (int c = 0; c < Cin; c++)
                P[((y + pad) * Wp + (x + pad)) * Cin + c] =
                    in[(y * Wd + x) * Cin + c];
    int N = ho * wo;
    /* co-parallel (disjoint out columns, order-free sums — same pattern
     * as conv.c, gated bit-exact there; re-gated here by chain parity) */
#pragma omp parallel for schedule(static)
    for (int co = 0; co < Cout; co++)
        conv_co(P, Hp, Wp, H, Wd, Cin, ho, wo, cw, co, bias_or_null, m_out,
                stride, base, out, Cout);
    (void)N;
    free(P);
}

/* scalar one-channel worker (shared by fused main path + SIMD tails) */
static void conv_co(const trip_t *P, int Hp, int Wp, int H, int Wd, int Cin,
                    int ho, int wo, const cW_t *cw, int co,
                    const trip_t *bias_or_null, int m_out, int stride,
                    int64_t base, trip_t *out, int Cout) {
    int kh = cw->kh;
    int N = ho * wo;
    int64_t *acc = (int64_t *)malloc(sizeof(int64_t) * N);
    for (int i = 0; i < N; i++) acc[i] = 0;
    if (bias_or_null) {
        trip_t b = bias_or_null[co];
        int64_t q = 0;
        if (!b.z) {
            int64_t d = (int64_t)m_out - (int64_t)b.e;
            q = d < 0 ? (int64_t)b.s * base
                : (d > FRAC_CAP ? 0 : (int64_t)b.s * LUT_FRAC[d]);
        }
        for (int i = 0; i < N; i++) acc[i] = q;
    }
    for (int oy = 0; oy < ho; oy++) {
        for (int ox = 0; ox < wo; ox++) {
            int64_t tot = 0;
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy * stride + ky;
                for (int kx = 0; kx < kh; kx++) {
                    int ix = ox * stride + kx;
                    const trip_t *A = &P[(iy * Wp + ix) * Cin];
                    const int8_t *ws =
                        &cw->s[((co * kh + ky) * kh + kx) * Cin];
                    const int32_t *we =
                        &cw->e[((co * kh + ky) * kh + kx) * Cin];
                    const uint8_t *wz =
                        &cw->z[((co * kh + ky) * kh + kx) * Cin];
                    for (int ci = 0; ci < Cin; ci++) {
                        int8_t ps = (int8_t)(A[ci].s * ws[ci]);
                        int64_t pe = (int64_t)A[ci].e + (int64_t)we[ci]
                            - PHI_BIAS;
                        if (pe < 0) pe = 0;
                        else if (pe > 65535) pe = 65535;
                        if (!(A[ci].z | wz[ci])) {
                            int64_t d = (int64_t)m_out - pe;
                            tot += d < 0
                                ? (int64_t)ps * base
                                : (d > FRAC_CAP
                                       ? 0
                                       : (int64_t)ps * LUT_FRAC[d]);
                        }
                    }
                }
            }
            acc[oy * wo + ox] += tot;
        }
    }
    fq_t f = {acc, N, m_out};
    trip_t *ot = (trip_t *)malloc(sizeof(trip_t) * N);
    triples_from_fixed(&f, ot);
    for (int oy = 0; oy < ho; oy++)
        for (int ox = 0; ox < wo; ox++)
            out[(oy * wo + ox) * Cout + co] = ot[oy * wo + ox];
    free(ot);
    free(acc);
    (void)Hp;
    (void)H;
    (void)Wd;
}

/* SIMD variant (see fconv.h). Vector axis = co-blocks of 8 (weights
 * co-contiguous in OHWI); OpenMP over pixels; scalar tails per pixel.
 * Lane contract (schedules-as-semantics): lanes independent, any
 * permutation valid; pixel order irrelevant (disjoint outputs). */
#include "simd.h"

void phi_conv_fused_simd(const trip_t *in, int H, int Wd, int Cin,
                         const cW_t *cw, const trip_t *bias_or_null, int m_out,
                         int stride, int pad, trip_t *out, int *Ho, int *Wo) {
    int kh = cw->kh, Cout = cw->Cout;
    int ho = (H + 2 * pad - kh) / stride + 1;
    int wo = (Wd + 2 * pad - kh) / stride + 1;
    *Ho = ho;
    *Wo = wo;
    int Hp = H + 2 * pad, Wp = Wd + 2 * pad;
    int64_t base = LUT_FRAC[0];
    trip_t *P = (trip_t *)calloc((size_t)Hp * Wp * Cin, sizeof(trip_t));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < Wd; x++)
            for (int c = 0; c < Cin; c++)
                P[((y + pad) * Wp + (x + pad)) * Cin + c] =
                    in[(y * Wd + x) * Cin + c];
    int N = ho * wo;
    int Cc = (Cout / 8) * 8;
    int Cb = Cc / 8;
    int TAPS = kh * kh * Cin;
    /* transposed weights: [cb][tap][ci][8] contiguous (one pass over the
     * weights per call, amortized over N pixels; inner loop then streams
     * sequentially — no strided loads, no gathers) */
    int8_t *Ws8 =
        (int8_t *)malloc((size_t)Cb * TAPS * 8);
    int32_t *We8 =
        (int32_t *)malloc((size_t)Cb * TAPS * 8 * sizeof(int32_t));
    uint8_t *Wz8 =
        (uint8_t *)malloc((size_t)Cb * TAPS * 8);
    for (int cb = 0; cb < Cb; cb++) {
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kh; kx++) {
                for (int ci = 0; ci < Cin; ci++) {
                    size_t b = ((size_t)cb * TAPS + (ky * kh + kx) * Cin + ci) * 8;
                    for (int j = 0; j < 8; j++) {
                        int off = (((cb * 8 + j) * kh + ky) * kh + kx) * Cin
                            + ci;
                        Ws8[b + j] = cw->s[off];
                        We8[b + j] = cw->e[off];
                        Wz8[b + j] = cw->z[off];
                    }
                }
            }
        }
    }
#pragma omp parallel for schedule(static) collapse(2)
    for (int oy = 0; oy < ho; oy++) {
        for (int ox = 0; ox < wo; ox++) {
            int o = oy * wo + ox;
            for (int cb = 0; cb < Cb; cb++) {
                size_t wbase = (size_t)cb * TAPS * 8;
                __m512i acc =
                    _mm512_setzero_si512();
                if (bias_or_null) {
                    int8_t bs[8];
                    int32_t be[8];
                    uint8_t bz[8];
                    for (int j = 0; j < 8; j++) {
                        bs[j] = bias_or_null[cb * 8 + j].s;
                        be[j] = bias_or_null[cb * 8 + j].e;
                        bz[j] = bias_or_null[cb * 8 + j].z;
                    }
                    acc = bridge8(load8_u8(bs), load8_u32(be), load8_u8(bz),
                                  m_out, base, LUT_FRAC);
                }
                for (int ky = 0; ky < kh; ky++) {
                    int iy = oy * stride + ky;
                    for (int kx = 0; kx < kh; kx++) {
                        int ix = ox * stride + kx;
                        const trip_t *A = &P[(iy * Wp + ix) * Cin];
                        size_t tb = wbase + ((ky * kh + kx) * Cin) * 8;
                        for (int ci = 0; ci < Cin; ci++, tb += 8) {
                            /* broadcast input scalar; co8 weights stream
                             * contiguous from the transposed copy */
                            __m128i a16 = _mm_set1_epi16((short)A[ci].s);
                            __m128i w16 =
                                _mm_cvtepi8_epi16(load8_u8(&Ws8[tb]));
                            __m128i ps16 =
                                _mm_mullo_epi16(a16, w16);
                            __m128i ps8 = _mm_packs_epi16(ps16, ps16);
                            __m256i pe32 = _mm256_add_epi32(
                                _mm256_add_epi32(
                                    _mm256_set1_epi32(A[ci].e),
                                    load8_u32(&We8[tb])),
                                _mm256_set1_epi32(-PHI_BIAS));
                            /* clip [0,65535] */
                            pe32 = _mm256_max_epi32(
                                pe32, _mm256_setzero_si256());
                            pe32 = _mm256_min_epi32(
                                pe32, _mm256_set1_epi32(65535));
                            __m128i pz8 = _mm_or_si128(
                                _mm_set1_epi8((char)A[ci].z),
                                load8_u8(&Wz8[tb]));
                            __m512i q = bridge8(ps8, pe32, pz8, m_out, base,
                                                LUT_FRAC);
                            acc = _mm512_add_epi64(acc, q);
                        }
                    }
                }
                int8_t s8[8];
                int32_t e8[8];
                uint8_t z8[8];
                from_fixed8(acc, m_out, LUT_COARSE, LUT_FINE, s8, e8, z8);
                for (int j = 0; j < 8; j++) {
                    trip_t *oo = &out[o * Cout + cb * 8 + j];
                    oo->s = s8[j];
                    oo->e = e8[j];
                    oo->z = z8[j];
                }
            }
            /* scalar tail cos */
            for (int co = Cc; co < Cout; co++) {
                int64_t tot = 0;
                if (bias_or_null) {
                    trip_t b = bias_or_null[co];
                    if (!b.z) {
                        int64_t d = (int64_t)m_out - (int64_t)b.e;
                        tot = d < 0 ? (int64_t)b.s * base
                            : (d > FRAC_CAP ? 0 : (int64_t)b.s * LUT_FRAC[d]);
                    }
                }
                for (int ky = 0; ky < kh; ky++) {
                    int iy = oy * stride + ky;
                    for (int kx = 0; kx < kh; kx++) {
                        int ix = ox * stride + kx;
                        const trip_t *A = &P[(iy * Wp + ix) * Cin];
                        for (int ci = 0; ci < Cin; ci++) {
                            int off = ((co * kh + ky) * kh + kx) * Cin + ci;
                            int8_t ps = (int8_t)(A[ci].s * cw->s[off]);
                            int64_t pe = (int64_t)A[ci].e + (int64_t)cw->e[off]
                                - PHI_BIAS;
                            if (pe < 0) pe = 0;
                            else if (pe > 65535) pe = 65535;
                            if (!(A[ci].z | cw->z[off])) {
                                int64_t d = (int64_t)m_out - pe;
                                tot += d < 0
                                    ? (int64_t)ps * base
                                    : (d > FRAC_CAP
                                           ? 0
                                           : (int64_t)ps * LUT_FRAC[d]);
                            }
                        }
                    }
                }
                fq_t fw = {&tot, 1, m_out};
                trip_t oo;
                triples_from_fixed(&fw, &oo);
                out[o * Cout + co] = oo;
            }
        }
    }
    (void)N;
    free(Ws8);
    free(We8);
    free(Wz8);
    free(P);
}
