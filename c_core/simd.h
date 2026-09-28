/* SIMD bridge lowerings (AVX512F+BW+VL+DQ) — explicit lowering specimens.
 *
 * LOWERING CONTRACT (for the future generator): each primitive states
 * its IR node, substrate, and exactness proof. All are bit-identical to
 * the scalar bridge (order-free integer math; gathers are pure
 * functions of indices). Schedule note (schedules as semantics):
 * lanes are independent by construction — any lane permutation is
 * valid; recorded here explicitly rather than assumed.
 *
 * IR node: to_fixed(s[8], e[8], z[8], m) -> q[8]  (2^-18 @ U(m))
 * Substrate: AVX512F (qword arith) + BW (byte extend) + VL (dword
 * gather) + DQ (qword multiply). Compile: -mavx512f -mavx512bw
 * -mavx512vl -mavx512dq.
 */
#pragma once
#include <immintrin.h>
#include <stdint.h>

#include "phi_types.h"

/* Loads: 8 int8 -> __m128i; 8 int32 -> __m256i (caller-side, documented
 * so lane mapping is unambiguous: lane i = element i, low to high). */
static inline __m128i load8_u8(const void *p) {
    return _mm_loadu_si64(p);
}
static inline __m256i load8_u32(const void *p) {
    return _mm256_loadu_si256((const __m256i *)p);
}

/* 8-wide bridge: product triples -> fixed counts. Mirrors the scalar
 * to_fixed elementwise (d<0 -> s*base; d>FRAC_CAP -> 0 [or z];
 * else s*FRAC[d]). z forces 0. Masked gather touches only lanes 0-7
 * (upper index lanes never reach memory — no OOB by construction). */
static inline __m512i bridge8(__m128i ps8, __m256i pe32, __m128i pz8, int m,
                              int64_t base, const int64_t *frac) {
    __m512i s64 = _mm512_cvtepi8_epi64(ps8);
    __m512i ee = _mm512_cvtepi32_epi64(pe32);
    __m512i d = _mm512_sub_epi64(_mm512_set1_epi64((long long)m), ee);
    __m512i qsat = _mm512_mullo_epi64(s64, _mm512_set1_epi64((long long)base));
    __m512i dc = _mm512_max_epi64(
        _mm512_min_epi64(d, _mm512_set1_epi64((long long)FRAC_CAP)),
        _mm512_set1_epi64((long long)0));
    __m256i idx8 = _mm512_cvtepi64_epi32(dc);
    /* int64 table read as low32: scale=8 strides int64 entries while
     * reading 4 bytes each (values < 2^21, high halves zero — exact). */
    __m256i g32 = _mm256_mask_i32gather_epi32(
        _mm256_setzero_si256(), (const int *)frac, idx8, _mm256_set1_epi32(-1), 8);
    __m512i qg = _mm512_mullo_epi64(s64, _mm512_cvtepi32_epi64(g32));
    __mmask8 m_neg = _mm512_cmplt_epi64_mask(d, _mm512_setzero_si512());
    __mmask8 m_big =
        _mm512_cmpgt_epi64_mask(d, _mm512_set1_epi64((long long)FRAC_CAP));
    __mmask16 mz16 = _mm512_cmpneq_epi32_mask(_mm512_cvtepi8_epi32(pz8),
                                              _mm512_setzero_si512());
    __mmask8 m_z = (__mmask8)(mz16 & 0xFF);
    __m512i q = _mm512_mask_blend_epi64(m_neg, qg, qsat);
    q = _mm512_mask_blend_epi64(m_big, q, _mm512_setzero_si512());
    q = _mm512_mask_blend_epi64(m_z, q, _mm512_setzero_si512());
    return q;
}

/* 8-wide integer bit_length (0 for a==0). Mirrors bit_length_int.
 * Variable shifts (srlv) — no immediates, fully unrolled-safe. */static inline __m512i bitlen8(__m512i a) {
    __m512i bl = _mm512_setzero_si512();
    __m512i x = a;
    const int SH[6] = {32, 16, 8, 4, 2, 1};
    for (int i = 0; i < 6; i++) {
        __m512i cnt = _mm512_set1_epi64((long long)SH[i]);
        __mmask8 big =
            _mm512_cmpgt_epi64_mask(_mm512_srlv_epi64(x, cnt), _mm512_setzero_si512());
        bl = _mm512_mask_add_epi64(bl, big, bl, cnt);
        x = _mm512_mask_srlv_epi64(x, big, x, cnt);
    }
    __mmask8 nz = _mm512_cmpneq_epi64_mask(a, _mm512_setzero_si512());
    return _mm512_mask_add_epi64(bl, nz, bl, _mm512_set1_epi64((long long)1));
}

/* 8-wide from_fixed: acc[8] @ scale m -> (s,e,z)[8]. Mirrors
 * _from_fixed_inner elementwise (gathers pure in their indices).
 * Stores: s = low 8 bytes, e = 8 int32, z = 8 bytes 0/1. */
static inline void from_fixed8(__m512i q, int m, const int64_t *coarse,
                               const int64_t *fine, int8_t *s, int32_t *e,
                               uint8_t *z) {
    __m512i one = _mm512_set1_epi64((long long)1);
    /* scalar: s = q>0 ? +1 : -1 (q==0 -> -1 with z=1). blend(mask, a, b)
     * picks a where mask set: mask must be (q>0). */
    __m512i sv = _mm512_mask_blend_epi64(
        _mm512_cmpgt_epi64_mask(q, _mm512_setzero_si512()),
        _mm512_set1_epi64((long long)-1), one);
    __m512i a = _mm512_abs_epi64(q);
    __m512i bl = bitlen8(a);
    __m512i shift = _mm512_sub_epi64(bl, _mm512_set1_epi64((long long)15));
    __mmask8 nonneg = _mm512_cmpge_epi64_mask(shift, _mm512_setzero_si512());
    __m512i mant = _mm512_mask_blend_epi64(
        nonneg, _mm512_sllv_epi64(a, _mm512_sub_epi64(_mm512_setzero_si512(), shift)),
        _mm512_srlv_epi64(a, shift));
    __m512i t = _mm512_add_epi64(shift, _mm512_set1_epi64((long long)(14 - FIXED_F)));
    __m512i idx = _mm512_add_epi64(t, _mm512_set1_epi64((long long)64));
    __m512i idc = _mm512_max_epi64(_mm512_min_epi64(idx, _mm512_set1_epi64((long long)192)),
                                   _mm512_setzero_si512());
    __m256i ci32 = _mm512_cvtepi64_epi32(idc);
    __m256i cg = _mm256_mask_i32gather_epi32(
        _mm256_setzero_si256(), (const int *)coarse, ci32, _mm256_set1_epi32(-1), 8);
    __m512i mantoff = _mm512_sub_epi64(
        mant, _mm512_set1_epi64((long long)16384));
    __m512i fi = _mm512_max_epi64(
        _mm512_min_epi64(mantoff, _mm512_set1_epi64((long long)16383)),
        _mm512_setzero_si512());
    __m256i fi32 = _mm512_cvtepi64_epi32(fi);
    __m256i fg = _mm256_mask_i32gather_epi32(
        _mm256_setzero_si256(), (const int *)fine, fi32, _mm256_set1_epi32(-1), 8);
    __m512i ev = _mm512_add_epi64(
        _mm512_set1_epi64((long long)m),
        _mm512_add_epi64(_mm512_cvtepi32_epi64(cg), _mm512_cvtepi32_epi64(fg)));
    ev = _mm512_max_epi64(_mm512_min_epi64(ev, _mm512_set1_epi64((long long)65535)),
                          _mm512_setzero_si512());
    __m128i s8 = _mm512_cvtepi64_epi8(sv);
    _mm_storel_epi64((__m128i *)s, s8);
    _mm256_storeu_si256((__m256i *)e, _mm512_cvtepi64_epi32(ev));
    __mmask8 zm = _mm512_cmpeq_epi64_mask(q, _mm512_setzero_si512());
    __m512i zv = _mm512_maskz_set1_epi8((__mmask64)zm, 1);
    *(uint64_t *)z = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(zv));
}
