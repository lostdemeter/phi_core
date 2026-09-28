// Shared device helpers (single implementation for all kernels).
// Bit-exact mirrors of the scalar bridge elementwise. FRAC/COARSE/FINE
// as int (values fit 32 bits; sums stay int64).
#pragma once
#include <stdint.h>

#define PHI_BIAS 32768
#define FRAC_CAP 13312

__device__ __forceinline__ int64_t to_fixed_elem(int ps, int pe, uint8_t pz,
                                                int m, int64_t base,
                                                const int *__restrict__ frac) {
    if (pz) return 0;
    int64_t d = (int64_t)m - (int64_t)pe;
    if (d < 0) return (int64_t)ps * base;
    if (d > FRAC_CAP) return 0;
    return (int64_t)ps * (int64_t)frac[d];
}

__device__ __forceinline__ void from_fixed_elem(int64_t q, int m,
                                               const int *__restrict__ coarse,
                                               const int *__restrict__ fine,
                                               int8_t *s, int *e, uint8_t *z) {
    *s = q > 0 ? 1 : -1;
    uint64_t a = q >= 0 ? (uint64_t)q : (uint64_t)(-(q + 1)) + 1u;
    int bl = 0;
    uint64_t x = a;
    if (x >> 32) { bl += 32; x >>= 32; }
    if (x >> 16) { bl += 16; x >>= 16; }
    if (x >> 8) { bl += 8; x >>= 8; }
    if (x >> 4) { bl += 4; x >>= 4; }
    if (x >> 2) { bl += 2; x >>= 2; }
    if (x >> 1) { bl += 1; }
    bl += (a != 0);
    int shift = bl - 15;
    uint64_t mant = shift >= 0 ? (shift > 0 ? (a >> shift) : a)
                               : (a << (-shift));
    int t = shift + 14 - 18;
    int idx = t + 64;
    if (idx < 0) idx = 0;
    if (idx > 192) idx = 192;
    int64_t fi = (int64_t)mant - 16384;
    if (fi < 0) fi = 0;
    if (fi > 16383) fi = 16383;
    int64_t ev = (int64_t)m + (int64_t)coarse[idx] + (int64_t)fine[fi];
    if (ev < 0) ev = 0;
    if (ev > 65535) ev = 65535;
    *e = (int)ev;
    *z = (q == 0);
}
