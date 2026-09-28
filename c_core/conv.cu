// CUDA phi-conv — explicit lowering specimen (see simd.h contract notes).
//
// IR node: conv3x3s1/s2 (HWC triples in, OHWI triples W, bias, m) ->
//   HWC triples out. Substrate: CUDA, thread per (o, co).
// Exactness: integer sums order-free -> any thread mapping valid;
//   lane/thread order recorded, not assumed (schedules-as-semantics).
// Precision: products int32 (range-proven: |q| <= 2^18*2.7 < 2^31 by
//   calibration margin — same invariant as the reverted int32 trial,
//   but here the SUM stays int64 and nothing is cast twice).
// Weights pre-transposed ONCE (model-frozen): [tap][ci][Co] co-minor,
//   so a warp (32 consecutive co) loads contiguously. LUTs in global
//   via __ldg (read-only cache; d clusters per layer).
#include "cuda_dev.cuh"

__global__ void phi_conv_cuda(
    const int8_t *__restrict__ in_s, const int *__restrict__ in_e,
    const uint8_t *__restrict__ in_z, int H, int Wd, int Cin,
    const int8_t *__restrict__ w_s, const int *__restrict__ w_e,
    const uint8_t *__restrict__ w_z, int Cout, int kh,
    const int8_t *__restrict__ b_s, const int *__restrict__ b_e,
    const uint8_t *__restrict__ b_z, int has_bias, int m_out, int stride,
    int pad,     int Ho, int Wo, const int *__restrict__ frac,
    const int *__restrict__ coarse, const int *__restrict__ fine,
    int8_t *__restrict__ out_s, int *__restrict__ out_e, uint8_t *__restrict__ out_z) {
    int64_t o = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Ho * Wo;
    if (o >= N || co >= Cout) return;
    int oy = (int)(o / Wo), ox = (int)(o % Wo);
    int64_t base = frac[0];
    int64_t acc = 0;
    if (has_bias) {
        int8_t bs = b_s[co];
        int be = b_e[co];
        uint8_t bz = b_z[co];
        if (!bz) acc = to_fixed_elem(bs, be, 0, m_out, base, frac);
    }
    // taps: [tap][ci][Co] co-minor transposed weights
    for (int ky = 0; ky < kh; ky++) {
        int iy = oy * stride + ky - pad;
        for (int kx = 0; kx < kh; kx++) {
            int ix = ox * stride + kx - pad;
            // bounds-check == zero-pad triples (s=0,e=0,z=0 -> contributes 0)
            for (int ci = 0; ci < Cin; ci++) {
                int8_t as = 0;
                int ae = 0;
                uint8_t az = 0;
                if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)Wd) {
                    int64_t ai = ((int64_t)iy * Wd + ix) * Cin + ci;
                    as = in_s[ai];
                    ae = in_e[ai];
                    az = in_z[ai];
                }
                int64_t wi = ((int64_t)(ky * kh + kx) * Cin + ci) * Cout + co;
                int8_t ws = w_s[wi];
                int we = w_e[wi];
                uint8_t wz = w_z[wi];
                int ps = (int)as * (int)ws;
                int pe = ae + we - PHI_BIAS;
                if (pe < 0) pe = 0;
                else if (pe > 65535) pe = 65535;
                uint8_t pz = az | wz;
                acc += to_fixed_elem(ps, pe, pz, m_out, base, frac);
            }
        }
    }
    // from_fixed scalar cascade (mirrors _from_fixed_inner elementwise)
    int64_t q = acc;
    int8_t s = q > 0 ? 1 : -1;
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
    uint64_t mant = shift >= 0 ? (a >> (shift > 0 ? shift : 0))
                               : (a << (-shift));
    int t = shift + 14 - 18;
    int idx = t + 64;
    if (idx < 0) idx = 0;
    if (idx > 192) idx = 192;
    int64_t fi = (int64_t)mant - 16384;
    if (fi < 0) fi = 0;
    if (fi > 16383) fi = 16383;
    int64_t e = (int64_t)m_out + (int64_t)__ldg(&coarse[idx])
        + (int64_t)__ldg(&fine[fi]);
    if (e < 0) e = 0;
    if (e > 65535) e = 65535;
    out_s[o * Cout + co] = s;
    out_e[o * Cout + co] = (int)e;
    out_z[o * Cout + co] = (q == 0);
}

// Fused concat+conv: input channels gathered on-the-fly from up to 5
// member blobs (no concat materialization in global memory). Member
// select is UNIFORM across each warp (same (tap,ci) for all lanes:
// threads differ only in co) -> no divergence. Bit-exact: identical
// products in a different assembly order (sums order-free).
// ms_*/me_*/mz_*: member base pointers (s,e,z planes); moff[i] = starting
// channel of member i in the concat; nmem members; Cin = total channels.
__global__ void phi_conv_fcat(
    const int8_t *__restrict__ ms0, const int *__restrict__ me0,
    const uint8_t *__restrict__ mz0, const int8_t *__restrict__ ms1,
    const int *__restrict__ me1, const uint8_t *__restrict__ mz1,
    const int8_t *__restrict__ ms2, const int *__restrict__ me2,
    const uint8_t *__restrict__ mz2, const int8_t *__restrict__ ms3,
    const int *__restrict__ me3, const uint8_t *__restrict__ mz3,
    const int8_t *__restrict__ ms4, const int *__restrict__ me4,
    const uint8_t *__restrict__ mz4, const int *moff, const int *mch, int nmem,
    int H, int Wd, int Cin,
    const int8_t *__restrict__ w_s, const int *__restrict__ w_e,
    const uint8_t *__restrict__ w_z, int Cout, int kh,
    const int8_t *__restrict__ b_s, const int *__restrict__ b_e,
    const uint8_t *__restrict__ b_z, int has_bias, int m_out, int stride,
    int pad, int Ho, int Wo, const int *__restrict__ frac,
    const int *__restrict__ coarse, const int *__restrict__ fine,
    int8_t *__restrict__ out_s, int *__restrict__ out_e,
    uint8_t *__restrict__ out_z) {
    int64_t o = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Ho * Wo;
    if (o >= N || co >= Cout) return;
    int oy = (int)(o / Wo), ox = (int)(o % Wo);
    int64_t base = frac[0];
    int64_t acc = 0;
    if (has_bias) {
        int8_t bs = b_s[co];
        int be = b_e[co];
        uint8_t bz = b_z[co];
        if (!bz) {
            int64_t d = (int64_t)m_out - (int64_t)be;
            acc = d < 0 ? (int64_t)bs * base
                : (d > FRAC_CAP ? 0 : (int64_t)bs * (int64_t)frac[d]);
        }
    }
    for (int ky = 0; ky < kh; ky++) {
        int iy = oy * stride + ky - pad;
        for (int kx = 0; kx < kh; kx++) {
            int ix = ox * stride + kx - pad;
            for (int ci = 0; ci < Cin; ci++) {
                int8_t as = 0;
                int ae = 0;
                uint8_t az = 0;
                if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)Wd) {
                    int64_t ai = 0;
                    // member select by channel (uniform per warp: same
                    // (tap,ci) for all lanes -> no divergence)
                    int mi = 0;
                    if (nmem > 1 && ci >= moff[1]) mi = 1;
                    if (nmem > 2 && ci >= moff[2]) mi = 2;
                    if (nmem > 3 && ci >= moff[3]) mi = 3;
                    if (nmem > 4 && ci >= moff[4]) mi = 4;
                    int c = ci - moff[mi];
                    int64_t px = (int64_t)iy * Wd + ix;
                    if (mi == 0) { ai = px * mch[0] + c; as = ms0[ai]; ae = me0[ai]; az = mz0[ai]; }
                    else if (mi == 1) { ai = px * mch[1] + c; as = ms1[ai]; ae = me1[ai]; az = mz1[ai]; }
                    else if (mi == 2) { ai = px * mch[2] + c; as = ms2[ai]; ae = me2[ai]; az = mz2[ai]; }
                    else if (mi == 3) { ai = px * mch[3] + c; as = ms3[ai]; ae = me3[ai]; az = mz3[ai]; }
                    else { ai = px * mch[4] + c; as = ms4[ai]; ae = me4[ai]; az = mz4[ai]; }
                }
                int64_t wi = ((int64_t)(ky * kh + kx) * Cin + ci) * Cout + co;
                int8_t ws = w_s[wi];
                int we = w_e[wi];
                uint8_t wz = w_z[wi];
                int ps = (int)as * (int)ws;
                int pe = ae + we - PHI_BIAS;
                if (pe < 0) pe = 0;
                else if (pe > 65535) pe = 65535;
                uint8_t pz = az | wz;
                if (!pz) {
                    int64_t d = (int64_t)m_out - (int64_t)pe;
                    acc += d < 0 ? (int64_t)ps * base
                        : (d > FRAC_CAP
                               ? 0
                               : (int64_t)ps * (int64_t)frac[d]);
                }
            }
        }
    }
    int8_t s;
    int e;
    uint8_t z;
    from_fixed_elem(acc, m_out, coarse, fine, &s, &e, &z);
    out_s[o * Cout + co] = s;
    out_e[o * Cout + co] = e;
    out_z[o * Cout + co] = z;
}
