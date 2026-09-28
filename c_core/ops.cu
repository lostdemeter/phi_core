// Small-op elementwise kernels (thread per element; all bit-exact).
// Uses cuda_dev.cuh shared helpers (single implementation).
#include <stdint.h>
#include "cuda_dev.cuh"

// leaky: neg=(s<0)&~z ? tmul : passthrough. slope = scalar const triple.
__global__ void k_leaky(const int8_t *s, const int *e, const uint8_t *z,
                        int ss, int se, uint8_t sz, int n, int8_t *os, int *oe,
                        uint8_t *oz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int8_t a = s[i];
    int b = e[i];
    uint8_t c = z[i];
    if (((int)a < 0) && !c) {
        int ps = (int)a * ss;
        int pe = b + se - PHI_BIAS;
        if (pe < 0) pe = 0;
        else if (pe > 65535) pe = 65535;
        os[i] = (int8_t)ps;
        oe[i] = pe;
        oz[i] = (uint8_t)(c | sz);
    } else {
        os[i] = a;
        oe[i] = b;
        oz[i] = c;
    }
}

// fixed add at common scale m (mirrors _bin2 add at m_out)
__global__ void k_addfix(const int8_t *sa, const int *ea, const uint8_t *za,
                         const int8_t *sb, const int *eb, const uint8_t *zb,
                         int n, int m, int64_t base, const int *frac,
                         const int *coarse, const int *fine, int8_t *os, int *oe,
                         uint8_t *oz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t q = to_fixed_elem(sa[i], ea[i], za[i], m, base, frac)
        + to_fixed_elem(sb[i], eb[i], zb[i], m, base, frac);
    int8_t s;
    int e;
    uint8_t z;
    from_fixed_elem(q, m, coarse, fine, &s, &e, &z);
    os[i] = s;
    oe[i] = e;
    oz[i] = z;
}

// exact triple scale (tmul by scalar const)
__global__ void k_tmulsc(const int8_t *s, const int *e, const uint8_t *z,
                         int ss, int se, uint8_t sz, int n, int8_t *os, int *oe,
                         uint8_t *oz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    os[i] = (int8_t)((int)s[i] * ss);
    int pe = e[i] + se - PHI_BIAS;
    if (pe < 0) pe = 0;
    else if (pe > 65535) pe = 65535;
    oe[i] = pe;
    oz[i] = (uint8_t)(z[i] | sz);
}

// concat member copy (HWC, channel slice [c0,c0+cc) -> out at Coff)
__global__ void k_copych(const int8_t *s, const int *e, const uint8_t *z, int H,
                         int W, int Ci, int c0, int cc, int Co, int Coff,
                         int8_t *os, int *oe, uint8_t *oz) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = (int64_t)H * W * cc;
    if (t >= N) return;
    int c = (int)(t % cc);
    int64_t px = t / cc;
    int64_t si = px * Ci + c0 + c, di = px * Co + Coff + c;
    os[di] = s[si];
    oe[di] = e[si];
    oz[di] = z[si];
}

// nearest x2 on triples (pure reindex)
__global__ void k_nearest2(const int8_t *s, const int *e, const uint8_t *z,
                           int H, int W, int C, int8_t *os, int *oe,
                           uint8_t *oz) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = (int64_t)H * W * C;
    if (t >= N) return;
    int c = (int)(t % C);
    int64_t px = t / C;
    int y = (int)(px / W), x = (int)(px % W);
    int64_t si = (px * C) + c;
    int W2 = W * 2;
    int64_t d0 = (((int64_t)(2 * y) * W2 + 2 * x) * C) + c;
    int64_t d1 = d0 + C, d2 = d0 + (int64_t)W2 * C, d3 = d2 + C;
    os[d0] = os[d1] = os[d2] = os[d3] = s[si];
    oe[d0] = oe[d1] = oe[d2] = oe[d3] = e[si];
    oz[d0] = oz[d1] = oz[d2] = oz[d3] = z[si];
}
