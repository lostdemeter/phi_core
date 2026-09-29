"""Numpy substrate ops (extracted verbatim from the proven port).

Bit-exact mirrors live in c_core per substrate; this module is the
slow-exact reference. Gated 0-diff against the originals.
"""
"""Phase-1 integer datapath for rife-v2.3 (mirrors ref_exec op-for-op).

Representation: triples (s:int8, e:int32, z:uint8) between ops; fixed-point
int64 inside MACs/warps, via student_emu bridge (to_fixed/from_fixed/tdiv,
phi_conv). Two-scale calibration (M dict): m_acc covers max single PRODUCT
per conv/IP; m_cov covers max|value| wherever triples are DECODED
(match/warp/sigmoid inputs). Sigmoid is the SOLE bounded op (U=1 absolute,
span +-16, violations AUDITED); PReLU/Leaky are exact (scale-free).
Warp/interp share bilinear_sample_fixed (nihui-exact form, proven equal
to warp_int everywhere: out-of-range taps clamp to one pixel, weights sum
to 1 in both forms).
"""
import math
import os
import numpy as np

from phi_core import lattice as S

SIG_SPAN = 16 * 16384
_SIG = None
_SIGX = None


def sigx_lut():
    """EXPACT (exact triples -> 2^-14): round(lattice(e)*2^14) for every
    e in 0..65535, lattice(e) = PHI^((e-BIAS)/K). Lets sigmoid_int accept
    ANY pre-activation range with zero encode saturation (the old
    to_fixed-at-U1 head clipped |x|>1 to +-1). Same load-or-build frozen
    pattern as sig_lut/L_FRAC."""
    global _SIGX
    if _SIGX is None:
        p = "/home/thorin/Documents/OpenCode/new_target/rife_geo/generated/sigx_lut.npy"
        try:
            _SIGX = np.load(p)
        except Exception:
            _SIGX = np.round(np.power(
                S.PHI, (np.arange(65536, dtype=np.float64) - S.BIAS) / S.K
                ) * 16384).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, _SIGX)
            except Exception:
                pass
    return _SIGX


def sig_lut():
    """SIGMOID_LUT (mirrors gen_luts GELU pattern): round(sig(x)*2^14),
    x in 2^-14 over [-16,16]. Asymptotes exact (0 / 16384)."""
    global _SIG
    if _SIG is None:
        p = "/home/thorin/Documents/OpenCode/new_target/rife_geo/generated/sig_lut.npy"
        try:
            _SIG = np.load(p)
        except Exception:
            k = np.arange(2 * 16 * 16384 + 1, dtype=np.float64)
            x = (k - 16 * 16384) / 16384.0
            _SIG = np.round(1.0 / (1.0 + np.exp(-x)) * 16384).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, _SIG)
            except Exception:
                pass
    return _SIG


AUDIT = {"sig_clip": 0, "sig_span": 0, "sig_n": 0}


def tshape(t):
    return t[0].shape


def tmul(a, b):
    """Exact triple multiply (sign-XOR + exp-add). Zero-aware."""
    s = (a[0].astype(np.int16) * b[0].astype(np.int16)).astype(np.int8)
    e = np.clip(a[1].astype(np.int64) + b[1].astype(np.int64) - S.BIAS,
                0, S.NLVL - 1).astype(np.int32)
    z = (a[2] | b[2]).astype(np.uint8)
    return s, e, z


def const_triple(v, shape):
    return S.encode(np.full(shape, v, np.float64))


def sigmoid_int(t):
    """Sigmoid exact-to-lattice: x14 via EXPACT gather (no saturation at
    any range) -> LUT over span +-16, asymptotes outside. sig_clip is
    structurally 0 now (nothing CAN clip); sig_span counts LUT-range
    exits (still must read 0 for fidelity)."""
    X = sigx_lut()
    e = np.clip(t[1].astype(np.int64), 0, 65535)
    x14 = np.where(t[2].astype(bool), 0, t[0].astype(np.int64) * X[e])
    AUDIT["sig_n"] += int(x14.size)
    AUDIT["sig_span"] += int((np.abs(x14) > SIG_SPAN).sum())
    G = sig_lut()
    idx = x14 + SIG_SPAN
    y14 = np.where(x14 < -SIG_SPAN, 0, np.where(
        x14 > SIG_SPAN, 16384, G[np.clip(idx, 0, len(G) - 1)]))
    return S.from_fixed(y14 * np.int64(16), S.BIAS)


def prelu_int(t, slope):
    """PReLU exact: y = x if x>=0 (sign bit) else x*slope_triple."""
    neg = (t[0].astype(np.int16) < 0) & (~t[2].astype(bool))
    ps, pe, pz = tmul(t, slope)
    s = np.where(neg, ps, t[0])
    e = np.where(neg, pe, t[1]).astype(np.int32)
    z = np.where(neg, pz, t[2]).astype(np.uint8)
    return s, e, z


def bilinear_sample_fixed(F, X, Y):
    """F int64 (H,W,C); X,Y int64 coords in 2^-14 px. Integer bilinear
    with edge replicate. Returns int64 (same fixed unit as F)."""
    H, W, C = F.shape
    ONE = 1 << 14
    Xc = np.clip(X, 0, (W - 1) << 14)
    Yc = np.clip(Y, 0, (H - 1) << 14)
    xi, xf = Xc >> 14, Xc & (ONE - 1)
    yi, yf = Yc >> 14, Yc & (ONE - 1)
    xi1 = np.minimum(xi + 1, W - 1)
    yi1 = np.minimum(yi + 1, H - 1)
    g = lambda YY_, XX_: F[YY_, XX_, np.arange(C)[None, None, :]]
    tot = (g(yi[..., None], xi[..., None]) * (ONE - xf)[..., None] * (ONE - yf)[..., None]
           + g(yi[..., None], xi1[..., None]) * xf[..., None] * (ONE - yf)[..., None]
           + g(yi1[..., None], xi[..., None]) * (ONE - xf)[..., None] * yf[..., None]
           + g(yi1[..., None], xi1[..., None]) * xf[..., None] * yf[..., None]) >> 28
    return tot


def warp_fixed(feat_q, flow_q):
    """feat int64 (H,W,C) fixed; flow int64 (H,W,2) 2^-14. Samples at p + F
    (NIHUI convention, verified from warp.cpp source — NOTE opposite sign
    from softproj.warp_int's p - F! RIFE carries both signs explicitly via
    Neg ops, so callers pass flows as-is; do NOT pre-negate here)."""
    H, W, C = feat_q.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.int64), np.arange(H, dtype=np.int64))
    X = (gx << 14)[..., None] + flow_q[..., 0:1]
    Y = (gy << 14)[..., None] + flow_q[..., 1:2]
    return bilinear_sample_fixed(feat_q, X[..., 0], Y[..., 0])


def warp_triples_int(s, e, z, flow_q, m):
    F = S.to_fixed(s, e, z, m).reshape(s.shape)
    return S.from_fixed(warp_fixed(F, flow_q).reshape(-1), m)


def interp_fixed(feat_q, sy, sx):
    """Bilinear resample int64 fixed field. Sample coords in 2^-14 px,
    EXACT for all dyadic scales 2^-k / 2^k (RIFE uses 0.125/0.25/0.5/2/4):
      down 2^-k: X[o] = (2o+1)*2^(13+k) - 2^13   (shift, exact)
      up   2^k : X[o] = ((2o+1) - 2^k)*2^(13-k)  (shift, exact; negative
                 o=0 coords clip downstream = edge replicate, same as the
                 old hardcoded 2/4 cases, torch-verified).
    Non-dyadic scales fall back to truncating division (noted approx)."""
    H, W, C = feat_q.shape
    Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)

    def coord(n_out, s):
        o = np.arange(n_out, dtype=np.int64)
        for k in range(1, 14):
            if s == 2.0 ** -k:
                return ((2 * o + 1) << (13 + k)) - 8192
            if s == 2.0 ** k:
                return (((2 * o + 1) - (1 << k)) << (13 - k))
        if s == 1.0:
            return (o << 14) + 0 * o
        # generic fallback (truncation toward zero, noted)
        num = ((2 * o + 1) << 14)
        den = int(round(2 * s * 16384))
        q = np.abs(num) // den
        return np.where(num < 0, -q, q) - 8192

    XX, YY = np.meshgrid(coord(Wo, sx), coord(Ho, sy))
    return bilinear_sample_fixed(feat_q, XX, YY)


def avgpool_int(s, e, z, m):
    F = S.to_fixed(s, e, z, m).reshape(s.shape)
    H, W, C = F.shape
    tot = F.sum(axis=(0, 1))  # (C,) int64 exact
    mean = S.tdiv(tot, H * W)  # truncating (noted 1-LSB-class approximation)
    return S.from_fixed(mean, m)


def ip_int(s, e, z, Wd, m_out, Wb=None):
    """InnerProduct as dense MAC over raveled triples (+optional bias)."""
    N = int(np.prod(s.shape))
    acc = np.zeros((Wd["s"].shape[0],), np.int64)
    if Wb is not None:
        acc += S.to_fixed(Wb[0], Wb[1], Wb[2], m_out)
    fs, fe, fz = s.reshape(-1), e.reshape(-1), z.reshape(-1)
    for j in range(Wd["s"].shape[0]):
        ps = (fs.astype(np.int16) * Wd["s"][j].astype(np.int16)).astype(np.int8)
        pe = np.clip(fe.astype(np.int64) + Wd["e"][j].astype(np.int64) - S.BIAS,
                     0, S.NLVL - 1).astype(np.int32)
        pz = (fz | Wd["z"][j]).astype(np.uint8)
        acc[j] = S.to_fixed(ps, pe, pz, m_out).sum()
    return S.from_fixed(acc, m_out)


def deconv_int(s, e, z, Wd, m_out, stride=2, pad=1, Wb=None):
    """Transpose-conv via scatter-add over bridge (reference-slow).
    out[oy,ox] += W[ky,kx]*in[iy,ix] for oy = iy*s - pad + ky
    (verified vs torch conv_transpose2d exactly)."""
    H, W, Cin = s.shape
    Cout = Wd["s"].shape[0]
    kh = Wd["s"].shape[1]
    Ho, Wo = (H - 1) * stride - 2 * pad + kh, (W - 1) * stride - 2 * pad + kh
    acc = np.zeros((Ho, Wo, Cout), np.int64)
    # NOTE 2026-09-26: bias must ADD per-co AFTER the scatter (an earlier
    # form pre-added then overwrote per-co, silently dropping deconv bias;
    # float adds it, and RIFE deconv biases reach 0.29).
    bo = (S.to_fixed(Wb[0], Wb[1], Wb[2], m_out).reshape(-1)
          if Wb is not None else None)
    for co in range(Cout):
        A = np.zeros((Ho, Wo), np.int64)
        for ky in range(kh):
            for kx in range(kh):
                ys = ky - pad + np.arange(H) * stride
                xs = kx - pad + np.arange(W) * stride
                ok = (ys >= 0) & (ys < Ho)
                okx = (xs >= 0) & (xs < Wo)
                if not ok.any() or not okx.any():
                    continue
                sub = np.zeros((ok.sum(), okx.sum()), np.int64)
                ih = np.arange(H)[ok]
                iw = np.arange(W)[okx]
                oy = ky - pad + ih * stride
                ox = kx - pad + iw * stride
                for ci in range(Cin):
                    a_s = s[np.ix_(ih, iw, [ci])].reshape(-1)
                    a_e = e[np.ix_(ih, iw, [ci])].reshape(-1)
                    a_z = z[np.ix_(ih, iw, [ci])].reshape(-1)
                    wse = int(Wd["s"][co, ky, kx, ci])
                    wee = int(Wd["e"][co, ky, kx, ci])
                    wze = int(Wd["z"][co, ky, kx, ci])
                    ps_ = (a_s.astype(np.int16) * np.int16(wse)).astype(np.int8)
                    pe_ = np.clip(a_e.astype(np.int64) + wee - S.BIAS, 0, S.NLVL - 1).astype(np.int32)
                    pz_ = (a_z | np.int8(wze)).astype(np.uint8)
                    sub += S.to_fixed(ps_, pe_, pz_, m_out).reshape(ok.sum(), okx.sum())
                A[np.ix_(oy, ox)] += sub
        acc[:, :, co] = A + (bo[co] if bo is not None else 0)
    so, eo, zo = S.from_fixed(acc.reshape(-1), m_out)
    return so.reshape(Ho, Wo, Cout), eo.reshape(Ho, Wo, Cout), zo.reshape(Ho, Wo, Cout)


def nearest2(t):
    """Nearest x2 on triples (pure reindex, no math — exact by
    construction). Promoted from esrgan (generic op, all substrates)."""
    import numpy as np
    return (np.repeat(np.repeat(t[0], 2, 0), 2, 1),
            np.repeat(np.repeat(t[1], 2, 0), 2, 1),
            np.repeat(np.repeat(t[2], 2, 0), 2, 1))


# ---- conv (extracted verbatim; bare names bound to lattice below) ----
from phi_core.lattice import to_fixed as _tf, from_fixed as _ff
from phi_core.lattice import BIAS as _BIAS, NLVL as _NLVL
to_fixed, from_fixed, BIAS, NLVL = _tf, _ff, _BIAS, _NLVL
def im2col_idx(H, W, kh=3, stride=1, pad=1):
    Hp = H + 2 * pad
    yy, xx = np.meshgrid(np.arange(Hp), np.arange(Wp := W + 2 * pad), indexing="ij")
    oy, ox = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    Ho, Wo = oy.shape
    cols = []
    for ky in range(kh):
        for kx in range(kh):
            cols.append((oy + ky) * Wp + (ox + kx))
    return (Hp, Wp), np.stack(cols, axis=-1).reshape(-1, kh * kh), (Ho, Wo)


def _patch(Ap, Ho, Wo, stride, ky, kx):
    # NOTE 2026-09-26: columns need stride too (missing *stride broke all
    # stride-2 convs while s1 stayed exact; int+float shared the bug so M1
    # parity stayed green on wrong math — cross-validate against torch!).
    rr = (np.arange(Ho) * stride)[:, None] + np.zeros(Wo, np.int64)[None, :] + ky
    cc = (np.arange(Wo) * stride)[None, :] + np.zeros(Ho, np.int64)[:, None] + kx
    return Ap[rr, cc]


def phi_conv(s, e, z, W, m_out, stride=1, out_ch_chunk=8, Wb=None):
    """phi-conv with OPTIONAL bias triples Wb (RIFE port need; student has
    none). Bias added in fixed domain at m_out (exact, like any addend)."""
    """kh×kh phi-conv (kh=3 s1/s2 pad1, or kh=1). s,e,z: (H,W,Cin).
    W: dict(s,e,z) each (Cout,kh,kh,Cin). Returns triples (Ho,Wo,Cout).
    Bridge: products are absolute triples (XOR sign, e1+e2-BIAS);
    accumulated via to_fixed(., m_out) — m_out is the accumulator scale."""
    H, Wd, Cin = s.shape
    Cout, kh, _, CinW = W["s"].shape
    assert Cin == CinW
    pad = kh // 2 if kh == 3 else 0
    (Hp, Wp), _, _ = im2col_idx(H, Wd, kh, stride, pad)
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kh) // stride + 1
    N = Ho * Wo
    rH = (np.arange(Ho) * stride)[:, None]
    cW = (np.arange(Wo) * stride)[None, :]
    ps = np.pad(s, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    pe = np.pad(e.astype(np.int64), ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    pz = np.pad(z, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    taps = [(ky, kx, ci) for ky in range(kh) for kx in range(kh) for ci in range(Cin)]
    out_s = np.empty((N, Cout), np.int8)
    out_e = np.empty((N, Cout), np.int32)
    out_z = np.empty((N, Cout), np.uint8)
    for c0 in range(0, Cout, out_ch_chunk):
        c1 = min(Cout, c0 + out_ch_chunk)
        acc = np.zeros((N, c1 - c0), np.int64)
        if Wb is not None:
            # bias in fixed domain at m_out, broadcast over N pixels
            acc += to_fixed(Wb[0], Wb[1], Wb[2], m_out).reshape(1, -1)[:, c0:c1]
        for (ky, kx, ci) in taps:
            a_s = ps[rH + ky, cW + kx, ci].reshape(-1)
            a_e = pe[rH + ky, cW + kx, ci].reshape(-1)
            a_z = pz[rH + ky, cW + kx, ci].reshape(-1)
            for j, co in enumerate(range(c0, c1)):
                ws = int(W["s"][co, ky, kx, ci])
                we = int(W["e"][co, ky, kx, ci])
                wz = int(W["z"][co, ky, kx, ci])
                ps_ = (a_s.astype(np.int16) * ws).astype(np.int8)
                pe_ = np.clip(a_e + we - BIAS, 0, NLVL - 1).astype(np.int32)
                pz_ = (a_z | wz).astype(np.uint8)
                acc[:, j] += to_fixed(ps_, pe_, pz_, m_out)
        so, eo, zo = from_fixed(acc, m_out)
        out_s[:, c0:c1] = so.reshape(N, c1 - c0)
        out_e[:, c0:c1] = eo.reshape(N, c1 - c0)
        out_z[:, c0:c1] = zo.reshape(N, c1 - c0)
    return (out_s.reshape(Ho, Wo, Cout), out_e.reshape(Ho, Wo, Cout),
            out_z.reshape(Ho, Wo, Cout))


# NOTE on clamping: product exponents are clipped to [0,65535] before the
# bridge (C port clamps in phi_add/phi_head_predict the same way); m_out is
# the accumulator scale per M1-S2.


# ---- promoted shared ops (extracted verbatim from diffusion_reverse/
# unet_ops.py, proven by ITS S3 gates 14/14; third-copy trigger per the
# ops.py rule: diffusion built them, llama needs them, so they live HERE.
# Adaptations vs the original, both behavior-preserving: N.X references
# rebound to this module's own names; exp_lut bakes to this package's
# luts/ (same frozen auto-build pattern). Gated 0-diff in
# tests/test_promoted.py. gelu/geglu stay model-side (no trigger). ----

BOUND = 1 << 62  # intermediates must stay under this (4x int64 margin)


def _assert_bound(tag, *vals):
    mx = 0
    for v in vals:
        a = np.abs(np.asarray(v, dtype=np.int64))
        if a.size:
            mx = max(mx, int(a.max()))
    assert mx < BOUND, f"{tag}: intermediate {mx} >= 2^62"


def groupnorm_int(s, e, z, w, b, G, m, eps_c):
    """GroupNorm+affine, HWC triples in/out. w,b: triples (C,).
    eps_c: variance floor IN AMBIENT COUNTS
    (round(eps * 2^36 / U_m^2), precomputed offline in cal)."""
    H, W, C = s.shape
    assert C % G == 0
    q = S.to_fixed(s, e, z, m).astype(np.int64)
    wq = S.to_fixed(w[0], w[1], w[2], m).astype(np.int64)
    bq = S.to_fixed(b[0], b[1], b[2], m).astype(np.int64)
    n = (H * W * (C // G))
    out = np.empty_like(q)
    for g in range(G):
        sl = slice(g * (C // G), (g + 1) * (C // G))
        qq = q[:, :, sl].reshape(-1)
        mean = S.tdiv(np.sum(qq, dtype=np.int64), n)
        dev = qq - mean
        _assert_bound("gn:dev^2", dev)
        var = S.tdiv(np.sum(dev * dev, dtype=np.int64), n) + eps_c
        std = math.isqrt(int(var))  # var in 2^-36 -> std in 2^-18
        if std == 0:
            norm = np.zeros_like(dev)
        else:
            _assert_bound("gn:norm-num", dev)
            norm = S.tdiv(dev * (1 << 18), std)
        _assert_bound("gn:affine", norm)
        # qq is row-major (h,w,c): channel weights tile, not repeat.
        y = S.tdiv(norm * np.tile(wq[sl], H * W), (1 << 18))
        y = y + np.tile(bq[sl], H * W)
        out[:, :, sl] = y.reshape(H, W, -1)
    so, eo, zo = S.from_fixed(out.reshape(-1), m)
    sh = (H, W, C)
    return so.reshape(sh), eo.reshape(sh), zo.reshape(sh)


def int_layernorm_rows(s, e, z, w, b, m, eps_c):
    """Per-row LayerNorm+affine, (N,C) triples in/out. w,b: triples
    (C,). eps_c in ambient counts (see groupnorm_int)."""
    Nn, C = s.shape
    q = S.to_fixed(s, e, z, m).astype(np.int64)
    wq = S.to_fixed(w[0], w[1], w[2], m).astype(np.int64)
    bq = S.to_fixed(b[0], b[1], b[2], m).astype(np.int64)
    out = np.empty_like(q)
    for n in range(Nn):
        row = q[n]
        mean = S.tdiv(np.sum(row, dtype=np.int64), C)
        dev = row - mean
        var = S.tdiv(np.sum(dev * dev, dtype=np.int64), C) + eps_c
        std = math.isqrt(int(var))
        norm = np.zeros_like(dev) if std == 0 else S.tdiv(
            dev * (1 << 18), std)
        _assert_bound("ln:affine", norm)
        out[n] = S.tdiv(norm * wq, (1 << 18)) + bq
    so, eo, zo = S.from_fixed(out.reshape(-1), m)
    return so.reshape(Nn, C), eo.reshape(Nn, C), zo.reshape(Nn, C)


def matmul_int(A, B, m_acc, n_chunk=32):
    """Batched triples matmul: A (...,N,K) x B (...,K,M) -> (...,N,M).
    Per-product bridge at m_acc (phi_conv discipline); row-chunked."""
    As, Ae, Az = A
    Bs, Be, Bz = B
    Nn, K = As.shape[-2], As.shape[-1]
    M = Bs.shape[-1]
    assert Bs.shape[-2] == K
    lead = As.shape[:-2]
    A2 = (As.reshape(-1, Nn, K), Ae.reshape(-1, Nn, K), Az.reshape(-1, Nn, K))
    B2 = (Bs.reshape(-1, K, M), Be.reshape(-1, K, M), Bz.reshape(-1, K, M))
    Nb = A2[0].shape[0]
    assert B2[0].shape[0] in (1, Nb)
    outs, oes, ozs = [], [], []
    broadcast_b = B2[0].shape[0] == 1
    for bi in range(Nb):
        bb = tuple(x[0] if broadcast_b else x[bi] for x in B2)
        acc = np.zeros((Nn, M), np.int64)
        for i0 in range(0, Nn, n_chunk):
            i1 = min(Nn, i0 + n_chunk)
            nk = i1 - i0
            for k in range(K):
                a3 = tuple(np.broadcast_to(
                    x[bi, i0:i1, k].reshape(nk, 1), (nk, M)) for x in A2)
                b3 = tuple(np.broadcast_to(bb[j][k, :], (nk, M))
                           for j in range(3))
                pa_s = tmul(a3, b3)
                acc[i0:i1] += to_fixed(pa_s[0], pa_s[1], pa_s[2], m_acc)
        _assert_bound("matmul:acc", acc)
        so, eo, zo = S.from_fixed(acc.reshape(-1), m_acc)
        outs.append(so.reshape(Nn, M))
        oes.append(eo.reshape(Nn, M))
        ozs.append(zo.reshape(Nn, M))
    sh = lead + (Nn, M)
    return (np.stack(outs).reshape(sh).astype(np.int8),
            np.stack(oes).reshape(sh).astype(np.int32),
            np.stack(ozs).reshape(sh).astype(np.uint8))


_EXP_LUT = None


def exp_lut():
    """EXP_LUT[d] = round(2^24 * exp(-d/2^14)), d in [0, 262144].
    Same formula as the proven softproj 2-way softmax."""
    global _EXP_LUT
    if _EXP_LUT is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "luts", "exp_lut.npy")
        try:
            _EXP_LUT = np.load(p)
        except Exception:
            d = np.arange(262145, dtype=np.float64)
            _EXP_LUT = np.round((2.0 ** 24) * np.exp(-d / 16384.0)
                                ).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, _EXP_LUT)
            except Exception:
                pass
    return _EXP_LUT


def rescale_via_triples(q, m_from, m_to):
    """The IR `rescale` op (only scale changer): fixed@m_from ->
    triples -> fixed@m_to. Exact up to lattice quantum."""
    s, e, z = S.from_fixed(q.reshape(-1), m_from)
    return S.to_fixed(s, e, z, m_to).reshape(q.shape)


def softmaxN_fixed(q, m):
    """Stable N-way softmax over last axis of fixed scores @ m.
    Scores must be ABSOLUTE (nonlinear — see M1 finding #4); callers
    pass BIAS-scale counts, asserted here. Returns (num 2^24, den)."""
    assert m == S.BIAS, f"softmax needs BIAS-scale counts, got m={m}"
    s14 = S.tdiv(q.astype(np.int64), 16)  # 2^-18 -> 2^-14 (M5 contract)
    vmax = s14.max(axis=-1, keepdims=True)
    dd = np.clip(vmax - s14, 0, 262144).astype(np.int64)
    num = exp_lut()[dd]
    den = np.sum(num, axis=-1, dtype=np.int64)
    den = np.where(den == 0, 1, den)
    _assert_bound("softmax:den", den)
    return num, den


def softmaxN_triples(t):
    """N-way softmax from triples: bridge at BIAS (absolute) -> LUT."""
    s, e, z = t
    q = S.to_fixed(s, e, z, S.BIAS)
    return softmaxN_fixed(q.reshape(s.shape), S.BIAS)


def silu_int(t):
    """SiLU x*sigmoid(x): sigmoid_int (any range) + exact tmul."""
    return tmul(sigmoid_int(t), t)
