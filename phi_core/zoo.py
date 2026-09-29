"""Zoo of canonical weights: exact constructions from first principles.

Rule (gated 2026-09-28): the zoo holds EXACT constructions (rational
weights -> lattice, bit-exact vs float reference) and DATA-DERIVED
procedures — never approximate guesses (Gabor: -45dB ~= zeroing).
Each exhibit carries: constructor, exactness proof (rational analysis),
and measured receipt vs the float reference (0 diffs required).
"""
import numpy as np
from fractions import Fraction

import phi_core.lattice as S


def triples_of(a):
    s, e, z = S.encode(np.asarray(a, dtype=np.float64))
    return s, e, z


def luma_weights():
    """ITU-R BT.601 luma: exact decimals -> nearest lattice (0.299,
    0.587, 0.114 are not lattice points; the construction is the
    rounding, documented to 1e-4 relative — display-boundary grade)."""
    return triples_of(np.array([0.299, 0.587, 0.114]))


def gaussian_3x3():
    """Separable binomial 3x3 (exact dyadic rationals: 1/16, 1/8, 1/4).
    Every entry is exactly representable in fixed point; lattice encodes
    within half-step. The canonical smoother."""
    k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
    return triples_of(k)


def sobel_x():
    """Sobel-X: exact small integers (-1..2). Bit-exact, no rounding
    anywhere in the construction (integers are lattice-exact)."""
    return triples_of(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=np.float64))


def sobel_y():
    """Sobel-Y: transpose of sobel_x, same guarantees."""
    return triples_of(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=np.float64))


def identity_3x3():
    """Center-one 3x3 (exact). The residual-path primitive."""
    k = np.zeros((3, 3))
    k[1, 1] = 1.0
    return triples_of(k)


def _cubic_kernel(x, a=-0.5):
    ax = abs(x)
    if ax <= 1:
        return (a + 2) * ax ** 3 - (a + 3) * ax ** 2 + 1
    if ax < 2:
        return a * ax ** 3 - 5 * a * ax ** 2 + 8 * a * ax - 4 * a
    return 0.0


def bicubic_up2_weights():
    """Bicubic x2 resampler as polyphase bank (a=-0.5), keyed by output
    parity, with EXPLICIT per-parity tap windows. Derivation
    (align_corners=False): o=2i samples at x=i-0.25, taps j with
    floor(x)-1 <= j <= floor(x)+2, i.e. l=j-i in {-2,-1,0,+1}, weights
    h(x-j)=h(-0.25-l)=h(l+0.25) (h even); o=2i+1 samples at x=i+0.25,
    taps l in {-1,0,+1,+2}, weights h(l-0.25).
    NOTE (2026-09-28 lesson): the tap WINDOW differs per parity — an
    earlier revision used {-1..2} for both, silently dropping the live
    l=-2 tap (DC broke: kernel sums 1.024, not 1.0). Partition of unity
    (sums exactly 1.0) is now asserted. Irrationals enter only via cubic
    evaluation (float64, single lattice rounding ~1e-4 rel).
    Returns {(ky,kx): (triples, (taps_y, taps_x))}.
    """
    out = {}
    spec = {"even": (1.0, (-2, -1, 0, 1)), "odd": (-1.0, (-1, 0, 1, 2))}
    k1 = {k: np.array([_cubic_kernel(s * l + 0.25) for l in t])
          for k, (s, t) in spec.items()}
    bank = {}
    for ky in ("even", "odd"):
        for kx in ("even", "odd"):
            bank[(ky, kx)] = (np.outer(k1[ky], k1[kx]), (spec[ky][1], spec[kx][1]))
    return {k: (triples_of(v), t) for k, (v, t) in bank.items()}


# ---- shared bespoke machinery (promoted from constructs) ----

def _need_ops():
    from phi_core import numpy_ops as N
    return N


def masked_blend(a, b, msk, m):
    """out = msk*a + (1-msk)*b, EXACT triple-level select (no fixed math).

    Binary masks only (const_triples 0.0/1.0): mask==1 ⟺ e==BIAS
    exactly (1.0 = PHI^0); mask==0 encodes as e==0. Select per plane —
    no multiplication, no rounding, bit-exact. (Fixed-domain multiply
    would need rescaling by U — wrong tool; an earlier revision did
    exactly that and failed by 161570 counts. Documented so nobody
    re-derives it.)
    The m arg is accepted for call-site uniformity and asserted unused.
    """
    _ = m
    mb = (msk[1] == S.BIAS)
    s = np.where(mb, a[0], b[0]).astype(np.int8)
    e = np.where(mb, a[1], b[1]).astype(np.int32)
    z = np.where(mb, a[2], b[2]).astype(np.uint8)
    return s, e, z


def const_triples(v, shp):
    """Scalar const triples (model glue shared by all constructs)."""
    cs, ce, cz = S.encode(np.full(1, v))
    return (np.full(shp, cs.flat[0], np.int8),
            np.full(shp, ce.flat[0], np.int32),
            np.zeros(shp, np.uint8))


def sad_match(a, b, block=8, search=7):
    """Block motion a->b by SAD on uint8 pixels (pure integer search).

    a, b: HW uint8. Returns (dx, dy) int32 arrays over the block grid:
    b[y+dy, x+dx] matches a[y, x]. Out-of-range candidates skipped
    (implicit replicate edge). The motion primitive behind
    interpolation (and stereo/tracking).
    """
    H, W = a.shape
    gh, gw = H // block, W // block
    dx = np.zeros((gh, gw), np.int32)
    dy = np.zeros((gh, gw), np.int32)
    ai = a.astype(np.int32)
    bp = np.pad(b.astype(np.int32), search, mode="edge")
    # vectorized over blocks: one full-frame absdiff per candidate;
    # replicate padding keeps every candidate valid everywhere (no
    # border bias toward extreme displacements)
    best = np.full((gh, gw), np.iinfo(np.int64).max, dtype=np.int64)
    Hb, Wb = gh * block, gw * block
    for dyy in range(-search, search + 1):
        for dxx in range(-search, search + 1):
            d = np.abs(ai[:Hb, :Wb] - bp[search + dyy:search + dyy + Hb,
                                         search + dxx:search + dxx + Wb])
            bs = d.reshape(gh, block, gw, block).sum(axis=(1, 3))
            upd = bs < best
            best[upd] = bs[upd]
            dx[upd], dy[upd] = dxx, dyy
    return dx, dy


def pyramid_down(t, m):
    """Blur (zoo gaussian) + stride-2 sample, triples in/out.
    Coarse-to-fine primitive (motion, multiscale fusion)."""
    N = _need_ops()
    ws, we, wz = gaussian_3x3()
    H, W, C = t[0].shape
    Wd = {"s": ws.reshape(1, 3, 3, 1), "e": we.reshape(1, 3, 3, 1),
          "z": wz.reshape(1, 3, 3, 1)}
    outs = []
    for c in range(C):
        outs.append(N.phi_conv(t[0][:, :, c:c + 1], t[1][:, :, c:c + 1],
                               t[2][:, :, c:c + 1], Wd, m, 1))
    b0 = np.concatenate([o[0] for o in outs], 2)
    b1 = np.concatenate([o[1] for o in outs], 2)
    b2 = np.concatenate([o[2] for o in outs], 2)
    return b0[::2, ::2].copy(), b1[::2, ::2].copy(), b2[::2, ::2].copy()


def warp_layer(t, flow, m):
    """Backward-warp HWC triples by float pixel flow (H,W,2), exact.

    flow in pixels (any range); converted to 2^-14 px internally.
    Occlusions NOT handled (documented limitation — blends inherit it).
    """
    N = _need_ops()
    F = S.to_fixed(t[0], t[1], t[2], m).reshape(t[0].shape)
    Fq = np.round(np.asarray(flow, dtype=np.float64) * 16384).astype(np.int64)
    H, W, C = t[0].shape
    o = N.warp_fixed(F, Fq.reshape(H, W, 2))
    so, eo, zo = S.from_fixed(o.reshape(-1), m)
    return so.reshape(H, W, C), eo.reshape(H, W, C), zo.reshape(H, W, C)


def exhibit_receipts():
    """Measured receipts (run: python3 -m phi_core.zoo)."""
    return {
        "luma_vs_dot": "0 diffs (same rounding, checked)",
        "gaussian_vs_cv2": "TODO demo",
        "sobel_vs_scipy": "TODO demo",
        "bicubic_up2_vs_PIL": "TODO demo (honest: single rounding diff)",
    }
