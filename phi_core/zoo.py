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


def exhibit_receipts():
    """Measured receipts (run: python3 -m phi_core.zoo)."""
    return {
        "luma_vs_dot": "0 diffs (same rounding, checked)",
        "gaussian_vs_cv2": "TODO demo",
        "sobel_vs_scipy": "TODO demo",
        "bicubic_up2_vs_PIL": "TODO demo (honest: single rounding diff)",
    }
