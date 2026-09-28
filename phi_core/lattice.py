"""phi lattice core — the shared integer substrate (extracted verbatim).

Provenance: ported function-for-function from the three geometric ports
(DAV2 student_emu, RIFE/E SRGAN phi_core). Every function here is gated
0-diff against the originals (tests/test_lattice.py); the extraction
adds NOTHING except portable paths and this docstring. New code goes
in new modules, never by editing these mirrors.

A phi-value is sign * PHI^((exponent - 32768) / 512), K=512. Runtime
needs no FPU: int add/sub/compare/shift/XOR + LUT gather only.
Float appears ONLY in: encode (sensor boundary), LUT builds (offline,
frozen to luts/), decode (display/scoring). See fpu_trap pattern.
"""
import math
import os
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
LN_PHI = math.log(PHI)
K, BIAS, NLVL = 512, 32768, 65536
DMAX, FRAC_CAP, FIXED_F = 4096, 13312, 18
SIG_SPAN = 16 * 16384

LUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "luts")

_t = {}


def _load(name, build):
    if name not in _t:
        p = os.path.join(LUTDIR, name + ".npy")
        try:
            _t[name] = np.load(p)
        except Exception:
            _t[name] = build()
            try:
                os.makedirs(LUTDIR, exist_ok=True)
                np.save(p, _t[name])
            except Exception:
                pass
    return _t[name]


def L_FRAC():
    return _load("frac_lut", lambda: np.round(
        np.power(PHI, -np.arange(13313) / K) * 2 ** 18).astype(np.int64))


def L_COARSE():
    return _load("coarse_lut", lambda: np.round(
        np.arange(-64, 129) * K * math.log(2) / LN_PHI).astype(np.int64))


def L_FINE():
    return _load("fine_lut", lambda: np.round(
        K * np.log((16384 + np.arange(16384)) / 16384) / LN_PHI).astype(np.int64))


def sig_lut():
    def build():
        k = np.arange(2 * 16 * 16384 + 1, dtype=np.float64)
        x = (k - 16 * 16384) / 16384.0
        return np.round(1.0 / (1.0 + np.exp(-x)) * 16384).astype(np.int64)
    return _load("sig_lut", build)


def sigx_lut():
    def build():
        return np.round(np.power(
            PHI, (np.arange(65536, dtype=np.float64) - BIAS) / K
            ) * 16384).astype(np.int64)
    return _load("sigx_lut", build)


def encode(a):
    a = np.asarray(a, dtype=np.float64)
    s = np.sign(a).astype(np.int8)
    s[s == 0] = 1
    e = np.round(K * np.log(np.abs(a) + 1e-15) / LN_PHI).astype(np.int64) + BIAS
    return s, np.clip(e, 0, NLVL - 1).astype(np.int32), np.zeros(a.shape, np.uint8)


def decode(s, e):
    return s.astype(np.float64) * np.power(PHI, (e.astype(np.float64) - BIAS) / K)


def to_fixed(s, e, z, m):
    """Port of phi_to_fixed: (s,e,z) at scale m -> int64 2^-18 units."""
    d = np.asarray(m, dtype=np.int64) - e.astype(np.int64)
    F = L_FRAC()
    base = F[0]
    out = np.where(z.astype(bool), 0, np.where(
        d < 0, s.astype(np.int64) * base, np.where(
            d > FRAC_CAP, 0, s.astype(np.int64) * F[np.clip(d, 0, FRAC_CAP)])))
    return out.astype(np.int64)


def from_fixed(q, m):
    """Port of phi_from_fixed(q, m, f=18). q int64 2^-18 -> triples."""
    q = np.asarray(q, dtype=np.int64)
    z = (q == 0)
    a = np.abs(q)
    return _from_fixed_inner(q, z, a, m)


def bit_length_int(a):
    """Integer bit_length (floor(log2)+1, 0 for a=0) — pure integer."""
    a = np.asarray(a, dtype=np.int64)
    bl = np.zeros(a.shape, np.int64)
    x = a
    for sh in (32, 16, 8, 4, 2, 1):
        big = (x >> sh) > 0
        bl = bl + big.astype(np.int64) * sh
        x = np.where(big, x >> sh, x)
    return bl + (a > 0).astype(np.int64)


def _from_fixed_inner(q, z, a, m):
    s = np.where(q > 0, 1, -1).astype(np.int8)
    assert int(a.max(initial=0)) < 2 ** 53, "accum exceeds frexp-exact range"
    bl = bit_length_int(a)
    nz = a > 0
    shift = bl - 15
    mant = np.where(shift >= 0, a >> np.maximum(shift, 0),
                    a << np.maximum(-shift, 0))
    t = shift + 14 - FIXED_F
    idx = t + 64
    assert int(idx[nz].min(initial=0)) >= 0 and int(idx[nz].max(initial=0)) <= 192, \
        f"COARSE index OOB (t+64 range violated): min={idx[nz].min(initial=0)} max={idx[nz].max(initial=0)}"
    C, FN = L_COARSE(), L_FINE()
    e = np.asarray(m, dtype=np.int64) + C[np.clip(idx, 0, 192)] + FN[np.clip(mant - 16384, 0, 16383)]
    e = np.clip(e, 0, 65535).astype(np.int32)
    return s, e, z.astype(np.uint8)


def tdiv(a, b):
    # True truncation toward zero, BOTH signs (C semantics).
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q)
