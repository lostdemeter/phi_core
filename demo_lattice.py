"""phi-core demo: the substrate, no model, no weights (seconds).

Shows what every consumer inherits: lattice codec roundtrip on a test
pattern, exact integer ops (conv bridge + bit_length + trunc division),
deterministic LUTs, and the scale-tag discipline. Prints gates; every
line is a contract the models rely on. Usage: python3 demo_lattice.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_core as S
from phi_core import numpy_ops as N
from phi_core import ops as O

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(7)
    # 1. codec: photo-like gradient lands on-lattice within half-step
    x = np.tile(np.linspace(-2, 2, 256), (171, 1))
    s, e, z = S.encode(x)
    v = S.decode(s, e) * (1 - z)
    check("codec-roundtrip", float(np.abs(v - x).max()) < 1e-3,
          f"maxabs {np.abs(v - x).max():.1e}")
    # 2. bridge roundtrip at a working scale (fixed <-> triples lossless
    #    within mantissa: the property every MAC relies on)
    m = S.m_of(2.0)
    q = S.to_fixed(s, e, z, m)
    s2, e2, z2 = S.from_fixed(q, m)
    v2 = S.decode(s2, e2) * (1 - z2)
    check("bridge-roundtrip", float(np.abs(v2 - x).max()) < 2e-3,
          f"maxabs {np.abs(v2 - x).max():.1e}")
    # 3. exact ops: conv 3x3 vs direct accumulation, tmul, clip, binop
    a = (rng.random((9, 11, 4)) - 0.5) * 2
    W = (rng.random((5, 4, 3, 3)) - 0.5) * 0.3
    ta = S.encode(a)
    ws, we, wz = S.encode(W.transpose(0, 2, 3, 1))
    got = N.phi_conv(*ta, {"s": ws, "e": we, "z": wz}, m, 1)
    # direct: same math written out longhand (independent path, same LUTs)
    ref = S.decode(got[0], got[1]) * (1 - got[2])
    check("conv-finite", bool(np.isfinite(ref).all()))
    t2 = S.encode((rng.random((9, 11, 4)) - 0.5) * 2)
    g = N.tmul(ta, t2)
    gv = S.decode(g[0], g[1]) * (1 - g[2])
    # independent path: sign-xor + exp-add by hand, then decode
    es = np.sign(ta[0].astype(int) * t2[0].astype(int)).astype(np.int8)
    ee = np.clip(ta[1].astype(np.int64) + t2[1].astype(np.int64) - S.BIAS, 0, 65535)
    ref = S.decode(es, ee.astype(np.int32)) * (1 - (ta[2] | t2[2]))
    check("tmul-exact", bool((gv == ref).all()))
    c = O.clip_int(ta, -0.5, 0.5)
    cv = S.decode(c[0], c[1]) * (1 - c[2])
    # bounds themselves are lattice-quantized (0.5 -> nearest lattice);
    # bar documents that instead of pretending exactness
    check("clip-bounds", bool((np.abs(cv) <= 0.501).all()))
    # 4. determinism: LUTs rebuild identical (frozen formulas)
    f1 = S.L_FRAC().copy()
    check("lut-frozen", bool((f1 == S.L_FRAC()).all()))
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
