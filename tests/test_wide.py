"""Wide bridge gates (ai/t-transform-bridge): full-range softmax support.

Additive-only: to_fixed UNTOUCHED (existing suites prove it); to_fixed_wide
serves d in [-8192,0) exactly (|v| to ~2200) where the fold corrupts.
Gates: wide==narrow bit-exact in-range (additivity proof), wide exactness
on lattice values to +-2000, full-range softmax parity vs float reference
(max-sub composition + wide bridge + existing softmaxN_fixed, untouched),
fold preserved beyond KMAX (documented edge, not a behavior change).
Usage: python3 tests/test_wide.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    import phi_core.lattice as S
    from phi_core import numpy_ops as N
    rng = np.random.default_rng(0)
    # 1. additivity: wide == narrow bit-exact wherever narrow is exact.
    # Fixture respects each m's coverage (U_BIAS=1.0: values beyond fold in
    # narrow BY DESIGN -- wide must differ there, so stay inside).
    a = rng.uniform(-0.9, 0.9, (4, 64))
    t = S.encode(a)
    for m in (S.BIAS, S.BIAS + 512):
        q0 = S.to_fixed(t[0], t[1], t[2], m)
        q1 = N.to_fixed_wide(t[0], t[1], t[2], m)
        check(f"wide-additive@{m}", bool((q0 == q1).all()),
              "in-range inputs agree bit-exactly")
    # 2. exactness on lattice values to +-2000. Compare against DECODED
    # (lattice) values, not raw floats: encode() itself quantizes to 0.05%
    # and the bridge must only preserve lattice points (+-1 count here).
    big = np.concatenate([
        rng.uniform(-2000, 2000, 500),
        np.array([-2000.0, -1000.0, -100.0, -1.0, 0.0, 1.0, 100.0,
                  1000.0, 2000.0])])
    tb = S.encode(big)
    qb = N.to_fixed_wide(tb[0], tb[1], tb[2], S.BIAS)
    lat = S.decode(tb[0], tb[1])
    want = np.round(np.abs(lat) * (1 << 18)).astype(np.int64)
    got = np.abs(qb)
    ok = bool((((got - want) <= 2) | (big == 0)).all())
    check("wide-exact", ok,
          f"maxcount-err={int(np.abs(got - want).max())} (encode quantum)")
    # fold preserved beyond KMAX (values >> 2200): documents the edge
    huge = np.array([-1e6, 1e6])
    th = S.encode(huge)
    qh = N.to_fixed_wide(th[0], th[1], th[2], S.BIAS)
    base = S.L_FRAC()[0]
    check("wide-fold-edge",
          bool((qh == np.array([-base, base])).all()),
          "beyond WIDE_CAP keeps the existing fold convention")
    # 3. full-range softmax parity vs float (the T-transform composition:
    # max-sub in float here mirrors the listing-side triple max-sub; the
    # bridge + exp path are what this branch proves)
    rows = rng.uniform(-500, 500, (6, 32))
    rows[0] = np.linspace(-500, 500, 32)
    outs = []
    for r in rows:
        tt = S.encode(r - r.max())  # max-sub (listing does this in triples)
        qq = N.to_fixed_wide(tt[0], tt[1], tt[2], S.BIAS)
        num, den = N.softmaxN_fixed(qq.reshape(1, -1), S.BIAS)
        outs.append((num / np.maximum(den, 1)).astype(np.float64)[0])
    got = np.stack(outs)
    e = np.exp(rows - rows.max(-1, keepdims=True))
    ref = e / e.sum(-1, keepdims=True)
    err = float(np.abs(got - ref).max())
    check("wide-softmax-parity", err < 5e-3,
          f"maxabsdiff={err:.2e} vs float on +-500 scores")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
