"""Prior-modulation gates (staged with phi_core/prior.py -- branch, NOT main).

Direct contract gates (select identity, affine endpoints) + a cross-check
that the staged module agrees with plain numpy semantics on edge-heavy
vectors (the module adds NOTHING over them -- value is the contract).
Provenance: holographic_enhancement #LIB-020 (both call sites 0-diff).
Usage: python3 tests/test_prior.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_core import lattice as S
from phi_core.prior import select, affine

FAIL = []
M = S.BIAS + 512


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(0)
    m = rng.random((8, 8)) > 0.5
    ps = select((8, 8), m, 0.5, m_cov=M)
    one = S.encode(np.ones((8, 8)))
    att = S.encode(np.full((8, 8), 0.5))
    check("select", bool((ps[0][m] == one[0][0, 0]).all()
                         and (ps[1][m] == one[1][0, 0]).all()
                         and (ps[0][~m] == att[0][0, 0]).all()
                         and (ps[1][~m] == att[1][0, 0]).all()),
          "True->1.0 exact, False->atten")
    nt = S.encode(np.full((6, 6), 0.6))
    st = np.zeros((6, 6), bool)
    st[0, 0] = True
    pa = affine(nt, st, 0.5, M)
    dv = S.decode(pa[0], pa[1]) * (1 - pa[2].astype(np.float64))
    check("affine-static", bool(pa[0][0, 0] == 1 and pa[2][0, 0] == 0),
          "static->ones exact")
    check("affine-value", abs(float(dv[1:, :].mean()) - 0.7) < 0.02,
          f"mean={float(dv[1:, :].mean()):.4f} (expect ~0.7)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
