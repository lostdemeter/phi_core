"""phi-core lattice gates: 0-diff vs the originals (extraction proof).

Each test compares this package against new_target/student_emu on
edge-heavy vectors. Any diff = extraction damage (the port adds
NOTHING). Usage: python3 tests/test_lattice.py (needs NEW_TARGET env
or ../new_target checkout beside phi-core).
Usage: NEW_TARGET=/path python3 tests/test_lattice.py
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
    import phi_core.lattice as L
    nt = os.environ.get("NEW_TARGET", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "new_target"))
    sys.path.insert(0, nt)
    import student_emu as S
    rng = np.random.default_rng(0)
    # encode/decode/to_fixed/from_fixed/tdiv/bit_length, edge-heavy
    x = np.concatenate([rng.uniform(-30, 30, 3000), [0., 1., -1., 1e-15, 1e6]])
    s, e, z = S.encode(x)
    for fn in ("encode",):
        a = getattr(L, fn)(x)
        b = getattr(S, fn)(x)
        check(f"{fn}", all(bool((i == j).all()) for i, j in zip(a, b)))
    check("decode", bool((L.decode(s, e) == S.decode(s, e)).all()))
    for m in (30000, 32768, 34000):
        check(f"to_fixed@{m}", bool((L.to_fixed(s, e, z, m) == S.to_fixed(s, e, z, m)).all()))
    q = rng.integers(-(1 << 50), 1 << 50, size=4000).astype(np.int64)
    q = np.concatenate([q, np.array([0, 1, -1, (1 << 52) - 1, 1 << 40], dtype=np.int64)])
    for m in (32000,):
        a = L.from_fixed(q, m)
        b = S.from_fixed(q, m)
        check(f"from_fixed@{m}", all(bool((i == j).all()) for i, j in zip(a, b)))
    a = rng.integers(-10**6, 10**6, size=2000).astype(np.int64)
    b = rng.integers(-1000, 1000, size=2000).astype(np.int64)
    b[b == 0] = 7
    check("tdiv", bool((L.tdiv(a, b) == S.tdiv(a, b)).all()))
    edge = np.array([0, 1, 2, 2**15, 2**31 - 1, 2**32, 2**52, 2**63 - 1], dtype=np.int64)
    check("bit_length", bool((L.bit_length_int(edge) == S.bit_length_int(edge)).all()))
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
