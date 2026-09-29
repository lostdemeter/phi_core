"""Promoted-ops extraction proof: 0-diff vs diffusion_reverse/unet_ops.

Each test runs this package against the proven originals on edge-heavy
vectors. Any diff = extraction damage (the promotion adds NOTHING).
Adaptations (N.X rebinding, exp_lut bake path) are behavior-preserving
by construction AND verified here.
Usage: python3 tests/test_promoted.py (needs DIFFUSION env or
../diffusion_reverse checkout beside phi-core).
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


def triples(S, a):
    s, e, z = S.encode(np.asarray(a, dtype=np.float64))
    return s, e, z


def main():
    import phi_core.lattice as L
    import phi_core.numpy_ops as N
    df = os.environ.get("DIFFUSION", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
        "diffusion_reverse"))
    sys.path.insert(0, df)
    import unet_ops as O
    rng = np.random.default_rng(4)

    # groupnorm: HWC small, G=4 + G=1, edge values
    x = np.concatenate([((rng.random(157) - 0.5) * 6),
                        [0., 10., -10.]]).reshape(4, 5, 8)
    w = (rng.random(8) - 0.5) * 2
    b = (rng.random(8) - 0.5) * 2
    from phi_core.calibrate import m_of
    m = m_of(10.0)
    ec = 4514
    for gg in (4, 1):
        a = N.groupnorm_int(*triples(L, x), triples(L, w), triples(L, b),
                            gg, m, ec)
        c = O.groupnorm_int(*triples(L, x), triples(L, w), triples(L, b),
                            gg, m, ec)
        check(f"groupnorm-G{gg}", all(
            bool((i == j).all()) for i, j in zip(a, c)))

    # layernorm rows
    xr = (rng.random((7, 32)) - 0.5) * 6
    wr = (rng.random(32) - 0.5) * 2
    br = (rng.random(32) - 0.5) * 2
    a = N.int_layernorm_rows(*triples(L, xr), triples(L, wr),
                             triples(L, br), m, ec)
    c = O.int_layernorm_rows(*triples(L, xr), triples(L, wr),
                             triples(L, br), m, ec)
    check("layernorm-rows", all(bool((i == j).all()) for i, j in zip(a, c)))

    # matmul: batched, k1 + rectangular
    A = (rng.random((2, 5, 9)) - 0.5) * 3
    B = (rng.random((2, 9, 4)) - 0.5) * 3
    ma = m_of(4.5)
    a = N.matmul_int(triples(L, A), triples(L, B), ma)
    c = O.matmul_int(triples(L, A), triples(L, B), ma)
    check("matmul", all(bool((i == j).all()) for i, j in zip(a, c)))

    # softmaxN: rows incl. ties, spikes, N=77
    q = ((rng.random((3, 77)) - 0.5) * 8 * 262144).astype(np.int64)
    a = N.softmaxN_fixed(q, L.BIAS)
    c = O.softmaxN_fixed(q, L.BIAS)
    check("softmaxN", all(bool((i == j).all()) for i, j in zip(a, c)))

    # rescale roundtrip
    q2 = (rng.random((4, 9)) * 1000).astype(np.int64) - 500
    a = N.rescale_via_triples(q2, m, L.BIAS)
    c = O.rescale_via_triples(q2, m, L.BIAS)
    check("rescale", bool((a == c).all()))

    # silu
    xs = (rng.random((5, 6)) - 0.5) * 10
    a = N.silu_int(triples(L, xs))
    c = O.silu_int(triples(L, xs))
    check("silu", all(bool((i == j).all()) for i, j in zip(a, c)))

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
