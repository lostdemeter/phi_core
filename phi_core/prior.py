"""Prior modulation (staged for promotion review -- branch, NOT main).

Multiplicative caution from offline priors: an offline prior, converted at
the boundary, becomes exact multiplier triples composed onto boost via tmul.
The unaffected subset takes EXACT identity (ones: tmul identity is exact),
so priors cost nothing where they don't apply. Two forms, one family:
select() for binary priors, affine() for continuous ones with exact-ones
override. Composition of already-lowered ops (tmul, bridge, np.where
selects); no new C needed.

Provenance: authored in holographic_enhancement (chain/prior.py, #LIB-020):
depth near/far select + motion flow-scale affine, both call sites refactored
0-diff with suites green. This staging ports it onto phi-core substrates
(lattice + numpy_ops only). Branch ai/prior-mult, merge pending review.
"""
import numpy as np

from phi_core import lattice as S
from phi_core.numpy_ops import tmul


def _encode(v, shape):
    return S.encode(np.full(shape, float(v), dtype=np.float64))


def _binop(a, b, m, op="add"):
    qa = S.to_fixed(a[0], a[1], a[2], m)
    qb = S.to_fixed(b[0], b[1], b[2], m)
    return S.from_fixed(qa + qb if op == "add" else qa - qb, m)


def select(shape, mask, atten, full=1.0, m_cov=None):
    """Bool mask True -> full, False -> atten. Exact select, no arithmetic."""
    full_t = _encode(full, shape)
    att_t = _encode(atten, shape)
    m = np.where(np.ascontiguousarray(mask, dtype=bool), 0, 1).astype(np.int8)
    return (np.where(m == 0, full_t[0], att_t[0]).astype(np.int8),
            np.where(m == 0, full_t[1], att_t[1]).astype(np.int32),
            np.where(m == 0, full_t[2], att_t[2]).astype(np.uint8))


def affine(field_trip, static_mask, atten, m_cov):
    """1-(1-atten)*field with exact ones where static."""
    shape = field_trip[0].shape
    ka = S.encode(np.full(shape, 1.0 - float(atten), dtype=np.float64))
    one = S.encode(np.ones(shape, dtype=np.float64))
    dec = _binop(one, tmul(ka, field_trip), m_cov, op="sub")
    st = np.ascontiguousarray(static_mask, dtype=bool)
    assert st.shape == shape, f"static geometry {st.shape} vs {shape}"
    return (np.where(st, one[0], dec[0]).astype(np.int8),
            np.where(st, one[1], dec[1]).astype(np.int32),
            np.where(st, one[2], dec[2]).astype(np.uint8))
