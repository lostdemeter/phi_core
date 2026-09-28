"""Shared op helpers (extracted verbatim from the proven port).

Small exact integer ops used by every executor: const/slope triple
builders, lattice compare, clip, and fixed-point binary ops. Gated
0-diff against the originals (tests/test_ops_helpers.py pattern inside
test_lattice run). Model executors import these instead of carrying
copies (third copy was the trigger — see LIBRARY_NOTES).
"""
import numpy as np

from phi_core import lattice as S

_CONST_CACHE = {}


def m_cov(M, blob):
    """Scale-table convention: covering scales live under "cov:<blob>"."""
    return M["cov:" + blob]


def WD(Wt, name, suffix):
    d = Wt[name + "." + suffix]
    return {"s": d["s"], "e": d["e"], "z": d["z"]}


def WB(Wt, name):
    key = name + ".bias"
    if key not in Wt:
        return None
    d = Wt[key]
    return (d["s"], d["e"], d["z"])


def const_like(sh, arr):
    # scalars only (array constants use slope_triples). Cached by value:
    # S.encode uses float log (setup-class; must not run per-frame in loop).
    key = (tuple(sh), float(arr))
    if key not in _CONST_CACHE:
        s, e, z = S.encode(np.atleast_1d(np.asarray(arr, dtype=np.float64)))
        assert s.size == 1, f"non-scalar const {arr}"
        _CONST_CACHE[key] = (
            np.full(sh, s.flat[0], np.int8), np.full(sh, e.flat[0], np.int32),
            np.zeros(sh, np.uint8))
    return _CONST_CACHE[key]


def slope_triples(sl, shp):
    # sl: {"s","e","z"} slope arrays, scalar or (C,); broadcast to (H,W,C)
    C = shp[2]
    if sl["s"].size == 1:
        return (np.full(shp, sl["s"].flat[0], np.int8),
                np.full(shp, sl["e"].flat[0], np.int32),
                np.zeros(shp, np.uint8))
    assert sl["s"].size == C, f"slope count {sl['s'].size} vs C {C}"
    return (np.broadcast_to(sl["s"].reshape(1, 1, C), shp).copy().astype(np.int8),
            np.broadcast_to(sl["e"].reshape(1, 1, C), shp).copy().astype(np.int32),
            np.zeros(shp, np.uint8))


def cmp_gt(a, b):
    az, bz = a[2].astype(bool), b[2].astype(bool)
    out = np.zeros(a[0].shape, bool)
    live = ~az & ~bz
    same = live & (a[0] == b[0])
    out |= same & (a[1] > b[1]) & (a[0] > 0)
    out |= same & (a[1] < b[1]) & (a[0] < 0)
    out |= live & (a[0] > b[0])
    out |= ~az & bz & (a[0] > 0)
    return out


def clip_int(t, lo, hi):
    lo_t = const_like(t[0].shape, lo)
    hi_t = const_like(t[0].shape, hi)
    s = np.where(cmp_gt(lo_t, t), lo_t[0], t[0])
    e = np.where(cmp_gt(lo_t, t), lo_t[1], t[1]).astype(np.int32)
    z = np.where(cmp_gt(lo_t, t), lo_t[2], t[2]).astype(np.uint8)
    s2 = np.where(cmp_gt((s, e, z), hi_t), hi_t[0], s)
    e2 = np.where(cmp_gt((s, e, z), hi_t), hi_t[1], e).astype(np.int32)
    z2 = np.where(cmp_gt((s, e, z), hi_t), hi_t[2], z).astype(np.uint8)
    return s2.astype(np.int8), e2, z2


def bin2(op, a, ma, c, mc, m_out):
    if op == 2:
        s = (a[0].astype(np.int16) * c[0].astype(np.int16)).astype(np.int8)
        e = np.clip(a[1].astype(np.int64) + c[1].astype(np.int64) - S.BIAS,
                    0, S.NLVL - 1).astype(np.int32)
        return s, e, (a[2] | c[2]).astype(np.uint8)
    fa = S.to_fixed(a[0], a[1], a[2], m_out)
    fc = S.to_fixed(c[0], c[1], c[2], m_out)
    if op == 0:
        q = fa + fc
    elif op == 1:
        q = fa - fc
    elif op == 7:
        q = fc - fa
    else:
        raise SystemExit(f"error: binary op {op}")
    return S.from_fixed(q, m_out)
