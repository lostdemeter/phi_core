"""Weight probe suite: classify what structures DO from outside.

Black-box functional census (no training, no inference): singular
spectra, frequency response, identity-likeness per conv. Answers "what
is this layer for" well enough to decide treatment (reuse as-is?
replace with synthesis? prune? distill?). See LIBRARY_NOTES pattern:
probe before surgery, always.
"""
import numpy as np


def spectrum(W):
    """Singular values of (Cout, Cin*kh*kw) weight matrix, descending."""
    M = np.asarray(W, dtype=np.float64).reshape(W.shape[0], -1)
    return np.linalg.svd(M, compute_uv=False)


def eff_rank(sv):
    """Effective rank: exp(entropy of normalized spectrum). 1 = rank-1."""
    p = sv / (sv.sum() + 1e-30)
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def kernel_stats(W):
    """Spatial character per (out,in) kernel, aggregated over layer.

    dc: |sum| (lowpass gain); frob: total energy; edge: highpass energy
    fraction via center-surround difference. Returns dict of means.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 4:
        return {}
    kh, kw = W.shape[2], W.shape[3]
    dc = np.abs(W.sum(axis=(2, 3)))
    frob = np.sqrt((W ** 2).sum(axis=(2, 3))) + 1e-30
    out = {"dc_ratio": float((dc / frob).mean())}
    if kh == 3 and kw == 3:
        center = np.abs(W[:, :, 1, 1])
        surround = np.abs(W).sum(axis=(2, 3)) - center
        out["center_frac"] = float((center / (center + surround + 1e-30)).mean())
        # edge-ness: antisymmetric energy (Sobel-like) vs symmetric
        hk = W[:, :, 1, 0] - W[:, :, 1, 2]
        vk = W[:, :, 0, 1] - W[:, :, 2, 1]
        out["edge_frac"] = float(((hk ** 2 + vk ** 2).sum() / ((W ** 2).sum() + 1e-30)))
    if kh == 1 and kw == 1:
        out["is_pointwise"] = True
    return out


def identity_score(W):
    """How close to identity (only meaningful if square Cin==Cout, k3).

    Fraction of Frobenius energy on the (c,c,1,1) center diagonal.
    Residual-path layers score high; mixing layers score ~1/C.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 4 or W.shape[0] != W.shape[1] or W.shape[2] != 3:
        return 0.0
    C = W.shape[0]
    diag = sum(W[c, c, 1, 1] ** 2 for c in range(C))
    tot = (W ** 2).sum() + 1e-30
    return float(diag / tot)


def classify(name, W):
    """Label + metrics dict. Bins (thresholds documented, tunable):
    LOWPASS / EDGE / IDENTITY / MIXING / LOWRANK / POINTWISE."""
    W = np.asarray(W, dtype=np.float64)
    sv = spectrum(W)
    er = eff_rank(sv)
    top1 = float(sv[0] / (sv.sum() + 1e-30))
    st = kernel_stats(W)
    out = {"shape": W.shape, "eff_rank": round(er, 1), "top1_frac": round(top1, 3)}
    out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})
    Co = W.shape[0]
    label = "MIXING"
    if st.get("is_pointwise"):
        label = "POINTWISE"
    elif out.get("dc_ratio", 0) > 0.75:
        label = "LOWPASS"
    elif out.get("edge_frac", 0) > 0.35:
        label = "EDGE"
    if identity_score(W) > 0.25:
        label = "IDENTITY"
    if top1 > 0.6:
        label += "+LOWRANK"
    out["label"] = label
    out["name"] = name
    return out
