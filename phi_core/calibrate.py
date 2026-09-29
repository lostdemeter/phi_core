"""Calibration: offline range analysis -> frozen integer scales.

Doctrine: measure maxima on real inputs (torch hooks or any reference),
cover with margin, freeze into the artifact. Calibration NEVER runs in
the shipped path. torch is imported lazily (optional dependency).
"""
import math
import numpy as np

import phi_core.lattice as S


def m_of(pmax, margin=512):
    """Lattice scale covering pmax with margin (~2.7x at 512 steps)."""
    return S.BIAS + int(round(S.K * math.log(pmax + 1e-12) / S.LN_PHI)) + margin


def calibrate_from_hooks(net, imgs01):
    """Record per-Conv2d input/output abs-maxima over imgs01 (CHW float32
    numpy arrays). Returns {name: (amax_in, amax_out)}. The model maps
    these to m_acc (products) and m_cov (add/warp/special inputs) with
    m_of — see esr_cal.py in esrgan_reverse for the worked pattern
    (residual-bound derivation included)."""
    import torch
    maxima = {}
    hooks = []

    def hk(name):
        def f(mod, inp, out):
            a = float(inp[0].detach().abs().max())
            b = float(out.detach().abs().max())
            d = maxima.setdefault(name, [0.0, 0.0])
            d[0] = max(d[0], a)
            d[1] = max(d[1], b)
        return f

    for name, mod in net.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            hooks.append(mod.register_forward_hook(hk(name)))
    net.eval()
    with torch.no_grad():
        for im in imgs01:
            x = torch.from_numpy(np.ascontiguousarray(im[None])).float()
            net(x)
    for h in hooks:
        h.remove()
    return maxima


def fold_batchnorm(conv_w, conv_b, mean, var, bn_w, bn_b, eps=1e-5):
    """Fold inference BatchNorm into conv weights (exact, offline).

    BN(x) = bw*(x-mean)/sqrt(var+eps) + bb applied after conv(x) =
    conv_w*x + conv_b. Folded: W' = W*bw/sqrt(var+eps) (per-out-ch),
    b' = (b-mean)*bw/sqrt(var+eps) + bb. First BN sighting across four
    models (DDColor UNet); every future BN model inherits this.
    Gate: folded vs torch F.batch_norm <= 1e-6 (test below pattern).
    All inputs/outputs float64 numpy, shapes OIHW / (C,).
    """
    import numpy as np
    conv_w = np.asarray(conv_w, dtype=np.float64)
    conv_b = np.asarray(conv_b, dtype=np.float64)
    scale = np.asarray(bn_w, dtype=np.float64) / np.sqrt(
        np.asarray(var, dtype=np.float64) + eps)
    W = conv_w * scale.reshape(-1, *([1] * (conv_w.ndim - 1)))
    b = (conv_b - np.asarray(mean, dtype=np.float64)) * scale + np.asarray(
        bn_b, dtype=np.float64)
    return W, b
