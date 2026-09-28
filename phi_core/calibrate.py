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
