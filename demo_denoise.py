"""Bespoke construct #3: edge-preserving denoiser (new capability).

Gaussian smooths everything (edges bleed); this smooths ONLY where the
Sobel mask says flat, keeping original pixels on structure — all in
lattice triples via shared zoo helpers. Gate: beat the noisy input AND
match-or-beat plain gaussian (edge preservation must show up in the
numbers, not just the figure). Honest protocol: GT known, noise added.
Usage: python3 demo_denoise.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_core as S
from phi_core import numpy_ops as N
from phi_core import zoo as Z
from phi_core import score as SC

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(t[0], t[1]) * (1 - t[2])).astype(np.float64)


def blur2(t, m):
    """Two gaussian passes (5x5 effective) via shared zoo kernel."""
    ws, we, wz = Z.gaussian_3x3()
    W = {"s": ws.reshape(1, 3, 3, 1), "e": we.reshape(1, 3, 3, 1),
         "z": wz.reshape(1, 3, 3, 1)}
    H, Wd, C = t[0].shape
    outs = []
    for c in range(C):
        o = N.phi_conv(t[0][:, :, c:c + 1], t[1][:, :, c:c + 1], t[2][:, :, c:c + 1],
                       W, m, 1)
        outs.append(o)
    b = (np.concatenate([o[0] for o in outs], 2), np.concatenate([o[1] for o in outs], 2),
         np.concatenate([o[2] for o in outs], 2))
    outs2 = []
    for c in range(C):
        o = N.phi_conv(b[0][:, :, c:c + 1], b[1][:, :, c:c + 1], b[2][:, :, c:c + 1],
                       W, m, 1)
        outs2.append(o)
    return (np.concatenate([o[0] for o in outs2], 2),
            np.concatenate([o[1] for o in outs2], 2),
            np.concatenate([o[2] for o in outs2], 2))


def main():
    from PIL import Image
    FR = "/home/thorin/Documents/OpenCode/new_target/natural/frames"
    gt = np.asarray(Image.open(f"{FR}/f_012.png").convert("RGB"))
    rng = np.random.default_rng(3)
    nz = (rng.normal(0, 10, gt.shape)).astype(np.float64)
    noisy = np.clip(gt.astype(np.float64) + nz, 0, 255).astype(np.uint8)
    H, Wd, _ = gt.shape
    m = S.m_of(1.0)
    t = S.encode(noisy.astype(np.float64) / 255.0)
    # Sobel magnitude on luma, then binary flat mask (mag < T)
    lu = (0.299 * noisy[:, :, 0] + 0.587 * noisy[:, :, 1] + 0.114 * noisy[:, :, 2])
    xs, xe, xz = Z.sobel_x()
    Wx = {"s": xs.reshape(1, 3, 3, 1), "e": xe.reshape(1, 3, 3, 1),
          "z": xz.reshape(1, 3, 3, 1)}
    ys, ye, yz = Z.sobel_y()
    Wy = {"s": ys.reshape(1, 3, 3, 1), "e": ye.reshape(1, 3, 3, 1),
          "z": yz.reshape(1, 3, 3, 1)}
    lus, lue, luz = S.encode(lu / 255.0)
    gx = dec(N.phi_conv(lus.reshape(H, Wd, 1), lue.reshape(H, Wd, 1), luz.reshape(H, Wd, 1),
                        Wx, S.m_of(4.0), 1))[:, :, 0]
    gy = dec(N.phi_conv(lus.reshape(H, Wd, 1), lue.reshape(H, Wd, 1), luz.reshape(H, Wd, 1),
                        Wy, S.m_of(4.0), 1))[:, :, 0]
    mag = np.abs(gx) + np.abs(gy)
    T = float(np.percentile(mag, 70))
    print(f"flat threshold T={T:.4f} (p70 magnitude)")
    flat = np.where(mag < T, 1.0, 0.0)
    # build HWC mask triples with e==BIAS exactly where flat
    ms, me, mz = S.encode(np.stack([flat] * 3, -1))
    b = blur2(t, m)
    out = Z.masked_blend(b, t, (ms, me, mz), m)
    ov = (np.clip(dec(out), 0, 1) * 255 + 0.5).astype(np.uint8)
    bv = (np.clip(dec(b), 0, 1) * 255 + 0.5).astype(np.uint8)
    dn, dg, do = SC.psnr(noisy, gt), SC.psnr(bv, gt), SC.psnr(ov, gt)
    print(f"noisy {dn:.2f} gauss {dg:.2f} ours {do:.2f}")
    check("beats-noisy", do > dn + 1.0, f"+{do - dn:.2f}dB")
    check("beats-gauss", do >= dg - 0.2, f"{do - dg:+.2f}dB (edge preservation)")
    fig = np.concatenate([noisy, bv, ov, gt], axis=1)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(d, exist_ok=True)
    Image.fromarray(fig).save(f"{d}/construct_denoise.png")
    print("saved docs/construct_denoise.png")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
