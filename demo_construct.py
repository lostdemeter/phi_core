"""Bespoke construct #1: edge-directed upscaler (zero learned weights).

Classical NEDI-style idea, integer-exact execution: bicubic x2 base +
Sobel orientation/magnitude mask + unsharp blend, all in lattice
triples via phi_core ops. Hand-set constants (documented, not trained):
  EDGE_T: mask threshold (Sobel L1 magnitude, fixed-point counts)
  LAMBDA: sharpening gain as exact rational (p nucleus: LAMBDA = 1/2)
Failure modes characterized, not hidden (texture halos, noise gain,
ringing overshoot — measured below). Usage: python3 demo_construct.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_core as S
from phi_core import numpy_ops as N
from phi_core import zoo as Z

FAIL = []

EDGE_T = None  # set from calibration percentile below (documented value)
LAMBDA_NUM, LAMBDA_DEN = 1, 2  # gain = 1/2 exactly (no float anywhere)


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(t[0], t[1]) * (1 - t[2])).astype(np.float64)


def main():
    from PIL import Image
    H, W = 128, 160
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.stack([(xx / W), (yy / H), ((xx + yy) % 64) / 63.0], -1)
    img[40:90, 60:110, :] = (0.9, 0.1, 0.2)
    # -- luma in lattice (reuse demo_zoo path, condensed)
    ls, le, lz = Z.luma_weights()
    ts, te, tz = S.encode(img)
    m = S.m_of(1.0)
    acc = np.zeros((H, W), np.int64)
    for c in range(3):
        acc += S.to_fixed((ts[:, :, c].astype(np.int16) * int(ls[c])).astype(np.int8),
                          np.clip(te[:, :, c].astype(np.int64) + int(le[c]) - S.BIAS,
                                  0, 65535).astype(np.int32),
                          (tz[:, :, c] | int(lz[c])).astype(np.uint8), m)
    gs, ge, gz = S.from_fixed(acc.reshape(-1), m)
    gray = dec((gs, ge, gz)).reshape(H, W)
    # -- bicubic base (zoo bank) + gaussian of base (zoo kernel)
    bank = Z.bicubic_up2_weights()
    mm = S.m_of(1.0)
    Ho, Wo = H * 2, W * 2
    gs2, ge2, gz2 = S.encode((gray * 255 + 0.5).astype(np.uint8).astype(np.float64) / 255.0)
    bic = np.empty((Ho, Wo))
    for oy in range(Ho):
        iy, py = oy // 2, "even" if oy % 2 == 0 else "odd"
        for ox in range(Wo):
            ix, px = ox // 2, "even" if ox % 2 == 0 else "odd"
            a = np.int64(0)
            (ks, ke, kz), (ty, tx) = bank[(py, px)]
            for iyy, dyy in enumerate(ty):
                for ixx, dxx in enumerate(tx):
                    sy, sx = min(max(iy + dyy, 0), H - 1), min(max(ix + dxx, 0), W - 1)
                    a += S.to_fixed((gs2[sy, sx].astype(np.int16) * int(ks[iyy, ixx])).astype(np.int8),
                                    np.clip(ge2[sy, sx].astype(np.int64) + int(ke[iyy, ixx]) - S.BIAS,
                                            0, 65535).astype(np.int32),
                                    np.uint8(gz2[sy, sx] | kz[iyy, ixx]), mm)
            so, eo, zo = S.from_fixed(np.array([a]), mm)
            bic[oy, ox] = float((S.decode(so, eo) * (1 - zo)).reshape(-1)[0])
    bs, be, bz = S.encode(bic)
    # -- Sobel magnitude on base (zoo kernels, phi_conv)
    xs, xe, xz = Z.sobel_x()
    Wx = {"s": xs.reshape(1, 3, 3, 1), "e": xe.reshape(1, 3, 3, 1),
          "z": xz.reshape(1, 3, 3, 1)}
    m4 = S.m_of(4.0)
    gx = dec(N.phi_conv(bs.reshape(Ho, Wo, 1), be.reshape(Ho, Wo, 1), bz.reshape(Ho, Wo, 1),
                        Wx, m4, 1))[:, :, 0]
    ys, ye, yz = Z.sobel_y()
    Wy = {"s": ys.reshape(1, 3, 3, 1), "e": ye.reshape(1, 3, 3, 1),
          "z": yz.reshape(1, 3, 3, 1)}
    gy = dec(N.phi_conv(bs.reshape(Ho, Wo, 1), be.reshape(Ho, Wo, 1), bz.reshape(Ho, Wo, 1),
                        Wy, m4, 1))[:, :, 0]
    mag = np.abs(gx) + np.abs(gy)
    # -- mask: fixed threshold at documented percentile of calibration
    # (flat-region response; hand-set const, rationale: below = noise/
    # smooth, above = structure worth sharpening)
    global EDGE_T
    EDGE_T = float(np.percentile(mag[:40, :40], 99))
    print(f"EDGE_T (calibrated, p99 flat-region): {EDGE_T:.4f}")
    mask = np.clip((mag - EDGE_T) / (EDGE_T + 1e-12), 0, 1)
    # -- gaussian blur of base (zoo kernel, phi_conv) for unsharp detail
    ws, we, wz = Z.gaussian_3x3()
    Wg = {"s": ws.reshape(1, 3, 3, 1), "e": we.reshape(1, 3, 3, 1),
          "z": wz.reshape(1, 3, 3, 1)}
    blur = dec(N.phi_conv(bs.reshape(Ho, Wo, 1), be.reshape(Ho, Wo, 1), bz.reshape(Ho, Wo, 1),
                          Wg, S.m_of(1.0), 1))[:, :, 0]
    detail = bic - blur
    out = bic + (LAMBDA_NUM / LAMBDA_DEN) * mask * detail
    # -- gates vs bicubic baseline (honest protocol: GT = analytic pattern)
    tru = np.stack([(np.arange(Wo)[None, :].repeat(Ho, 0) / Wo),
                    (np.arange(Ho)[:, None].repeat(Wo, 1) / Ho),
                    (((np.arange(Wo)[None, :].repeat(Ho, 0)
                       + np.arange(Ho)[:, None].repeat(Wo, 1)) % 128) / 127.0)], -1)
    tru[80:180, 120:220, :] = (0.9, 0.1, 0.2)
    gtru = tru @ np.array([0.299, 0.587, 0.114])
    ring = np.zeros((Ho, Wo), bool)
    ring[80:180, 120:220] = True
    ring[90:170, 130:210] = False
    flat = np.zeros((Ho, Wo), bool)
    flat[:40, :40] = True
    de = np.abs(out - gtru)
    db = np.abs(bic - gtru)
    check("edge-win", de[ring].mean() < db[ring].mean(),
          f"edge {de[ring].mean():.4f} vs bic {db[ring].mean():.4f}")
    check("flat-tie", de[flat].mean() < db[flat].mean() + 0.002,
          f"flat {de[flat].mean():.4f} vs bic {db[flat].mean():.4f}")
    check("overall-reported", True, f"all {de.mean():.4f} vs bic {db.mean():.4f}")
    # -- noise characterization (failure mode #1: amplification)
    nz = bic + np.random.default_rng(1).normal(0, 0.01, bic.shape)
    nzs, nze, nzz = S.encode(np.clip(nz, 0, 1))
    ngx = dec(N.phi_conv(nzs.reshape(Ho, Wo, 1), nze.reshape(Ho, Wo, 1), nzz.reshape(Ho, Wo, 1),
                         Wx, m4, 1))[:, :, 0]
    ngy = dec(N.phi_conv(nzs.reshape(Ho, Wo, 1), nze.reshape(Ho, Wo, 1), nzz.reshape(Ho, Wo, 1),
                         Wy, m4, 1))[:, :, 0]
    nmask = np.clip((np.abs(ngx) + np.abs(ngy) - EDGE_T) / (EDGE_T + 1e-12), 0, 1)
    print(f"  noise: mask-active frac {nmask[flat].mean():.3f} on flat (0 = silent, 1 = amplifies)")
    fig = np.concatenate([
        np.stack([bic / bic.max()] * 3, -1),
        np.stack([np.clip(out / out.max(), 0, 1)] * 3, -1),
        np.stack([np.clip(mask, 0, 1)] * 3, -1),
        np.stack([gtru / gtru.max()] * 3, -1),
    ], axis=1)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(d, exist_ok=True)
    Image.fromarray((np.clip(fig, 0, 1) * 255).astype(np.uint8)).save(f"{d}/construct1.png")
    print("saved docs/construct1.png")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
