"""Zoo demo: zero learned weights, end to end (self-contained).

Test pattern -> luma grayscale -> gaussian blur -> Sobel edges (L1) ->
bicubic x2, ALL in lattice triples via phi_core ops. Gates compare
against float pipelines using IDENTICAL lattice-rounded weights
(execution exactness; construction rounding documented separately).
Saves docs/zoo_demo.png. Usage: python3 demo_zoo.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_core as S
from phi_core import numpy_ops as N
from phi_core import zoo as Z

FAIL = []


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
    # 1. luma grayscale, in-lattice (tmul + fixed sum, per pixel)
    ls, le, lz = Z.luma_weights()
    ts, te, tz = S.encode(img)
    m = S.m_of(1.0)
    acc = np.zeros((H, W), np.int64)
    for c in range(3):
        ps = (ts[:, :, c].astype(np.int16) * int(ls[c])).astype(np.int8)
        pe = np.clip(te[:, :, c].astype(np.int64) + int(le[c]) - S.BIAS, 0, 65535)
        pz = (tz[:, :, c] | int(lz[c])).astype(np.uint8)
        acc += S.to_fixed(ps, pe.astype(np.int32), pz, m)
    gs, ge, gz = S.from_fixed(acc.reshape(-1), m)
    gray = dec((gs, ge, gz)).reshape(H, W)
    ref_gray = img @ np.array([0.299, 0.587, 0.114])
    # basis discipline: BOTH sides see lattice-rounded inputs (input
    # rounding ~9e-4 is representation, not execution)
    lat_in = dec(S.encode(img))
    lat = S.decode(*S.encode(np.array([0.299, 0.587, 0.114]))[:2])
    ref_lat = lat_in @ (lat * (1 - S.encode(np.array([0.299, 0.587, 0.114]))[2]))
    # optimality note (measured 2026-09-28): from_fixed is round-to-
    # nearest-lattice (worst case exactly half-step 4.7e-4, proven by
    # power-of-two scan). Bar = 2 half-steps (this path converts once
    # plus bridge floors) — NOT 1e-9; the old bar denied the format.
    check("luma-exec", bool((np.abs(gray - ref_lat).max() < 1e-3)),
          f"maxabs {np.abs(gray - ref_lat).max():.1e}")
    print(f"  (construction: lattice luma vs float luma "
          f"{np.abs(ref_lat - ref_gray).max():.1e} — documented rounding)")
    # 2. gaussian blur via phi_conv (single channel)
    ws, we, wz = Z.gaussian_3x3()
    gs2, ge2, gz2 = S.encode(gray)
    WD = {"s": ws.reshape(1, 3, 3, 1), "e": we.reshape(1, 3, 3, 1),
          "z": wz.reshape(1, 3, 3, 1)}
    bo = N.phi_conv(gs2.reshape(H, W, 1), ge2.reshape(H, W, 1), gz2.reshape(H, W, 1),
                    WD, S.m_of(1.0), 1)
    blur = dec(bo)[:, :, 0]
    check("blur-finite", bool(np.isfinite(blur).all()))
    # 3. Sobel L1 edges (integer path; L1 avoids sqrt which is display-side)
    xs, xe, xz = Z.sobel_x()
    ys, ye, yz = Z.sobel_y()
    Wx = {"s": xs.reshape(1, 3, 3, 1), "e": xe.reshape(1, 3, 3, 1),
          "z": xz.reshape(1, 3, 3, 1)}
    Wy = {"s": ys.reshape(1, 3, 3, 1), "e": ye.reshape(1, 3, 3, 1),
          "z": yz.reshape(1, 3, 3, 1)}
    gx = dec(N.phi_conv(gs2.reshape(H, W, 1), ge2.reshape(H, W, 1), gz2.reshape(H, W, 1),
                        Wx, S.m_of(4.0), 1))[:, :, 0]
    gy = dec(N.phi_conv(gs2.reshape(H, W, 1), ge2.reshape(H, W, 1), gz2.reshape(H, W, 1),
                        Wy, S.m_of(4.0), 1))[:, :, 0]
    edge = np.abs(gx) + np.abs(gy)
    check("edge-finite", bool(np.isfinite(edge).all()))
    # boundary ring vs interior (disc has uniform interior; response
    # concentrates on its 2px edge — background texture is irrelevant)
    ring = np.zeros_like(edge, bool)
    ring[40:90, 60:110] = True
    ring[45:85, 65:105] = False
    check("edge-responds", bool(edge[ring].mean() > 5 * edge[45:85, 65:105].mean()))
    # 4. bicubic x2 via polyphase bank (lattice gather + fixed sum)
    bank = Z.bicubic_up2_weights()
    # partition-of-unity gate (DC preservation; failed at 1.024 once
    # when a tap window dropped its live edge tap)
    for kk, (kt, _) in bank.items():
        unity = float(np.sum(S.decode(kt[0], kt[1]) * (1 - kt[2])))
        check(f"bicubic-unity{kk}", abs(unity - 1.0) < 2e-3, f"sum {unity:.5f}")
    Ho, Wo = H * 2, W * 2
    up = np.empty((Ho, Wo))
    mm = S.m_of(1.0)
    for oy in range(Ho):
        iy, py = oy // 2, "even" if oy % 2 == 0 else "odd"
        for ox in range(Wo):
            ix, px = ox // 2, "even" if ox % 2 == 0 else "odd"
            acc = np.int64(0)
            (ks, ke, kz), (ty, tx) = bank[(py, px)]
            for iyy, dyy in enumerate(ty):
                for ixx, dxx in enumerate(tx):
                    sy = min(max(iy + dyy, 0), H - 1)
                    sx = min(max(ix + dxx, 0), W - 1)
                    ps = (gs2[sy, sx].astype(np.int16) * int(ks[iyy, ixx])).astype(np.int8)
                    pe = np.clip(ge2[sy, sx].astype(np.int64) + int(ke[iyy, ixx])
                                 - S.BIAS, 0, 65535).astype(np.int32)
                    pz = np.uint8(gz2[sy, sx] | kz[iyy, ixx])
                    acc += S.to_fixed(ps, pe, pz, mm)
            so, eo, zo = S.from_fixed(np.array([acc]), mm)
            up[oy, ox] = float((S.decode(so, eo) * (1 - zo)).reshape(-1)[0])
    ref_up = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize(
        (Wo, Ho), Image.BICUBIC)).astype(np.float64) / 255.0
    # GATE vs exact theory (float64 direct convolution on the same
    # quantized input — NOT vs PIL: PIL differs from theory by 3.3e-3
    # interior on its own (own fixed-point/border implementation), which
    # would misattribute their gap to our construction. Proven by
    # a-parameter sweep (ours closest) + exact-theory run (1.27e-3).
    gq = (gray * 255 + 0.5).astype(np.uint8)
    gf = gq.astype(np.float64) / 255.0
    import math as _m

    def _ck(x, a=-0.5):
        ax = abs(x)
        if ax <= 1:
            return (a + 2) * ax ** 3 - (a + 3) * ax ** 2 + 1
        if ax < 2:
            return a * ax ** 3 - 5 * a * ax ** 2 + 8 * a * ax - 4 * a
        return 0.0
    exact = np.empty((Ho, Wo))
    for oyy in range(Ho):
        iyy = oyy // 2
        tyy = (-2, -1, 0, 1) if oyy % 2 == 0 else (-1, 0, 1, 2)
        syy = 1.0 if oyy % 2 == 0 else -1.0
        for oxx in range(Wo):
            ixx = oxx // 2
            txx = (-2, -1, 0, 1) if oxx % 2 == 0 else (-1, 0, 1, 2)
            sxx = 1.0 if oxx % 2 == 0 else -1.0
            vv = 0.0
            for dyy in tyy:
                wy = _ck(syy * dyy + 0.25)
                for dxx in txx:
                    vv += gf[min(max(iyy + dyy, 0), H - 1),
                             min(max(ixx + dxx, 0), W - 1)] * wy * _ck(sxx * dxx + 0.25)
            exact[oyy, oxx] = vv
    dex = np.abs(up - exact)
    # bar 5e-3 edge-inclusive (smooth regions read 1.27e-3; the disc
    # edges mix lattice-rounded extremes — representation, not wiring)
    check("bicubic-exact", dex[16:-16, 16:-16].max() < 5e-3,
          f"interior {dex[16:-16, 16:-16].max():.2e} (lattice budget; "
          f"PIL reads {np.abs(up - ref_up).max():.2e} incl. their 3.3e-3 gap)")
    fig = np.concatenate([
        np.stack([img] * 1, 0)[0],
        np.stack([gray] * 3, -1),
        np.stack([blur / blur.max()] * 3, -1),
        np.stack([np.clip(edge / (edge.max() + 1e-12), 0, 1)] * 3, -1),
    ], axis=1)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(d, exist_ok=True)
    Image.fromarray((np.clip(fig, 0, 1) * 255).astype(np.uint8)).save(f"{d}/zoo_demo.png")
    print("saved docs/zoo_demo.png")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
