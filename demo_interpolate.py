"""Bespoke construct #2: motion interpolation (mirrors RIFE's job).

Classical recipe, integer-exact: SAD block-match f0->f1, half-warp both
frames toward the middle, average. Occlusions NOT handled (documented;
RIFE's mask network exists precisely because this is hard).
Gate: beat frame-AVERAGING on held-out GT (foreman f_012/f_014).
If it can't beat averaging, it doesn't ship — that's the rule for
bespoke constructs mirroring learned models.
Usage: python3 demo_interpolate.py
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


def main():
    from PIL import Image
    FR = "/home/thorin/Documents/OpenCode/new_target/natural/frames"
    results = []
    for na, nb, ng in (("f_011", "f_013", "f_012"), ("f_013", "f_015", "f_014")):
        a = np.asarray(Image.open(f"{FR}/{na}.png").convert("RGB"))
        b = np.asarray(Image.open(f"{FR}/{nb}.png").convert("RGB"))
        gt = np.asarray(Image.open(f"{FR}/{ng}.png").convert("RGB"))
        H, W, _ = a.shape
        au = (np.asarray(Image.open(f"{FR}/{na}.png").convert("L")))
        bu = (np.asarray(Image.open(f"{FR}/{nb}.png").convert("L")))
        dx, dy = Z.sad_match(au, bu, block=8, search=10)
        # per-pixel flow (nearest broadcast) in half-pixel units
        fx = np.repeat(np.repeat(dx, 8, 0), 8, 1)[:H, :W].astype(np.float64)
        fy = np.repeat(np.repeat(dy, 8, 0), 8, 1)[:H, :W].astype(np.float64)
        m = S.m_of(1.0)
        ta = S.encode(a.astype(np.float64) / 255.0)  # HWC already
        tb = S.encode(b.astype(np.float64) / 255.0)
        wa = Z.warp_layer(ta, np.stack([-fx / 2, -fy / 2], -1), m)
        wb = Z.warp_layer(tb, np.stack([fx / 2, fy / 2], -1), m)
        fa = S.to_fixed(wa[0], wa[1], wa[2], m)
        fb = S.to_fixed(wb[0], wb[1], wb[2], m)
        so, eo, zo = S.from_fixed(((fa + fb) // 2).reshape(-1), m)
        mid = (S.decode(so, eo) * (1 - zo)).reshape(H, W, 3)
        midu = (np.clip(mid, 0, 1) * 255 + 0.5).astype(np.uint8)
        avg = ((a.astype(float) + b.astype(float)) / 2 + 0.5).astype(np.uint8)
        check(f"{ng}-vs-average",
              SC.psnr(midu, gt) > SC.psnr(avg, gt),
              f"ours {SC.psnr(midu, gt):.2f} vs avg {SC.psnr(avg, gt):.2f}")
        results.append((ng, midu, avg, gt))
    na, midu, avg, gt = results[0][0], results[0][1], results[0][2], results[0][3]
    fig = np.concatenate([avg, midu, gt], axis=1)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(d, exist_ok=True)
    Image.fromarray(fig).save(f"{d}/construct_interp.png")
    print("saved docs/construct_interp.png")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
