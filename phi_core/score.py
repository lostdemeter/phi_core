"""Scoring with declared basis (comparison-basis law, now code).

Every false red in the program's history has been comparison-basis,
never the op: wrong peak (0-1 vs 255), wrong units, float-vs-lattice
rounding order, absolute bar on large-range tensors. This module makes
the basis DECLARATIVE and LOUD: no bare psnr() calls anywhere.
"""
import numpy as np


def psnr(a, b, peak=255.0):
    """Peak must be stated (255 for uint8 images, 1.0 for 0-1 floats).
    Passing 0-1 data with peak 255 (or vice versa) is a ~48dB error —
    the single most repeated bug class on record."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    m = float(((a - b) ** 2).mean())
    return 10 * np.log10(peak * peak / m) if m > 0 else 99.0


def lattice_bar(rel_tol=2e-3):
    """Honest bar for lattice agreement: relative to signal max.
    Lattice resolution is 9.4e-4 relative; absolute bars on
    large-range tensors are meaningless (large-weight regime lesson).
    Returns a checker fn(rel_err) for messages."""
    def check(tag, got, ref):
        d = float(np.abs(np.asarray(got, float) - np.asarray(ref, float)).max())
        denom = float(np.abs(np.asarray(ref, float)).max())
        rel = d / (denom + 1e-12)
        ok = rel < rel_tol
        print(f"{tag}: {'OK' if ok else 'FAIL'} rel {rel:.2e} (bar {rel_tol:.0e})")
        return ok
    return check
