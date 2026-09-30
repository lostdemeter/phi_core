"""Verdict gates: boundary float comparisons with stated inclusivity.

The helper is new code (authored, not extracted), so there is no extraction
original to 0-diff against. Instead: direct contract gates (ops, boundary
inclusivity, refusal) + a cross-check against naive numpy comparisons on
edge-heavy vectors (proves the helper adds NOTHING over plain semantics --
the value is the contract and the single conversion point, not math).
Provenance: third use in holographic_enhancement (motion static mask, depth
median split, router mode verdict), staged branch ai/verdict-helper.
Usage: python3 tests/test_verdict.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_core.verdict import verdict_mask

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-10, 10, 2000),
                        [0.0, -0.0, 0.25, 1.0, 1e-300, 1e300]])
    check("ge", bool((verdict_mask(x, 0.25, ">=") == (x >= 0.25)).all()))
    check("le", bool((verdict_mask(x, 0.25, "<=") == (x <= 0.25)).all()))
    check("eq", bool((verdict_mask(x, 0.0, "==") == (x == 0.0)).all()))
    # boundary inclusivity stated: >= engages AT equality (router depends).
    check("inclusive", bool(verdict_mask(np.array([0.25]), 0.25, ">=")[0]),
          ">= engages at equality")
    check("eq-zero", bool(verdict_mask(np.array([0.0]), 0.0, "==")[0]),
          "exact-zero static detection")
    try:
        verdict_mask(x, 0.0, "!=")
        check("refuses", False, "no error raised")
    except ValueError:
        check("refuses", True, "unknown op fails loud")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
