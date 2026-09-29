"""Starter kit: stamp a new geometric-model repo (LIBRARY_NOTES#1).

 hand-copying core files + renames + path surgery cost a full turn on
 project 3. This stamps it in seconds: skeleton dirs, README template,
 .gitignore (artifacts never uploaded), demo/test templates with the
 gate patterns baked in, local git init.
Usage: python3 -m phi_core.stamp <name> <path>  (e.g. upscaler v3)
"""
import os
import subprocess
import sys

README_TMPL = """# {name}: fully-geometric <MODEL>

![hero](docs/hero.png)

*One-line what + the headline parity number.*

## Results

Check | Result
---|---
Float assembly vs oracle | dB (gate: wiring proven, not assumed)
Int chain vs float | dB (bar >=40)
... | ...

## Quick start

```
git clone <url> && cd {sname}
pip install -r requirements.txt
python3 fetch_weights.py     # one-time, verified download
python3 test_core.py         # fast gates, expect ALL OK
python3 demo.py ...          # fidelity gate (>=40dB), expect GO
```

## How it works

(phi pipeline: frontend -> integer datapath -> backends. See phi-core IR.md.)

## License

GPLv3 — see [LICENSE](LICENSE).
"""

GITIGNORE = """# baked/cached artifacts (built locally, never uploaded)
*.npy
*.pth
*.pt
*.bin
weights/
luts/
captures/
mid_*/
seq*/
*.log
__pycache__/
*.pyc
.tmp/
"""

DEMO_TMPL = '''"""{sname} demo: <pair> -> output + fidelity gate (>=40dB).
Backend: numpy integer emu (no torch needed). Usage: see README.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BAR_DB = 40.0


def main():
    raise SystemExit("wire the chain here (see phi-core IR.md + a sibling *_reverse repo)")


if __name__ == "__main__":
    main()
'''

TEST_TMPL = '''"""Fast gates (seconds, no model files): comparison-basis declared
per test (units, peak, reference rounding). See LIBRARY_NOTES#compare-basis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    import phi_core as S
    import numpy as np
    # substrate smoke through the installed dependency (not vendored)
    x = (np.random.default_rng(0).random(500) - 0.5) * 4
    s, e, z = S.encode(x)
    v = S.decode(s, e) * (1 - z)
    check("codec-smoke", float(np.abs(v - x).max()) < 1e-3)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
'''

IMPORTS_TMPL = '''"""Import sweep: every module must import (catches migration damage).

Best-effort for torch/CUDA-absent systems on backend modules; hard fail
otherwise. Extend CORE as modules arrive. Usage: python3 test_imports.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CORE = []
BACKEND = []
FAIL = []


def main():
    import phi_core as S  # noqa: dependency itself resolves
    print("phi_core: OK", S.K)
    for m in CORE:
        try:
            __import__(m)
            print(f"{m}: OK")
        except Exception as ex:
            print(f"{m}: FAIL {str(ex)[:120]}")
            FAIL.append(m)
    try:
        import torch  # noqa
        have_torch = True
    except ImportError:
        have_torch = False
    for m in BACKEND:
        try:
            __import__(m)
            print(f"{m}: OK")
        except Exception as ex:
            if have_torch:
                print(f"{m}: FAIL {str(ex)[:120]}")
                FAIL.append(m)
            else:
                print(f"{m}: SKIP (no torch)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
'''

FINGERPRINT_TMPL = '''"""Weight fingerprint (executable artifact spec — fill from arch source).

Every field asserted at fetch time; format drift must break HERE, never
as a downstream math mystery (see the wrong-model lesson). Usage:
python3 fetch_weights.py (implement per model, keep these asserts).
"""
import sys

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


# FILL: size range, key roster anchors (first/last), exact shapes per
# block family, bias siblings, forbidden substrings (norms?), param total.
# Example shape (copy, don't invent — read from the arch source):
#   check("conv_first shape", tuple(sd["conv_first.weight"].shape) == (64, 3, 3, 3))
'''



def main(name, path):
    sname = name.lower().replace("-", "_")
    os.makedirs(path, exist_ok=False)
    for d in ("weights", "samples", "docs", "captures", "luts", "chain"):
        os.makedirs(os.path.join(path, d))
    with open(os.path.join(path, "README.md"), "w") as fh:
        fh.write(README_TMPL.format(name=name, sname=sname))
    with open(os.path.join(path, ".gitignore"), "w") as fh:
        fh.write(GITIGNORE)
    with open(os.path.join(path, "requirements.txt"), "w") as fh:
        fh.write("numpy\nPillow\n"
                 "phi-core @ git+https://github.com/lostdemeter/phi_core\n"
                 "# torch optional (CUDA backend / calibration only)\n")
    with open(os.path.join(path, "demo.py"), "w") as fh:
        fh.write(DEMO_TMPL.format(sname=sname))
    with open(os.path.join(path, "test_core.py"), "w") as fh:
        fh.write(TEST_TMPL)
    with open(os.path.join(path, "test_imports.py"), "w") as fh:
        fh.write(IMPORTS_TMPL)
    with open(os.path.join(path, "fetch_weights.py"), "w") as fh:
        fh.write(FINGERPRINT_TMPL)
    with open(os.path.join(path, "LIBRARY_NOTES.md"), "w") as fh:
        fh.write("# Library notes (running log — snags + commonalities)\n\n"
                 " goal: expose what repeats across models so phi-core grows\n"
                 " by evidence. Rule: dated entries, concrete pain/proposal.\n")
    subprocess.run(["git", "init", "-q", "-b", "main", path], check=True)
    print(f"stamped {path} ({name}) — next: RECON.md step 0, then S1 fetch")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
