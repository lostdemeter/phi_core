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
    print(f"{tag}: {{'OK' if cond else 'FAIL'}} {{extra}}")
    if not cond:
        FAIL.append(tag)


def main():
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {{FAIL}}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
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
        fh.write("numpy\nPillow\n# torch optional (CUDA backend / calibration only)\n")
    with open(os.path.join(path, "demo.py"), "w") as fh:
        fh.write(DEMO_TMPL.format(sname=sname))
    with open(os.path.join(path, "test_core.py"), "w") as fh:
        fh.write(TEST_TMPL)
    subprocess.run(["git", "init", "-q", "-b", "main", path], check=True)
    print(f"stamped {path} ({name}) — next: RECON.md step 0, then S1 fetch")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
