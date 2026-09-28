# phi-core: the shared geometric substrate

The integer core extracted from three fully-geometric reverse-engineering
projects (depth, interpolation, upscaling): one lattice codec, one opcode
spec, one lowering per substrate — so model #4 starts with wiring, not math.

A φ-value is `sign × φ^(exponent / 512)`, φ = (1+√5)/2. On this lattice,
multiplication is integer exponent addition; runtime needs no FPU.

## Contents

- `phi_core/` — `lattice.py` (codec/bridge/LUTs), `calibrate.py`
  (offline ranges → frozen scales), `stamp.py` (new-model starter kit)
- `IR.md` — opcode spec: 2 tensor types, closed op set, schedule rules
- `ARCHITECTURE.md` — data flow, ffmpeg mapping, soundness argument
- `c_core/` — C scalar/OpenMP + SIMD (AVX512) + CUDA lowerings
- `tests/` — 0-diff extraction gates vs the originals

## Quick start

```
git clone <url> && cd phi-core
pip install numpy                # only hard dependency
NEW_TARGET=/path/to/reference python3 tests/test_lattice.py  # ALL OK
python3 -m phi_core.stamp mymodel /path/to/mymodel  # new model repo
cd c_core && make check && make trap
```

## Consumers (thin by design)

Model repos own: frontend (weight formats), wiring (arch → op list),
calibration data, schedule search, demos/figures/docs. Everything else
— lattice, scales, lowerings, gates, stamper — is inherited here.

## License

GPLv3 — see [LICENSE](LICENSE).
