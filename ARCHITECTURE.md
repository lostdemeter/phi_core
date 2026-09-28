# phi-core architecture (the ffmpeg mapping, made concrete)

```
phi-core/
├── phi_core/lattice.py    # codec: triples <-> fixed, LUTs, tdiv, bit_length
├── phi_core/calibrate.py  # offline ranges -> frozen scales (m_of, hooks)
├── phi_core/stamp.py      # starter kit (stamps model repos)
├── IR.md                  # opcode spec: 2 types, ~19 ops, schedule rules
├── c_core/                # C scalar/OpenMP + SIMD + CUDA lowerings
├── tests/test_lattice.py  # 0-diff extraction proof vs originals
└── luts/                  # GITIGNORED baked tables (auto-built on first use)
```

## Data flow (one direction, no cycles)

```
weights (any format)
  -> frontend (model repo, offline): triples + shapes
  -> calibration (phi_core.calibrate + model maxima): frozen M scales
  -> executor (model repo): IR op list over triples/fixed
  -> substrate (phi_core / c_core): numpy | C | SIMD | CUDA
  -> triples out -> decode at display boundary
```

Float exists ONLY at the two boundaries (sensor encode, display
decode) and offline (weight ingest, calibration, LUT builds). The
runtime loop is integers + gathers on every substrate — structurally
(no float type in IR.md), with `fpu_trap`-pattern gates per model repo.

## Why generation is sound here (and not elsewhere)

Lowering preserves values **bit-exactly** because integer sums are
order-free. In float-land every backend is a new verification project
(order changes values); here backends are interchangeable by proof,
checked by gates. That single property is what makes "automatically"
honest: the framework generates *lowerings*, never *approximations*.

## What the package does NOT do (model repos own these)

Frontends (weight formats), wiring (arch -> op list), calibration
data, schedule search, demos/figures/docs. See IR.md "Per-model
remainder". The package stays small on purpose: lattice, scales,
lowerings, gates, stamper. Everything else is craft, per model.
