# The AI Codec Pipeline

*Composing geometric AI models like ffmpeg composes codecs — three
working examples, one shared integer language, measured end to end.*

## Thesis

Modern AI models are distributed as **opaque float graphs**: framework-
bound, FPU-mandatory, mutually unintelligible. Two models cannot share
intermediate results without decoding to pixels, re-encoding, and
crossing framework boundaries — every seam costs precision, code, and
speed.

The AI Codec Pipeline inverts this. Each model is reverse-engineered
into a **shared integer intermediate language** — the phi lattice —
where every value is `sign × φ^((exponent − 32768) / 512)` and runtime
is integer add/sub/compare/shift/XOR plus table gathers. Models become
**codecs**: a frontend (weight loader), an op list (arch wiring), and
standard gates. Pipelines compose codecs the way ffmpeg composes
filters: named triple streams, spec strings, runtime toggles. Once a
model is in the language, integer execution, CUDA kernels, and FPU-free
operation follow by lowering — not by re-engineering.

Three models prove it. Nothing below is proposed; everything is measured.

## The three codecs

### 1. DAV2 — depth (perception)
Monocular depth (DINOv2 ViT-S + DPT neck + head), rebuilt geometrically.
Full pipeline correlation **0.9999** vs HuggingFace; live webcam
**~95–100 FPS fp16 GPU**, ~15 FPS CPU-only; integer transformer layer
0.999996; C99 core bit-exact 0/64. Its datapath accumulates in the
**log domain** (phi_add/sub LUTs) — deliberately different from the
bridge below, and the framework holds both.

### 2. RIFE-v2.3 — interpolation (generation, time)
Bidirectional flow + context + fusion synthesis, all three nets in
integers. Float assembly matches the reference binary at **59.0 dB**;
full int chain **49.3 dB** vs float-mid (corr 0.99990); foreman
held-out **31.48 vs 31.54 dB** (+0.06 gap to the original); full 720p
video **38.28 dB** mids; worst frame **29.22 = 29.22 dB** identical to
the original's own failure. Fixed-point bridge datapath.

### 3. Real-ESRGAN ×4 — upscaling (generation, space)
RRDBNet (23 blocks, 16.7M params) in integers. Int-vs-torch **48–66 dB**
per frame; honest-protocol demo (bicubic-down in, original as GT);
16-frame 720p video **25.35 dB**; full 300-frame video scored. Same
bridge datapath as RIFE — the reuse that motivated the framework.

## The intermediate language

Two tensor types — the entire type system:

- `trip`: lattice triples `(int8 s, int32 e, uint8 z)`, untagged.
- `fix @ m`: int64 2^-18 counts **tagged with lattice scale m**.
  Producers set the tag; consumers assert it. There is no untagged
  fixed tensor, so the wrong-scale bug class (which cost us a real
  ×0.929 incident) is unrepresentable. The only scale-changer is an
  explicit, greppable `rescale`.

~19 opcodes, closed set: bridge (`to_fixed`, `from_fixed`, `rescale`,
`tdiv`), compute (`conv`, `deconv`, `warp`, `interp`, `sigmoid`,
`prelu`, `pool`, `tmul`, `binop`, `clip`, `neg`), moves (`concat`,
`split`, `crop`, `pixelshuffle`, `nearest2`, `transpose`). Full spec:
[IR.md](IR.md). New opcode = all-substrate implementations + gates
before any model may use it — the single framework rule that fits here.

No float type exists in the IR. FPU-free and torch-free are therefore
**structural** (unrepresentable, not audited): trap harnesses rig every
float primitive to raise during full chains (zero hits, repeatedly),
and the core runs with `torch` made unimportable.

## Composition semantics

Three levels, each demonstrated:

- **L1 — sequential.** RIFE-mid → ESRGAN-up as files. Works, additive
  cost, two lattice roundtrips + uint8 quantization at the seam.
- **L2 — seamless.** Triple-direct handoffs: no decode, no re-encode,
  unified scales where possible (same `m` = pointer pass). Measured
  dividend: **51 dB** preserved vs the PNG seam (~2–3 LSB) — the price
  of the shared representation, priced in dB, not claimed.
- **L3 — joint scheduling.** Dataflow execution across graphs (RIFE
  frame N+1 while ESRGAN upscales frame N). Throughput parallelism;
  true joint *architectures* remain research, correctly scoped out.

Toggles (ffmpeg `enable`-expression spirit): `"rife=on:esr=on:dav2=on"`
spec strings; OFF nodes forward declared bypasses; arbitrary N via
registry, no code changes. Proven: all three AIs in one pipeline with
every toggle combination asserted, including a backend (DAV2) that went
from unbaked-SKIP to live mid-program with zero framework changes.

Schedules are explicit, versioned data alongside the op list — never
buried in codegen. (Design rule adopted from a hunch: if execution
order ever carries semantics, the IR already has somewhere to put it.
Today all schedules are value-identical, gated; the representation
costs nothing either way.)

## Evidence ledger (all measured, all committed)

| claim | number | where |
|---|---|---|
| DAV2 full-pipe vs HF | 0.9999 corr | dav2_reverse |
| RIFE float assembly vs binary | 59.0 dB | rife_reverse |
| RIFE int chain vs float | 49.3 dB | rife_reverse |
| RIFE foreman gap to original | +0.06 dB | rife_reverse |
| RIFE 720p video mids | 38.28 dB | rife_reverse |
| ESRGAN int vs torch | 48–66 dB/frame | esrgan_reverse |
| ESRGAN 300f video | 25.35 dB | esrgan_reverse |
| Triple-seam dividend | 51 dB | esrgan_reverse/demo_compose |
| 3-AI toggles | all GO | esrgan_reverse/demo_compose3 |
| CUDA chain vs C | 0 diffs | esrgan_reverse |
| CUDA frame vs numpy-fidelity | bit-exact chain | esrgan_reverse |
| FPU trap, full chains | zero hits | all repos |
| torch-unimportable core run | proven | rife_reverse |

## Performance reality (measured, no projections)

Per-frame 720p ESRGAN: numpy ~16 min → C-OpenMP ~6 min → torch-int
397 s → integer-CUDA **~7 s** (59× over torch-int, ~140× over numpy).
RIFE pair: 598 s → 10 s torch (59×). Remaining gaps are quantified:
unfused traffic wall (measured flat), launch overhead (measured),
~1% GPU utilization (measured) — the next 10–50× is itemized
(persistent RDB kernels, fused concat+conv, shared-memory staging,
occupancy), not wished for. Realtime is engineering effort with a
priced backlog, and the gates travel with every speedup (every number
above was re-verified after each optimization).

## What generates vs what's manual (honest boundary)

Generates (framework-owned, proven 3×): integer execution on any
substrate, CUDA kernels from op templates, FPU-free/torch-free
structure, parity/trap/import gates, repo scaffolding, LUT baking.
Manual forever (the craft): weight-format frontends, arch→op wiring,
calibration data, schedule search, demos/figures. New exotic ops cost
one opcode with all emitters — bounded, visible, scheduled, never silent.

## Process (how the evidence was produced)

Theory-first, frozen gates, zero-promotion-valid: alternatives stay in
the ledger, rejected ones named. Fingerprint artifacts before math
(one wrong-model batch voided, never repeated). Op gates before pipe
runs. First-break locators alongside executors. Per-tap bars are
diagnostic; end-to-end dB is the gate. Compliance traps at constraint
introduction. Slow-exact reference first, fast mirror second,
bit-parity between. Demos assert (≥40 dB) and commit figures.
Snag-driven library notes (`LIBRARY_NOTES.md` per model repo) shape
the framework — six evidence-backed proposals banked so far, three
already built (starter kit, hook calibrator, fingerprint asserts).

## Roadmap

- [x] Three geometric codecs (depth, interpolation, upscaling)
- [x] phi-core package (lattice, IR, calibrate, compose, lowerings)
- [x] All three thinned onto phi-core (bit-exact, gated)
- [x] L2 composition demo + seam pricing + 3-AI toggles
- [ ] Model #4 on phi-core (the real framework test: wiring-only risk)
- [ ] Persistent-kernel CUDA (10–50×, priced backlog above)
- [ ] Realtime demonstration (the engineering summit, not research)
- [ ] Log-domain as second IR accumulation family (DAV2's datapath,
  designed for, not yet lowered)

*Part of the TruthSpace Geometric LCM research project. GPLv3.*
