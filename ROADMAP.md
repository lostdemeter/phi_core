# phi-core roadmap: from video substrate to general representation

End state: **phi IR is a credible alternative representation for open
models generally** — not a video-project substrate. "Credible" is
defined by acceptance criteria (§5), not by vibes. Everything below is
gated; each phase has an exit bar and a tripwire. Video projects were
scaffolding (cheap, visual, fast to verify) — the roadmap spends that
scaffolding deliberately.

## Where we are (measured, 2026-09-29)

- 4 geometric codecs: depth (DAV2), interpolation (RIFE), upscaling
  (Real-ESRGAN), generation (SD-1.5 UNet + DDIM demo). Parity 43–59dB
  or 0.9999 corr; FPU-free proven by trap; bit-exact across
  numpy/C/SIMD/CUDA where lowered.
- IR v0.1: 2 types, ~19 ops, closed set. Novelty intake 0,0,2,2 —
  the >3 tripwire never fired.
- Composition L1–L3 demonstrated (51dB triple-direct seam dividend).
- Known gaps: numpy reference is hours-scale at 64px+; VAE/CLIP/
  scheduler live at float boundaries; log-domain (DAV2) accumulation
  designed but not lowered; calibration is per-distribution manual work.

## Phase 1 — Performance parity (engineering, not research)

Goal: the reference path stops being the bottleneck for demos.
- Torch bit-exact backend for diffusion (RIFE precedent: 59x, 0 diffs).
- Persistent-kernel CUDA pass over the priced backlog (traffic wall,
  launch overhead, ~1% utilization): fused concat+conv, RDB/attention
  persistence, shared-memory staging. Target 10–50x (itemized).
- 512px DDIM-50 demo, fully integer loop; 720p60 RIFE+ESRGAN chain
  toward realtime.
- Exit: 512px image in <60s integer end-to-end; 720p pair <100ms.
- Tripwire: if any speedup breaks a gate twice (not once — once is a
  bug, twice is a discipline smell), freeze speed work and audit the
  gate harness first.

## Phase 2 — Boundary absorption (shrink the float)

Goal: float lives only where it must (sensor/display physics).
- VAE decoder into the IR (convs + GroupNorm + SiLU — all covered
  ops; ~0 new, mostly wiring + calibration). Kills the biggest
  display-boundary dependency.
- Log-domain accumulation as second IR family (DAV2's datapath:
  phi_add/sub LUTs). New substrate work per opcode is bounded and
  visible; the two families share types, LUTs, gates, tooling.
- Scheduler/timestep precompute stays offline (it is genuinely
  boundary: pure functions of scalars), documented as such.
- Exit: text-to-image with float ONLY in CLIP-offline + RGB-out.
- Tripwire: if VAE needs >2 new opcodes, the opcode census (§IR)
  is wrong — stop and re-census before continuing.

## Phase 3 — Coverage (the funnel at scale)

Goal: prove wiring-only risk transfers across domains. One model per
family, novelty-budgeted, in this order (coverage-per-effort):
1. **LLaMA-style transformer** (language flag): RMSNorm + SwiGLU +
   RoPE + GQA, all composing from existing primitives. Budget: ≤2 new.
   Unlocks the LLM world; text encoders (CLIP-L) fall out as a corollary.
2. **Mamba/SSM** (genuinely-new lowering stress test): selective scan
    is the first op class outside current patterns. Budget: ≤3 new.
    Worth it precisely as IR stress (like warps once were).
    STATUS 2026-09-30: PORTED — mamba-130m full integer port gated
    (block 72dB, tiled scan 94.6dB, drift sublinear per contraction
    theory, 50-token demo + transcript). Novelty K=1 (`scan_step` +
    tiled lowering, promoted with generality proof); K2 log1pexp
    rejected with evidence; abar/softplus compose from primitives.
3. **Modern CNN details**: ConvNeXt-V2 GRN, deformable convs
    (data-dependent indexing — the other genuinely-new class). Bounded.
    STATUS 2026-09-30: PORTED — ConvNeXtV2-atto full integer classify
    (3/3 top-1, logits 27.6/27.6/34.3dB; block 46.3dB) + Deformable-DETR
    MSDeformAttn 51.8dB on real backbone features. Novelty K=0 (+1 erf
    table, precedent): deformable sampling composes (coords-as-inputs),
    GRN/frozen-BN compose, GELU exact via PHI LUT (tanh rejected with
    evidence). Honest substitution recorded: deformable ATTENTION for
    deformable CONV (InternImage needs custom CUDA + remote code;
    op class identical). Calibration lessons: per-block M-dict at
    1000x GRN dynamic range; downsample M covers SOURCE peak.
    Phase-3 item CLOSED — coverage exit (7 families) holds.
- Process (already adopted): mandatory RECON novelty report —
  required N / shared M / new K + justification; promote by evidence
  (≥2 consumers or 1 + generality proof), never speculatively.
- Exit: 7 families green with per-model novelty ledger public.
- Tripwire: any model exceeding its novelty budget triggers a
  framework review (missing abstraction?) before the overrun is accepted.

## Phase 4 — Self-service (third parties port without us)

Goal: the authors are no longer the bottleneck.
- Stamper maturity: stamped repo passes its own gates with ZERO hand
  edits (skeleton + fingerprint asserts + hook calibrator + demo
  template + trap harness, all generated).
- Conformance suite: `phi-conform` — lattice/codec/gate checks any
  model repo must pass to claim IR compatibility; versioned with IR.
- IR v1.0 freeze: opcode set + type system + LUT formulas frozen with
  a compatibility promise (v1.x additive-only); evolution policy in
  writing (deprecation = 2-version warning + auto-migration).
- Metric that matters: **time-to-first-parity for a competent
  outsider** (target: days, measured by actually watching one).
- Exit: one external contributor lands a model port with review-only
  involvement from us.
- Tripwire: if the stamper needs per-model custom code twice, it is
  not a stamper — generalize or delete it.

## Phase 5 — Distribution (the alternative representation)

Goal: publishing and consuming phi-native models is boring.
- Registry: content-addressed artifacts (weights+scales+manifest),
  fingerprint-first downloads (wrong-model class impossible by
  construction), calibration datasets versioned alongside.
- Hub story: one-command conversion safetensors→phi (frontend +
  auto-calibration + gates) for supported archs; results cached
  publicly so conversion runs once globally.
- Docs: porting guide (the playbook written down), IR reference,
  perf cookbook, failure gallery (every falsified path, honored).
- Exit (the acceptance criteria for the whole roadmap):
  1. ≥3 model families with realtime integer paths on commodity GPU.
  2. ≥1 port completed entirely outside the core team.
  3. Zero float in every shipped loop (trap-proven, all repos).
  4. IR v1.x stable ≥6 months with additive-only changes.
  5. A demo an outsider calls "a real alternative" (their words).

## Cross-cutting (never phase-gated out)

- Gates travel with every speedup (every number re-verified after
  each optimization — standing rule since RIFE torch port).
- Rejected alternatives stay in ledgers with attribution.
- Fingerprint-before-math, ops-before-pipes, locators-alongside,
  traps-at-introduction, slow-exact-first. The process IS the product
  until Phase 4 says otherwise.
- Non-goals: training in-IR (inference representation only);
  beating float SOTA absolute quality (parity + structure is the
  claim); silent hallucination anywhere (abstain/average beats
  invention where uncertain).

*Part of the TruthSpace Geometric LCM research project. GPLv3.*
