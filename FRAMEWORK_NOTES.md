# Framework notes (cross-model lessons — model repos hold project-local notes)

## Model #4 lessons (DDColor census + zoo attempts — read before synthesizing)

- [2026-09-28] CENSUS: RIFE mostly LOWPASS-deep + EDGE-entries; ESRGAN
  215 LOWPASS / 133 EDGE with a clean c1(EDGE)->c5(LOWPASS) gradient;
  heads LOWRANK (fusionT62 erank 3, conv_last erank 2); identity 0.0
  everywhere (residuals live in arch, not kernels). Census code:
  phi_core/probe.py (spectrum/eff-rank/response/identity + classify).
- [2026-09-28] GAIN-PROFILING (new probe, important): 1% weight noise
  on ESRGAN conv_first -> 5.9dB (network gain ~10-50x, no norms to
  re-anchor). Consequence: synthesis must match learned weights to
  <<0.1%/layer — approximations (Gabor: -45dB ~= zeroing the layer)
  never had a chance. RANK-II SVD truncation of conv_last: -17.9dB
  (small weights x big features still matter). RULE: the zoo holds
  EXACT constructions (bicubic kernels, colorimetry, identity) and
  DATA-DERIVED procedures (SVD/behavioral, V20 pattern) — never
  approximate guesses. Perturbation-gain (1%-noise dB/layer) joins
  ablation as a standard probe: it measures MARGINAL sensitivity
  (what synthesis must beat), ablation measures TOTAL contribution.
- [2026-09-28] LATTICE-CONSISTENCY: the same gain math explains why
  0.05% lattice rounding lands at 48dB (gain x8 on tiny deterministic
  errors) while 1% noise lands at 6dB. Deterministic + tiny survives;
  random + 20x bigger doesn't. No paradox — linear regime throughout.

- [2026-09-28] ZOO-1 (bicubic, first exact construction): per-parity tap
  WINDOWS differ (even {-2..1}, odd {-1..2}) — a same-window version
  silently dropped a live tap (DC 1.024). Partition-of-unity gate now
  mandatory for resamplers. PIL differs from exact theory by 3.3e-3
  interior on its own: gate against theory (float64 direct), use PIL
  for display only. Lattice-vs-theory interior: 1.27e-3 smooth,
  ~3e-3 at sharp edges (mixing rounded extremes). Lesson: reference
  hierarchies (theory > library > third-party) must be explicit, or
  their gaps get misattributed to our construction.

- [2026-09-30] MASKED-CONTRACTS (from mamba S4/S6, now IR discipline):
  three rules that each caught a real fiction. (1) MASKED EXACT-ZERO:
  masked-out positions (conv pads, scan pre-history, attention sinks)
  must be structural zeros BEFORE the bridge, not small residues —
  a 53dB residue in conv prepad cost 8dB of block parity. (2)
  KEPT-ONLY VMAX: reductions over masked domains take max over kept
  entries only; the full-tensor max leaks attenuated masked values
  into every output. (3) SHAPE-STRICT COMPARE (cmpdb pattern): parity
  harnesses must assert shapes, never broadcast — numpy broadcasting
  fabricated a 137–220dB "pass" in a throwaway probe (gates were
  always clean; the probe lied, not the model). Corollary, the
  fourth fiction class: CROSS-MODEL COPY-PASTE of evidence (llama's
  determinism/causal numbers briefly cited in mamba's S6 record —
  caught on review, struck). Agreement counts in greedy twins are
  chaos-dominated: report flip point + pre-flip dB, never bare
  agreement (a ~50dB-level numeric shift moved the flip 9→2 with
  identical code discipline). Rerun determinism (bit-for-bit replay)
  is the honest determinism gate.
