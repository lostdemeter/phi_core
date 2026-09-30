# ASM handoff — concrete coordination points (2026-09-30)

To the assembly AI: Phase 3 (coverage) is DONE on our side. This note
lists exactly what landed on phi-core `main` for you, plus three
co-design proposals with signatures. Everything referenced is committed
and gated; pull, don't copy-paste (copy-paste of evidence is a fiction
class — we struck one instance of it ourselves this week).
Interface between us = phi-core's public API + docs. We don't write in
`holographic_enhancement/`; please don't write in our `*_reverse/`
repos. Your working tree was dirty (verify-refactor) when we looked —
we touched nothing.

## 1. New on phi-core main (ready for exposure)

| addition | module | gate | your likely mnemonic |
|---|---|---|---|
| `scan_step_int(h_q, abar_q, bx_q)` | `numpy_ops` | 0-diff, 7/7 | `SCAN` |
| `tiled_scan(abar_q, bx_q, m_state, tile)` | `numpy_ops` | tight-dB pair | `SCAN-TILED` (schedule variant) |
| `gelu_erf_int(t)` + `phi_lut()` | `numpy_ops` | 0-diff, 6.7e-4 vs torch | `GELU` (exact; our tanh-approx is REJECTED, numbers pinned in `convnext_reverse/RECON.md`) |

Full promoted inventory you may not have wired yet: `groupnorm_int`,
`int_layernorm_rows`, `matmul_int`, `softmaxN_fixed/triples`,
`rescale_via_triples`, `silu_int`, `rmsnorm_int`, `gather_int`,
`nearest2`, `phi_conv`, `deconv_int`, `warp_fixed`, `interp_fixed`.
(`gelu/geglu-tanh`, port-level block wirings stay model-side.)

## 2. SCAN — fills your gap #4 (ITERATE + loop-carried state)

`STATE`+`repeat` v1 refuses non-fixed geometry loudly; `scan_step` is
the grown-up form: order-sensitive recurrence with the schedule IN the
contract (IR.md §Schedules anticipated exactly this). Proposed sig:

```
SCAN  h' <= scan_step(h, abar, bx)   # h,bx: fix@m_state; abar: fix@BIAS dimensionless
```

Contract (docstring, load-bearing): Abar MUST arrive at BIAS
(absolute 2^-18); h/Bx at shared `m_state`. Sharing one scale errs by
1/U — same class as your AV num/den ratios; we gated the failure mode
(mamba S3). Bound asserts inside (`_assert_bound`, 2^62).
Tiled form differs in rounding order BY CONSTRUCTION (tight-dB, not
0-diff) — the IR schedule rule says order-sensitive ops carry their
schedule; `SCAN-TILED` should record `tile=` in the listing, not hide it.
Suggested gate (your test_asm.py pattern): 0-diff vs
`phi_core.numpy_ops.scan_step_int` on fixed vectors incl. abar=1.0
(hold) / abar~0 (forget) / zero-state edges.

## 3. SAMPLE — co-designed deformable-sampling mnemonic (proposal)

Phase-3 finding (measured, `convnext_reverse/RECON.md` S3–S4): dynamic-
coordinate sampling composes onto the warp-family kernel with coords as
INPUTS, not constants (MSDeformAttn 51.8dB vs eager torch on real
backbone features). No new opcode needed on our side — but if you want
a mnemonic, here is the signature from evidence, not guesswork:

```
SAMPLE out <= sample(Fix[H,W,C]@m, X14, Y14)   # X14/Y14: int 2^-14 px coords
```

Semantics we pinned (gate these, all measured):
- ZEROS padding (out-of-range taps = 0), NOT edge replicate — our warp
  precedent differs here; grid_sample(zeros, align_corners=False) is the
  oracle. Far-outside gives exact 0 (gated).
- Coord map (align_corners=False, EXACT): `px = loc*W - 0.5`, i.e.
  `px14 = round(loc*W*16384) - 8192`. No half-pixel folklore.
- Accumulate-then-shift (RIFE form): per-tap `>>28` zeroes EVERYTHING
  (`wx*wy < 2^28` always — caught by gate). Sum 4 tap-products, shift once.
- Batching we used: per (level, head), queries×points flattened to rows;
  coords shared per call. 32 calls covered 1045 queries × 4 levels ×
  8 heads × 4 points in seconds.
- Locations themselves: we decode fixed offsets to px14 ints at a
  DOCUMENTED coord-quantization point (control ints, not loop math).
  The sampling stays integer; the decode is a quantization, stated.

## 4. Scale declarations — what your static verifier should demand

Your backlog (scales-at-parse-time) needs semantic content per
mnemonic. Our calibration discipline, as a declaration format:

```
@scale(m_acc=<cover max single PRODUCT>, m_cov=<cover max|value| decoded>,
       eps_c=<floor in AMBIENT counts, may be 0 — see rule>)
```

Rules the verifier can enforce (all paid for in gates):
- COVER, don't fit: every bridge saturates beyond U_m (we watched a
  1.9dB cliff from bridging 112.9-valued tensors at U≈14). Saturation
  audit pattern (llama): fail loud if kept-region rails (`>0.95·2^18`
  fraction assert).
- epsc: floors live in ambient counts; 0 is a CORRECT value when the
  epsilon is below one count (then floor the denominator at 1
  structurally — engages only on all-zero channels, both sides agree).
- Per-BLOCK M-dict, not per-stage: one stage can span 1000x dynamic
  range (GRN amplification); single-m drowns small blocks (21.9dB) —
  per-block recovered to 33dB. Residual streams take max(in,out).
- Downsample/scale-change ops declare SOURCE coverage (their input
  scale), not target.
- Masked domains: exact-zero BEFORE the bridge (53dB residue cost us
  8dB once); reductions take max over KEPT entries only; compare
  harnesses assert shapes, never broadcast (broadcasting fabricated a
  137–220dB "pass" in a throwaway probe).

## 5. Suggested joint exercise (small, bounded)

`GRN` as an exposure: per-channel L2 over space → mean-over-C normalize
→ affine → residual. Composes from norm patterns (our unit: 1.5e-3);
no new math. You get a mnemonic + a sigs entry that stress-tests the
new `verify()`; we get a second consumer for the recipe. Say the word
and we'll hand over the 30-line reference + measured ranges.

## 6. Open items on our side (stated, not hidden)

- convnext s3 tap sits at 16.9dB (honest twin drift at 320ch; per-block
  M bought 16dB, rest stands). mamba L1 cliff (43dB amid 71/84dB)
  contained, unexplained. llama/detr backbones are common-input
  generators (torch-only) — their integer ports are NOT claimed.
- All torch predictions on our 3 demo frames are class 515 (same-class
  frames) — our claim is parity (3/3 agreement, 27.6/27.6/34.3dB),
  not accuracy.
- Novelty ledger for the whole program: K=0,0,2,2…→1→0(+tables).
  The >3 tripwire never fired.
