# Phi IR — opcode specification (v0.1, extracted from three ports)

A model in phi IR is an **ordered op list** over **typed tensors**.
There are exactly two tensor types (this is the whole type system):

- `trip[C...]`: lattice triples `(int8 s, int32 e, uint8 z)`, values
  `s * PHI^((e - 32768) / 512)`. Untagged (scales live in the scale table).
- `fix[C...] @ m`: int64 2^-18 counts **tagged with lattice scale m**.
  The tag is load-bearing: every producer sets it, every consumer
  asserts it. There is no untagged fixed tensor anywhere.

Scales (`m: int`) live in a side table, frozen at calibration. No float
type exists in the IR — FPU-free and torch-free fall out structurally:
unrepresentable, not audited.

## Opcodes (closed set — additions require all-substrate gates)

Bridge (triples <-> fixed):

| op | signature | contract |
|---|---|---|
| `to_fixed` | trip[*] × m → fix[*] @ m | elementwise LUT; saturate/clip per spec |
| `from_fixed` | fix[*] @ m → trip[*] | 15-bit mantissa + COARSE/FINE; asserts < 2^53 |
| `rescale` | fix[*] @ m1 × m2 → fix[*] @ m2 | the ONLY scale changer; explicit, greppable |
| `tdiv` | fix × int → fix (same tag) | trunc toward zero, both signs |

Compute (shapes: HWC, OHWI weights):

| op | signature | contract |
|---|---|---|
| `conv` | trip × OHWI × bias? × m_acc × (kh,stride,pad) → trip | per-product bridge, int64 accumulate (order-free) |
| `deconv` | trip × OHWI × bias? × m_acc × (kh,stride,pad) → trip | scatter-add, bias ADDS (see bias-drop lesson) |
| `warp` | fix @ m × flow[2^-14] → fix @ m | nihui-exact: unclamped alpha, edge replicate |
| `interp` | fix @ m × 2^k → fix @ m | dyadic-only shifts; non-dyadic is a LOWERING ERROR, not fallback |
| `sigmoid` | trip[*] → trip[*] | EXPACT gather + LUT, any input range, U1 output |
| `prelu` | trip[*] × slope → trip[*] | exact select + triple multiply |
| `pool_avg` | fix @ m → fix @ m (1,1,C) | int64 sum + tdiv (same tag in/out) |
| `tmul` | trip × trip → trip | exact (sign-XOR, exp-add clipped, zero-or) |
| `binop` | fix @ m × fix @ m → fix @ m (add/sub/rsub) | tags MUST match (assert); mul is `tmul` |
| `clip` | trip × const × const → trip | exact lattice compare |
| `neg` | trip → trip | sign negate |

Moves (no arithmetic — exact by construction):

`concat`, `split`, `crop`, `pixelshuffle`, `nearest2`, `transpose`.
Moves are never gated numerically (nothing to be wrong); they are
gated structurally (shapes in the op list).

## Schedules (semantics, not tuning)

Every lowering records its schedule explicitly: loop order, tiling
(pixel/co blocks + sizes), thread mapping (OpenMP/CUDA grid), lane
mapping (SIMD lanes). Rationale (user hunch, adopted as design rule):
schedules may carry semantics, so they live versioned alongside the op
list — never buried in codegen. Integer order-freedom makes all
currently-known schedules value-identical (gated); if a future op is
order-sensitive, its schedule becomes part of its contract HERE.

## Backends (substrates — one lowering each, shared by all models)

| backend | status | notes |
|---|---|---|
| numpy reference | ✅ proven ×3 | slow-exact; the spec executable |
| C scalar/OpenMP | ✅ proven | fused + unfused; bit-exact |
| SIMD (AVX512) | ✅ proven | bridge/gather/convectorized; specimens in `c_core` |
| CUDA | ✅ proven | thread-per-output; bit-exact; specimens in `c_core` |

New substrate = new lowering per opcode + the shared gate suite.
New opcode = all-substrate implementations + gates before any model
may use it. That is the whole framework rule, and it fits in this file.

## Per-model remainder (never generates — the craft)

1. **Frontend**: weight-format loader (offline only). Fingerprint asserts
   mandatory (executable spec of the artifact).
2. **Wiring**: arch → op list (+ shapes). Gated shape-first vs oracle.
3. **Calibration data**: representative inputs; hook maxima; frozen M.
4. **Schedule search**: tiling/occupancy per target (tuner output =
   schedule data, cached).
5. **Demo + figure**: the cheapest end-to-end gate; committed visuals.
