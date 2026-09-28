/* Fused phi-conv (drop-in bit-exact replacement for conv.c phi_conv).
 *
 * Same math, restructured for traffic: per (pixel, co) the tap loop
 * streams (s,e) pairs straight through the bridge into one int64
 * accumulator — product triples are NEVER materialized (the (Nt,TAPS,cc)
 * tensor was ~30 bytes/MAC of traffic). Integer sums are order-free,
 * so this is bit-identical to the triple-loop form (gated, not claimed).
 */
#pragma once
#include "conv.h"

/* Identical signature/semantics to phi_conv (see conv.h). */
void phi_conv_fused(const trip_t *in, int H, int Wd, int Cin, const cW_t *cw,
                    const trip_t *bias_or_null, int m_out, int stride, int pad,
                    trip_t *out, int *Ho, int *Wo);

/* SIMD variant: OpenMP over pixels, AVX512 over co-blocks of 8, scalar
 * tails. Bit-identical to phi_conv_fused (gated, not claimed). */
void phi_conv_fused_simd(const trip_t *in, int H, int Wd, int Cin,
                         const cW_t *cw, const trip_t *bias_or_null, int m_out,
                         int stride, int pad, trip_t *out, int *Ho, int *Wo);
