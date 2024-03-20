#ifndef __SIGMOID_H
#define __SIGMOID_H

#include "seagull_op.h"

union xnn_f32_sigmoid_params {
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c1;
    float one;
    float denorm_cutoff;
  } scalar_rr2_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c2;
    float one;
    float denorm_cutoff;
  } scalar_rr2_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float one;
    float denorm_cutoff;
  } scalar_rr2_p5;
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c1;
    float denorm_cutoff;
  } neon_rr2_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c2;
    float denorm_cutoff;
  } neon_rr2_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2_hi;
    float ln2_lo;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neon_rr2_p5;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c1;
    float denorm_cutoff;
  } neonfma_rr1_lut2048_p1;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c2;
    float denorm_cutoff;
  } neonfma_rr1_lut64_p2;
  struct {
    float magic_bias;
    float minus_log2e;
    float ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float denorm_cutoff;
  } neonfma_rr1_p5;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float sign_mask[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) uint32_t index_mask[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float one[4];
    XNN_ALIGN(16) float denorm_cutoff[4];
  } sse2_rr2_lut64_p2;
  struct {
    XNN_ALIGN(16) float sign_mask[4];
    XNN_ALIGN(16) float magic_bias[4];
    XNN_ALIGN(16) float log2e[4];
    XNN_ALIGN(16) float minus_ln2_hi[4];
    XNN_ALIGN(16) float minus_ln2_lo[4];
    XNN_ALIGN(16) float c5[4];
    XNN_ALIGN(16) float c4[4];
    XNN_ALIGN(16) float c3[4];
    XNN_ALIGN(16) float c2[4];
    XNN_ALIGN(16) float c1[4];
    XNN_ALIGN(16) float one[4];
    XNN_ALIGN(16) float denorm_cutoff[4];
  } sse2_rr2_p5;
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2_hi[8];
    XNN_ALIGN(32) float minus_ln2_lo[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float one[8];
    XNN_ALIGN(32) float two[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
    int32_t mask_table[14];
  } avx_rr2_p5;
  struct {
    XNN_ALIGN(32) float sign_mask[8];
    XNN_ALIGN(32) float magic_bias[8];
    XNN_ALIGN(32) float log2e[8];
    XNN_ALIGN(32) float minus_ln2[8];
    XNN_ALIGN(32) float c5[8];
    XNN_ALIGN(32) float c4[8];
    XNN_ALIGN(32) float c3[8];
    XNN_ALIGN(32) float c2[8];
    XNN_ALIGN(32) float c1[8];
    XNN_ALIGN(32) float one[8];
    XNN_ALIGN(32) float denorm_cutoff[8];
    int32_t mask_table[14];
  } avx2_rr1_p5;
  struct {
    uint32_t sign_mask;
    float magic_bias;
    float log2e;
    float minus_ln2;
    float c3;
    float c2;
    float one;
    XNN_ALIGN(64) float table[16];
  } avx512_rr1_lut16_p3;
  struct {
    uint32_t sign_mask;
    float magic_bias;
    float log2e;
    float minus_ln2_hi;
    float minus_ln2_lo;
    float c2;
    float c1;
    float one;
    XNN_ALIGN(64) float table_lo[16];
    XNN_ALIGN(64) float table_hi[16];
  } avx512_rr2_lut32_p2;
  struct {
    uint32_t sign_mask;
    float log2e;
    float minus_ln2;
    float c5;
    float c4;
    float c3;
    float c2;
    float c1;
    float one;
  } avx512_rr1_p5;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) uint32_t index_mask[2];
    XNN_ALIGN(8) float ln2_hi[2];
    XNN_ALIGN(8) float ln2_lo[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float denorm_cutoff[2];
  } wasmsimd_rr2_lut64_p2;
  struct {
    XNN_ALIGN(8) float magic_bias[2];
    XNN_ALIGN(8) float minus_log2e[2];
    XNN_ALIGN(8) float ln2_hi[2];
    XNN_ALIGN(8) float ln2_lo[2];
    XNN_ALIGN(8) float c5[2];
    XNN_ALIGN(8) float c4[2];
    XNN_ALIGN(8) float c3[2];
    XNN_ALIGN(8) float c2[2];
    XNN_ALIGN(8) float c1[2];
    XNN_ALIGN(8) float one[2];
    XNN_ALIGN(8) float denorm_cutoff[2];
  } wasmsimd_rr2_p5;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

void xnn_f32_vsigmoid_ukernel__rvv(
  size_t batch,
  const float* input,
  float* output,
  const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]);

void op_test_sigmoid_run(int batch_size);

#endif