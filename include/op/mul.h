#ifndef __MUL_H
#define __MUL_H

#include <stddef.h>
#include "seagull_op.h"

union xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float min[4];
    XNN_ALIGN(16) float max[4];
  } sse;
  struct {
    XNN_ALIGN(32) float min[8];
    XNN_ALIGN(32) float max[8];
    int32_t mask_table[14];
  } avx;
  struct {
    float min;
    float max;
    int8_t sign_mask;
  } avx512vnni;
  struct {
    float min;
    float max;
    int8_t sign_mask;
  } avxvnni;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float min[2];
    XNN_ALIGN(8) float max[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

void xnn_f32_vmul_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_vmul_minmax_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

void op_test_mul_run(int batch_size);

#endif