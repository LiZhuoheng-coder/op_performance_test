#ifndef __LRELU_H
#define __LRELU_H

#include <stddef.h>
#include "seagull_op.h"

union xnn_f32_lrelu_params {
  struct {
    float slope;
  } scalar;
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    XNN_ALIGN(16) float slope[4];
  } sse;
  struct {
    XNN_ALIGN(32) float slope[8];
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  struct {
    XNN_ALIGN(8) float slope[2];
  } wasmsimd;
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
};

void xnn_f32_vlrelu_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_vlrelu_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]);

void op_test_lrelu_run(int batch_size );

#endif