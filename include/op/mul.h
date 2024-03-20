#ifndef __MUL_H
#define __MUL_H

#include <stddef.h>
#include "seagull_op.h"
union xnn_f32_mul_minmax_params {
  struct {
    float min;
    float max;
    } scalar;
};

void xnn_f32_vmul_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_vmul_minmax_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

void op_test_mul_run(int batch_size);

#endif