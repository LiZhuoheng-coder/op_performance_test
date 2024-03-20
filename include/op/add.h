#ifndef __ADD_H
#define __ADD_H

#include <stddef.h>
#include "seagull_op.h"

void op_test_add_run();

union xnn_f32_add_minmax_params {
    struct{
    float min;
    float max;	  
    } scalar;
};

void xnn_f32_vadd_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_vadd_minmax_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]);

#endif
