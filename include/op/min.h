#ifndef __MIN_H
#define __MIN_H

#include <stddef.h>
#include "seagull_op.h"



void xnn_f32_vmin_ukernel__rvv_u1v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output//,
    // const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]
    );

void xnn_f32_vmin_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output//,
    // const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]
    );

void op_test_min_run(int batch_size);

#endif