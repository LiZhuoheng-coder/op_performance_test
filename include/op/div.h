#ifndef __DIV_H
#define __DIV_H

#include <stddef.h>
#include "seagull_op.h"


union xnn_f32_div_minmax_params {
  struct {
    float min;
    float max;
    } scalar;
};


extern void xnn_f32_vdiv_minmax_ukernel__rvv_u1v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_div_minmax_params *params);

extern void xnn_f32_vdiv_minmax_ukernel__scalar_u2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_div_minmax_params *params);

void op_test_div_run(int batch_size);

#endif
