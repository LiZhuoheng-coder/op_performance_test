#ifndef __PRELU_H
#define __PRELU_H

#include <stddef.h>
#include "seagull_op.h"


void xnn_f32_prelu_ukernel__rvv_2x8(
  size_t rows,
  size_t channels,
  const float* restrict input,
  size_t input_stride,
  const float* restrict weights,
  float* restrict output,
  size_t output_stride);

void xnn_f32_prelu_ukernel__scalar_2x4(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride);

void op_test_prelu_run(int batch_size);


#endif