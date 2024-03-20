#ifndef __GEMM_H
#define __GEMM_H

#include <stddef.h>
#include "seagull_op.h"

void op_test_gemm_run(int batch_size);



// void xnn_f32_gemm_ukernel_1x4__scalar(
//     size_t mr,
//     size_t nc,
//     size_t kc,
//     const float* restrict a,
//     size_t a_stride,
//     const float* restrict w,
//     float* restrict c,
//     size_t cm_stride,
//     size_t cn_stride,
//     const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]);



#endif