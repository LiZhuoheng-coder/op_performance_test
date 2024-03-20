#ifndef __GEMM_H
#define __GEMM_H

#include <stddef.h>
#include "seagull_op.h"

void op_test_gemm_run(int batch_size);
void op_test_gemm_rule_1x4_run(int batch_size);
void op_test_gemm_rule_2x4_run(int batch_size);

union xnn_f32_default_params {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  struct {
    int32_t mask_table[14];
  } avx;
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
};

// ReLU: serves to differentiate pointer types for micro-kernels with fused ReLU activation.
union xnn_f32_relu_params {
    char _; // Dummy member variable to comply with the C standard
};


void op_test_gemm_run(int batch_size);

void xnn_f32_gemm_ukernel_1x4__rvv_u1v(
    size_t mr, // max row
    size_t nc, // next col
    size_t kc, // dimention inside
    const float* restrict a, // pointer to matrix A
    size_t a_stride, // row direction span for A, num of elem which need to skip from begin of one row to next row
    const float* restrict w, // pointer to matrix B
    float* restrict c, // pointer to matrix C, the result of AxB
    size_t cm_stride, // row direction span for C
    size_t cn_stride, // col direction span for C
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]); //some default param


void xnn_f32_gemm_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_gemm_relu_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_gemm_relu_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_gemm_relu_ukernel_2x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]);

void xnn_f32_gemm_relu_ukernel_2x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]);

#endif
