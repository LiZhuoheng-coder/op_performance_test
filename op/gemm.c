#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "op/gemm.h"


void op_test_gemm_run(int batch_size)
{
    // 准备测试数据 
    srand(time(NULL));
    int alignment = 64; // 对齐字节

    size_t mr = 1;
    size_t nc = batch_size;
    size_t kc = batch_size;

    float* a = (float*)aligned_alloc(alignment, batch_size * batch_size * sizeof(float));
    float* w = (float*)aligned_alloc(alignment, batch_size * batch_size * sizeof(float));
    float* c = (float*)aligned_alloc(alignment, batch_size * batch_size * sizeof(float));

    // 确保内存分配成功
    if (a == NULL || w == NULL || c == NULL) {
        perror("aligned_alloc failed");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < batch_size * batch_size; i++)
    {
        a[i] = rand() % 100;
        w[i] = rand() % 100;
    }

    union xnn_f32_default_params params;
\
    printf("OP: GEMM 1X4 Test \n");

    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_gemm_ukernel_1x4__scalar(mr, nc, kc, a, 1, w, c, batch_size, batch_size, &params);
    int64_t end = clock();
    
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_gemm_ukernel_1x4__rvv_u1v(mr, nc, kc, a, 1, w, c, batch_size, batch_size, &params);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    printf("-------------------------------------\n");
    
    free(a);
    free(w);
    free(c);

}

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
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) //some default param
{
	assert(mr != 0);
	assert(mr <= 1); // max process 1 row
	assert(nc != 0);
	assert(kc != 0);
	assert(kc % sizeof(float) == 0);
	assert(a != NULL);
	assert(w != NULL);
	assert(c != NULL);

	const float* a0 = a; // matrix a 0th row pointer
	float* c0 = c; // 0th row start pointer
	size_t kcl = kc / sizeof(float);

	do {
		size_t vl = vsetvl_e32m1(nc); // vector length
		vfloat32m1_t vacc = vle32_v_f32m1(w, vl); // 1st row count
		w += vl;
		for(size_t k = 0; k < kcl ; k++){
			vfloat32m1_t vw = vle32_v_f32m1(w, vl);
			w += vl;
			vacc = vfmacc_vf_f32m1(vacc, *a0, vw, vl); // update 1st row count
			a0++;
		}
		vse32_v_f32m1(c0, vacc, vl); // store 1st row result
		if(nc >= 4){
      		c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
      		a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
		}
		nc -= vl;
	} while (nc != 0);
}

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
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}