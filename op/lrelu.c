#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "op/lrelu.h"


void op_test_lrelu_run(int batch_size)
{
    // 准备测试数据 
    srand(time(NULL));
    int alignment = 64; // 对齐字节


    float* input = (float*)aligned_alloc(alignment, batch_size * sizeof(float));
    float* output = (float*)aligned_alloc(alignment, batch_size * sizeof(float));
    // 确保内存分配成功
    if (input == NULL || output == NULL) {
        perror("aligned_alloc failed");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < batch_size; i++)
    {
        input[i] = rand() % 100;
    }

    union xnn_f32_lrelu_params params;
    params.scalar.slope = 0.1;
    printf("OP: Leaky Relu Test \n");
    printf("=====================================\n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_vlrelu_ukernel__scalar_u4(batch_size * sizeof(float), input, output, &params);
    int64_t end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_vlrelu_ukernel__rvv_u8v(batch_size * sizeof(float), input, output, &params);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);

    free(input);
    free(output);
}


void xnn_f32_vlrelu_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
	assert(batch != 0);
	assert(batch % sizeof(float) == 0);
	assert(input != NULL);
	assert(output != NULL);

	const float vslope = params->scalar.slope;
	size_t size = batch / sizeof(float);

	do {
		const size_t n = vsetvl_e32m8(size);

		vfloat32m8_t in_u8v = vle32_v_f32m8(input, n);
		input += n;
		vbool4_t mask = vmflt_vf_f32m8_b4(in_u8v, .0f, n);
		vfloat32m8_t out_u8v = vfmul_vf_f32m8_m(mask, in_u8v, in_u8v, vslope, n);
		//vfloat32m8_t out_u8v = vfmax_vf_f32m8(in_u8v, .0f, n);
		vse32_v_f32m8(output, out_u8v, n);

		output += n;
		size -= n;
	} while (size > 0);
}

void xnn_f32_vlrelu_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vslope = params->scalar.slope;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    float vacc0 = vx0 * vslope;
    float vacc1 = vx1 * vslope;
    float vacc2 = vx2 * vslope;
    float vacc3 = vx3 * vslope;

    vacc0 = XNN_UNPREDICTABLE(vx0 < 0.0f) ? vacc0 : vx0;
    vacc1 = XNN_UNPREDICTABLE(vx1 < 0.0f) ? vacc1 : vx1;
    vacc2 = XNN_UNPREDICTABLE(vx2 < 0.0f) ? vacc2 : vx2;
    vacc3 = XNN_UNPREDICTABLE(vx3 < 0.0f) ? vacc3 : vx3;

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      float vacc = vx * vslope;
      vacc = XNN_UNPREDICTABLE(vx < 0.0f) ? vacc : vx;
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}