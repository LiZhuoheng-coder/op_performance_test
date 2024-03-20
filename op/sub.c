#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "op/sub.h"

void op_test_sub_run(int batch_size)
{
    // 准备测试数据
    srand(time(NULL));
    int alignment = 64; // 对齐字节

    float* input_a = (float*)aligned_alloc(alignment, batch_size * sizeof(float));
    float* input_b = (float*)aligned_alloc(alignment, batch_size * sizeof(float));
    float* output = (float*)aligned_alloc(alignment, batch_size * sizeof(float));

    // 确保内存分配成功
    if (input_a == NULL || input_b == NULL || output == NULL)
    {
        perror("aligned_alloc failed");
        exit(EXIT_FAILURE);
    }


    for (size_t i = 0; i < batch_size; i++)
    {
        input_a[i] = rand() % 100;
    }

    for (size_t i = 0; i < batch_size; i++)
    {
        input_b[i] = rand() % 100;
    }

    union xnn_f32_sub_minmax_params params;
    params.scalar.min = -100.0;
    params.scalar.max = 100.0;

    printf("OP: Sub Test \n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_vsub_minmax_ukernel__scalar_u8(batch_size * sizeof(float), input_a, input_b, output, &params);
    int64_t end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_vsub_minmax_ukernel__rvv_u8v(batch_size * sizeof(float), input_a, input_b, output, &params);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    printf("-------------------------------------\n");

    free(input_a);
    free(input_b);
    free(output);
}

void xnn_f32_vsub_minmax_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_sub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    const float vb2 = input_b[2];
    const float vb3 = input_b[3];
    const float vb4 = input_b[4];
    const float vb5 = input_b[5];
    const float vb6 = input_b[6];
    const float vb7 = input_b[7];
    input_b += 8;

    float vacc0 = va0 - vb0;
    float vacc1 = va1 - vb1;
    float vacc2 = va2 - vb2;
    float vacc3 = va3 - vb3;
    float vacc4 = va4 - vb4;
    float vacc5 = va5 - vb5;
    float vacc6 = va6 - vb6;
    float vacc7 = va7 - vb7;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);
    vacc2 = math_max_f32(vacc2, voutput_min);
    vacc3 = math_max_f32(vacc3, voutput_min);
    vacc4 = math_max_f32(vacc4, voutput_min);
    vacc5 = math_max_f32(vacc5, voutput_min);
    vacc6 = math_max_f32(vacc6, voutput_min);
    vacc7 = math_max_f32(vacc7, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);
    vacc2 = math_min_f32(vacc2, voutput_max);
    vacc3 = math_min_f32(vacc3, voutput_max);
    vacc4 = math_min_f32(vacc4, voutput_max);
    vacc5 = math_min_f32(vacc5, voutput_max);
    vacc6 = math_min_f32(vacc6, voutput_max);
    vacc7 = math_min_f32(vacc7, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output[2] = vacc2;
    output[3] = vacc3;
    output[4] = vacc4;
    output[5] = vacc5;
    output[6] = vacc6;
    output[7] = vacc7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      const float vb = *input_b++;
      float vacc = va - vb;
      vacc = math_max_f32(vacc, voutput_min);
      vacc = math_min_f32(vacc, voutput_max);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}

void xnn_f32_vsub_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_sub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);
 
  size_t vl;

  size_t size = batch / sizeof(float);
  // 逐批处理
  for (size_t i = 0; i < size; i += vl) {
    // 动态计算向量长度（VL），这次基于剩余的元素数量
    size_t vl = vsetvl_e32m8(size - i);

    // 加载输入向量
    vfloat32m8_t va = vle32_v_f32m8(input_a + i, vl);
    vfloat32m8_t vb = vle32_v_f32m8(input_b + i, vl);

    // 执行向量减法
    vfloat32m8_t vacc = vfsub_vv_f32m8(va, vb, vl);

    // 应用最小值约束
    vfloat32m8_t vmin = vfmv_v_f_f32m8(params->scalar.min, vl); // 广播最小值到向量
    vacc = vfmax_vv_f32m8(vacc, vmin, vl);

    // 应用最大值约束
    vfloat32m8_t vmax = vfmv_v_f_f32m8(params->scalar.max, vl); // 广播最大值到向量
    vacc = vfmin_vv_f32m8(vacc, vmax, vl);

    // 将结果存储回输出数组
    vse32_v_f32m8(output + i, vacc, vl);
  }
}

