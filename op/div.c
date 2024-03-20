#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "op/div.h"



void op_test_div_run(int batch_size)
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

    union xnn_f32_div_minmax_params params;
    params.scalar.min = -100.0;
    params.scalar.max = 100.0;

    printf("OP: Div Test \n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_vdiv_minmax_ukernel__scalar_u2(batch_size * sizeof(float), input_a, input_b, output, &params);
    int64_t end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_vdiv_minmax_ukernel__rvv_u1v(batch_size * sizeof(float), input_a, input_b, output, &params);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    printf("-------------------------------------\n");

    free(input_a);
    free(input_b);
    free(output);

}

void xnn_f32_vdiv_minmax_ukernel__rvv_u1v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_div_minmax_params *params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t size = batch / sizeof(float);

  size_t vl;
  for (size_t i = 0; i < size; i += vl) {
    // 动态设置向量长度
    vl = vsetvl_e32m1(size - i);

    // 加载输入向量
    vfloat32m1_t va = vle32_v_f32m1(input_a + i, vl);
    vfloat32m1_t vb = vle32_v_f32m1(input_b + i, vl);

    // 执行向量除法
    vfloat32m1_t vacc = vfdiv_vv_f32m1(va, vb, vl);

    // 应用最小值约束
    vacc = vfmax_vf_f32m1(vacc, params->scalar.min, vl);

    // 应用最大值约束
    vacc = vfmin_vf_f32m1(vacc, params->scalar.max, vl);

    // 存储结果
    vse32_v_f32m1(output + i, vacc, vl);
  }
}

void xnn_f32_vdiv_minmax_ukernel__scalar_u2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_div_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    input_b += 2;

    float vacc0 = va0 / vb0;
    float vacc1 = va1 / vb1;


    vacc0 = math_max_f32(vacc0, voutput_min);
    vacc1 = math_max_f32(vacc1, voutput_min);

    vacc0 = math_min_f32(vacc0, voutput_max);
    vacc1 = math_min_f32(vacc1, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    const float vb = *input_b;
    float vacc = va / vb;
    vacc = math_max_f32(vacc, voutput_min);
    vacc = math_min_f32(vacc, voutput_max);
    *output = vacc;
  }
}