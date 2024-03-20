#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "op/min.h"

void op_test_min_run(int batch_size)
{
    // 准备测试数据
    srand(time(NULL));
    int alignment = 64; // 对齐字节

    float* input_a = (float*)aligned_alloc(alignment, batch_size * sizeof(float) * sizeof(float));
    float* input_b = (float*)aligned_alloc(alignment, batch_size * sizeof(float) * sizeof(float));
    float* output = (float*)aligned_alloc(alignment, batch_size * sizeof(float) * sizeof(float));

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

    // union xnn_f32_default_params params;

    printf("OP: Min Test \n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_vmax_ukernel__scalar_u8(batch_size * sizeof(float), input_a, input_b, output);
    int64_t end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_vmax_ukernel__rvv_u1v(batch_size * sizeof(float), input_a, input_b, output);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    printf("-------------------------------------\n");

    free(input_a);
    free(input_b);
    free(output);
}

void xnn_f32_vmin_ukernel__rvv_u1v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output//,
    // const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]
    )
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);
  while (batch > 0) {
      size_t vl = vsetvl_e32m1(batch);       // 设置向量寄存器每次操作的元素个数
      vfloat32m1_t va = vle32_v_f32m1(input_a, vl); // 从数组a中加载vl个元素到向量寄存器va中
      vfloat32m1_t vb = vle32_v_f32m1(input_b, vl); // 从数组b中加载vl个元素到向量寄存器vb中
      vfloat32m1_t vc = vfmin_vv_f32m1(va, vb, vl);   // 向量寄存器va和向量寄存器vb中vl个元素对应取min，结果为vc
      vse32_v_f32m1(output, vc, vl);   // 将向量寄存器中的vl个元素存到数组output中

      input_a += vl;
      input_b += vl;
      output += vl;
      batch -= vl;
  }
}

void xnn_f32_vmin_ukernel__scalar_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output//,
    // const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]
)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


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

    float vacc0 = math_min_f32(va0, vb0);
    float vacc1 = math_min_f32(va1, vb1);
    float vacc2 = math_min_f32(va2, vb2);
    float vacc3 = math_min_f32(va3, vb3);
    float vacc4 = math_min_f32(va4, vb4);
    float vacc5 = math_min_f32(va5, vb5);
    float vacc6 = math_min_f32(va6, vb6);
    float vacc7 = math_min_f32(va7, vb7);



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
      float vacc = math_min_f32(va, vb);
      *output++ = vacc;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}