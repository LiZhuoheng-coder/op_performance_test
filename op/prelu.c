#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "op/prelu.h"

void op_test_prelu_run(int batch_size)
{
    // 准备测试数据
    srand(time(NULL));
    int alignment = 64; // 对齐字节

    size_t channels = 16;
    size_t rows = batch_size;

    float* input = (float*)aligned_alloc(alignment, batch_size * channels * sizeof(float));
    float* output = (float*)aligned_alloc(alignment, batch_size * channels * sizeof(float));

    float* weights = (float*)aligned_alloc(alignment, channels * sizeof(float));

    // 确保内存分配成功
    if (input == NULL || output == NULL || weights == NULL)
    {
        perror("aligned_alloc failed");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < batch_size * channels; i++)
    {
        input[i] = rand() % 100;
    }

    for (size_t i = 0; i < channels; i++)
    {
        weights[i] = rand() % 100;
    }


    printf("OP: Prelu Test \n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_prelu_ukernel__scalar_2x4(rows, channels, input, channels, weights, output, channels);
    int64_t end = clock();

    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_prelu_ukernel__rvv_2x8(rows, channels, input, channels, weights, output, channels);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    printf("-------------------------------------\n");

    free(input);
    free(output);
    free(weights);

}

void xnn_f32_prelu_ukernel__rvv_2x8(
  size_t rows,
  size_t channels,
  const float* restrict input,
  size_t input_stride,
  const float* restrict weights,
  float* restrict output,
  size_t output_stride)
{
  assert(rows != 0); 
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) { // if rows < 2, process 1 row
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights; // pointer to first element of weights
    size_t c = channels; // initialize number of channels
    for(; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const size_t vl = vsetvl_e32m1(c); // set vector length
      const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights 
      w += 4;
      const vfloat32m1_t vw4567 = vle32_v_f32m1(w, vl); // load 4 weights
      w += 4;

      const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
      i0 += 4;
      const vfloat32m1_t vi0x4567 = vle32_v_f32m1(i0, vl); // load 4 input
      i0 += 4;
      const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
      i1 += 4;
      const vfloat32m1_t vi1x4567 = vle32_v_f32m1(i1, vl); // load 4 input
      i1 += 4;

      vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
      //neon: const uint32x4_t vm0x0123 = vcltq_s32(vreinterpretq_s32_f32(vi0x0123), vmovq_n_s32(0));
      const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
      vfloat32m1_t vacc0x4567 = vfmul_vv_f32m1(vi0x4567, vw4567, vl); // multiplication
      const vbool32_t vm0x4567 = vmflt_vf_f32m1_b32(vi0x4567, .0f, vl);
      vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
      const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);
      vfloat32m1_t vacc1x4567 = vfmul_vv_f32m1(vi1x4567, vw4567, vl); // multiplication
      const vbool32_t vm1x4567 = vmflt_vf_f32m1_b32(vi1x4567, .0f, vl);
      // neon:
      // vacc0x0123 = vbslq_f32(vm0x0123, vacc0x0123, vi0x0123);
      // vacc0x4567 = vbslq_f32(vm0x4567, vacc0x4567, vi0x4567);
      // vacc1x0123 = vbslq_f32(vm1x0123, vacc1x0123, vi1x0123);
      // vacc1x4567 = vbslq_f32(vm1x4567, vacc1x4567, vi1x4567);
      vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
      vacc0x4567 = vmerge_vvm_f32m1(vm0x4567, vacc0x4567, vi0x4567, vl);
      vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);
      vacc1x4567 = vmerge_vvm_f32m1(vm1x4567, vacc1x4567, vi1x4567, vl);

      vse32_v_f32m1(o0, vacc0x0123, vl); // store result
      o0 += 4;
      vse32_v_f32m1(o0, vacc0x4567, vl); // store result
      o0 += 4;
      vse32_v_f32m1(o1, vacc1x0123, vl); // store result
      o1 += 4;
      vse32_v_f32m1(o1, vacc1x4567, vl); // store result
      o1 += 4;
    }

    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) { // process 4 cols
      const size_t vl = vsetvl_e32m1(c);
      const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights
      w += 4;
      const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
      i0 += 4;
      const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
      i1 += 4;

      vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
      const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
      vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
      const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);

      vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
      vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);

      vse32_v_f32m1(o0, vacc0x0123, vl); // store result
      o0 += 4;
      vse32_v_f32m1(o1, vacc1x0123, vl); // store result
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) { // 
      const size_t vl = vsetvl_e32m1(c);
      const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights
      w += 4;
      const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
      i0 = (const float*) ((uintptr_t) i0 + c);
      const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
      i1 = (const float*) ((uintptr_t) i1 + c);

      vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
      const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
      vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
      const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);

      vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
      vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);

      vse32_v_f32m1(o0, vacc0x0123, vl); // store result
      o0 = (float*) ((uintptr_t) o0 + c); 
      vse32_v_f32m1(o1, vacc1x0123, vl); // store result
      o1 = (float*) ((uintptr_t) o1 + c);
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
  
}

void xnn_f32_prelu_ukernel__scalar_2x4(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vw0 = w[0];
      const float vw1 = w[1];
      const float vw2 = w[2];
      const float vw3 = w[3];

      const float vi0x0 = i0[0];
      const float vi0x1 = i0[1];
      const float vi0x2 = i0[2];
      const float vi0x3 = i0[3];
      i0 += 4;
      const float vi1x0 = i1[0];
      const float vi1x1 = i1[1];
      const float vi1x2 = i1[2];
      const float vi1x3 = i1[3];
      i1 += 4;

      const float vacc0x0 = XNN_UNPREDICTABLE(vi0x0 < 0.0f) ? vi0x0 * vw0 : vi0x0;
      const float vacc0x1 = XNN_UNPREDICTABLE(vi0x1 < 0.0f) ? vi0x1 * vw1 : vi0x1;
      const float vacc0x2 = XNN_UNPREDICTABLE(vi0x2 < 0.0f) ? vi0x2 * vw2 : vi0x2;
      const float vacc0x3 = XNN_UNPREDICTABLE(vi0x3 < 0.0f) ? vi0x3 * vw3 : vi0x3;
      const float vacc1x0 = XNN_UNPREDICTABLE(vi1x0 < 0.0f) ? vi1x0 * vw0 : vi1x0;
      const float vacc1x1 = XNN_UNPREDICTABLE(vi1x1 < 0.0f) ? vi1x1 * vw1 : vi1x1;
      const float vacc1x2 = XNN_UNPREDICTABLE(vi1x2 < 0.0f) ? vi1x2 * vw2 : vi1x2;
      const float vacc1x3 = XNN_UNPREDICTABLE(vi1x3 < 0.0f) ? vi1x3 * vw3 : vi1x3;

      o0[0] = vacc0x0;
      o0[1] = vacc0x1;
      o0[2] = vacc0x2;
      o0[3] = vacc0x3;
      o0 += 4;
      o1[0] = vacc1x0;
      o1[1] = vacc1x1;
      o1[2] = vacc1x2;
      o1[3] = vacc1x3;
      o1 += 4;

      w += 4;
    }
    for (; c != 0; c -= sizeof(float)) {
      const float vw = *w++;

      const float vi0 = *i0++;
      const float vi1 = *i1++;

      const float vacc0 = XNN_UNPREDICTABLE(vi0 < 0.0f) ? vi0 * vw : vi0;
      const float vacc1 = XNN_UNPREDICTABLE(vi1 < 0.0f) ? vi1 * vw : vi1;

      *o0++ = vacc0;
      *o1++ = vacc1;
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}