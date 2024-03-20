#include <assert.h>
#include <riscv_vector.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "op/sigmoid.h"

XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64] = {
  0x3F800000, 0x3F7F64D2, 0x3F7ECD87, 0x3F7E3A29, 0x3F7DAAC3, 0x3F7D1F62, 0x3F7C980F, 0x3F7C14D5,
  0x3F7B95C2, 0x3F7B1ADF, 0x3F7AA43A, 0x3F7A31DC, 0x3F79C3D3, 0x3F795A2B, 0x3F78F4F0, 0x3F78942D,
  0x3F7837F0, 0x3F77E046, 0x3F778D3A, 0x3F773EDA, 0x3F76F532, 0x3F76B051, 0x3F767043, 0x3F763516,
  0x3F75FED7, 0x3F75CD94, 0x3F75A15B, 0x3F757A3A, 0x3F75583F, 0x3F753B79, 0x3F7523F6, 0x3F7511C4,
  0x3F7504F3, 0x3F74FD92, 0x3F74FBAF, 0x3F74FF5B, 0x3F7508A4, 0x3F75179A, 0x3F752C4D, 0x3F7546CD,
  0x3F75672A, 0x3F758D75, 0x3F75B9BE, 0x3F75EC15, 0x3F76248C, 0x3F766334, 0x3F76A81E, 0x3F76F35B,
  0x3F7744FD, 0x3F779D16, 0x3F77FBB8, 0x3F7860F5, 0x3F78CCDF, 0x3F793F89, 0x3F79B907, 0x3F7A396A,
  0x3F7AC0C7, 0x3F7B4F30, 0x3F7BE4BA, 0x3F7C8177, 0x3F7D257D, 0x3F7DD0DF, 0x3F7E83B3, 0x3F7F3E0C,
};


void op_test_sigmoid_run(int batch_size)
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

    union xnn_f32_sigmoid_params params;
    // 初始化参数
    params.scalar_rr2_lut64_p2.magic_bias = 0x1.800000p+23;
    params.scalar_rr2_lut64_p2.minus_log2e = -0x1.715476p+0;
    params.scalar_rr2_lut64_p2.ln2_hi = 0x1.62e430p-1;
    params.scalar_rr2_lut64_p2.ln2_lo = 0x1.abc9e3p-25;
    params.scalar_rr2_lut64_p2.c2 = 0x1.62e430p-1;
    params.scalar_rr2_lut64_p2.one = 0x1.000000p+0;
    params.scalar_rr2_lut64_p2.denorm_cutoff = 0x1.5d589ep-9;

    printf("OP: Sigmoid Test \n");
    printf("=====================================\n");
    // 测试scalar
    printf("Test scalar\n");
    int64_t start = clock();
    xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2(batch_size * sizeof(float), input, output, &params);
    int64_t end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);
    // 测试rvv
    printf("Test rvv\n");
    start = clock();
    xnn_f32_vsigmoid_ukernel__rvv(batch_size * sizeof(float), input, output, &params);
    end = clock();
    printf("Time: %f ms\n", (end - start) * 1.0 / CLOCKS_PER_SEC * 1000);

    free(input);
    free(output);

}

void xnn_f32_vsigmoid_ukernel__rvv(
  size_t batch,
  const float* input,
  float* output,
  const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = params->scalar_rr2_lut64_p2.magic_bias;
  const float vminus_log2e = params->scalar_rr2_lut64_p2.minus_log2e;
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vln2_hi = params->scalar_rr2_lut64_p2.ln2_hi;
  const float vln2_lo = params->scalar_rr2_lut64_p2.ln2_lo;
  const float vc2 = params->scalar_rr2_lut64_p2.c2;
  const float vone = params->scalar_rr2_lut64_p2.one;
  const float vdenorm_cutoff = params->scalar_rr2_lut64_p2.denorm_cutoff;

  size_t vl;

//   for (size_t i = 0; i < batch; i += vl) {
//     vl = vsetvl_e32m8(batch - i);

//     vfloat32m8_t vx = vle32_v_f32m8(input + i, vl);
//     // get abs
//     vfloat32m8_t vz = vfabs_v_f32m8(vx, vl);

//     // vz*(-log2(e))+magic_bias
//     vfloat32m8_t vn = vfadd_vf_f32m8(vfmul_vf_f32m8(vz, vminus_log2e, vl), vmagic_bias, vl);

//     // get exponent
//     vuint32m8_t ve = vsll_vx_u32m8(vfcvt_xu_f_v_u32m8(vn, vl), 17, vl);

//     // find index in lookup table using mask
//     vuint32m8_t vidx = vand_vx_u32m8(vfcvt_xu_f_v_u32m8(vn, vl), vindex_mask, vl);
//     vfloat32m8_t vs =  vfcvt_xu_f_v_u32m8(vfadd_vv_f32m8(vrgather_vv_i32m1(xnn_table_exp2minus_k_over_64, vidx, vl), ve, vl), vl);

//     // remove magic bias
//     vn = vsub_vf_f32m8(vn, vmagic_bias, vl);

//     // find logarithm
//     vfloat32m8_t vt = vadd_vv_f32m8(vfmul_vf_f32m8(vn, vln2_hi, vl), vz, vl);
//     vt = vadd_vv_f32m8(vfmul_vf_f32m8(vn, vln2_lo, vl), vt, vl);

//     // calculate the quadratic term logarithmically.
//     vfloat32m8_t vp = vfmul_vf_f32m8(vt, vc2, vl);
//     vp = vfsub_vv_f32m8(vt, vfmul_vv_f32m8(vp, vt, vl), vl);

//     // caculate sigmoid polynomial approximation
//     vfloat32m8_t vy = vfsub_vv_f32m8(vs, vfmul_vv_f32m8(vs, vp, vl), vl);
//     vfloat32m8_t vd = vfadd_vf_f32m8(vy, vone, vl);
//     vfloat32m8_t vf = vfdiv_vv_f32m8(vy, vd, vl);

//     // store result
//     vse32_v_f32m8(output + i, vf, vl);
//   }
}

void xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmagic_bias = params->scalar_rr2_lut64_p2.magic_bias;
  const float vminus_log2e = params->scalar_rr2_lut64_p2.minus_log2e;
  const uint32_t vindex_mask = UINT32_C(0x3F);
  const float vln2_hi = params->scalar_rr2_lut64_p2.ln2_hi;
  const float vln2_lo = params->scalar_rr2_lut64_p2.ln2_lo;
  const float vc2 = params->scalar_rr2_lut64_p2.c2;
  const float vone = params->scalar_rr2_lut64_p2.one;
  const float vdenorm_cutoff = params->scalar_rr2_lut64_p2.denorm_cutoff;

  // process two point each time
  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    // get abs
    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);

    // vz*(-log2(e))+magic_bias
    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;

    // get exponent
    const uint32_t ve0 = float_as_uint32(vn0) << 17;
    const uint32_t ve1 = float_as_uint32(vn1) << 17;

    // find index in lookup table using mask
    const uint32_t vidx0 = float_as_uint32(vn0) & vindex_mask;
    const float vs0 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx0] + ve0);
    const uint32_t vidx1 = float_as_uint32(vn1) & vindex_mask;
    const float vs1 = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx1] + ve1);

    // remove magic bias
    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    // find logarithm
    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;

    // calculate the quadratic term logarithmically.
    float vp0 = vt0 * vc2;
    float vp1 = vt1 * vc2;

    vp0 = vt0 - vp0 * vt0;
    vp1 = vt1 - vp1 * vt1;

    // caculate sigmoid polynomial approximation
    const float vy0 = vs0 - vs0 * vp0;
    const float vy1 = vs1 - vs1 * vp1;

    const float vd0 = vy0 + vone;
    const float vd1 = vy1 + vone;

    float vf0 = vy0 / vd0;
    float vf1 = vy1 / vd1;

    // handling special values
    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }

    // store result
    output[0] = vf0;
    output[1] = vf1;
    output += 2;
  }
  // handling rest data
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;

    const float vz = fabsf(vx);

    float vn = vz * vminus_log2e + vmagic_bias;
    const uint32_t ve = float_as_uint32(vn) << 17;
    const uint32_t vidx = float_as_uint32(vn) & vindex_mask;
    const float vs = uint32_as_float(xnn_table_exp2minus_k_over_64[vidx] + ve);
    vn -= vmagic_bias;

    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    float vp = vt * vc2;
    vp = vt - vp * vt;

    const float vy = vs - vs * vp;
    const float vd = vy + vone;

    float vf = vy / vd;
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *output = vf;
  }
}