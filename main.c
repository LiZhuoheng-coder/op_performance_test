#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "seagull_op.h"


int main(int argc, char *argv[]) {
    printf("Seagull Single NNOP Performance Test\n");
    printf("=====================================\n");
    // 测试lrelu
    printf("Batch Size: 1000\n");
    op_test_lrelu_run(1000);
    printf("Batch Size: 10000\n");
    op_test_lrelu_run(10000);
    printf("Batch Size: 100000\n");
    op_test_lrelu_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_lrelu_run(1000000);
    printf("=====================================\n");
    // 测试gemm
    printf("Batch Size: 100\n");
    op_test_gemm_run(100);
    printf("Batch Size: 1000\n");
    op_test_gemm_run(1000);
    // printf("Batch Size: 5000\n");
    // op_test_gemm_run(5000);
    // printf("Batch Size: 10000\n");
    // op_test_gemm_run(10000);
    printf("=====================================\n");
    // 测试gemm rule 1x4
    printf("Batch Size: 100\n");
    op_test_gemm_rule_1x4_run(100);
    printf("Batch Size: 1000\n");
    op_test_gemm_rule_1x4_run(1000);
    // printf("Batch Size: 5000\n");
    // op_test_gemm_rule_1x4_run(5000);
    // printf("Batch Size: 10000\n");
    // op_test_gemm_rule_1x4_run(10000);
    printf("=====================================\n");
    // 测试gemm rule 2x4
    printf("Batch Size: 100\n");
    op_test_gemm_rule_2x4_run(100);
    printf("Batch Size: 1000\n");
    op_test_gemm_rule_2x4_run(1000);
    // printf("Batch Size: 5000\n");
    // op_test_gemm_rule_2x4_run(5000);
    // printf("Batch Size: 10000\n");
    // op_test_gemm_rule_2x4_run(10000);
    printf("=====================================\n");
    // 测试prelu
    printf("Batch Size: 1000\n");
    op_test_prelu_run(1000);
    printf("Batch Size: 10000\n");
    op_test_prelu_run(10000);
    printf("Batch Size: 100000\n");
    op_test_prelu_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_prelu_run(1000000);
    printf("=====================================\n");
    // 测试sigmoid
    // printf("Batch Size: 1000\n");
    // op_test_sigmoid_run(1000);
    // printf("Batch Size: 10000\n");
    // op_test_sigmoid_run(10000);
    // printf("Batch Size: 100000\n");
    // op_test_sigmoid_run(100000);
    // printf("Batch Size: 1000000\n");
    // op_test_sigmoid_run(1000000);
    // printf("=====================================\n");
    // 测试max()
    printf("Batch Size: 1000\n");
    op_test_max_run(1000);
    printf("Batch Size: 10000\n");
    op_test_max_run(10000);
    printf("Batch Size: 100000\n");
    op_test_max_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_max_run(1000000);
    printf("=====================================\n");
    // 测试min()
    printf("Batch Size: 1000\n");
    op_test_min_run(1000);
    printf("Batch Size: 10000\n");
    op_test_min_run(10000);
    printf("Batch Size: 100000\n");
    op_test_min_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_min_run(1000000);
    printf("=====================================\n");
    // 测试add
    printf("Batch Size: 1000\n");
    op_test_add_run(1000);
    printf("Batch Size: 10000\n");
    op_test_add_run(10000);
    printf("Batch Size: 100000\n");
    op_test_add_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_add_run(1000000);
    printf("=====================================\n");
    // 测试sub
    printf("Batch Size: 1000\n");
    op_test_sub_run(1000);
    printf("Batch Size: 10000\n");
    op_test_sub_run(10000);
    printf("Batch Size: 100000\n");
    op_test_sub_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_sub_run(1000000);
    printf("=====================================\n");
    // 测试mul
    printf("Batch Size: 1000\n");
    op_test_mul_run(1000);
    printf("Batch Size: 10000\n");
    op_test_mul_run(10000);
    printf("Batch Size: 100000\n");
    op_test_mul_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_mul_run(1000000);
    printf("=====================================\n");
    // 测试div
    printf("Batch Size: 1000\n");
    op_test_div_run(1000);
    printf("Batch Size: 10000\n");
    op_test_div_run(10000);
    printf("Batch Size: 100000\n");
    op_test_div_run(100000);
    printf("Batch Size: 1000000\n");
    op_test_div_run(1000000);
    printf("=====================================\n");




    


    return 0;
}
