#ifndef __SEAGULL_OP_H
#define __SEAGULL_OP_H

#include "xnnpack/common.h"
#include "xnnpack/math.h"

#include "op/gemm.h"
#include "op/lrelu.h"
#include "op/prelu.h"
#include "op/sigmoid.h"
#include "op/max.h"
#include "op/min.h"
#include "op/add.h"
#include "op/sub.h"
#include "op/mul.h"
#include "op/div.h"


typedef enum {
    SEAGULL_OP_LRELU = 0,
    SEAGULL_OP_GEMM,
    SEAGULL_OP_PRELU,
    SEAGULL_OP_SIGMOID,
    SEAGULL_OP_MAX,
    SEAGULL_OP_MIN,
    SEAGULL_OP_ADD,
    SEAGULL_OP_SUB,
    SEAGULL_OP_MUL,
    OP_MAX,
} seagull_op_t;



#endif
