#ifndef __SEAGULL_OP_H
#define __SEAGULL_OP_H

#include "xnnpack/common.h"

#define XNN_MIN_ELEMENTS(count) static count

typedef enum {
    SEAGULL_OP_LRELU = 0,
    SEAGULL_OP_GEMM,
    SEAGULL_OP_MAX
} seagull_op_t;



#endif