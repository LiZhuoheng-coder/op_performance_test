Seagull 算子性能测试
========

## build
- 修改 CMakeLists.txt 中 `set(CMAKE_CROSSCOMPILING TRUE)` 为 `set(CMAKE_CROSSCOMPILING False)`
```sh
mkdir build && cd build
cmake ..
ninja # or make -j4
```
## 运行

- 直接运行build目录中的 `./seagull_op_test`

## 性能统计

| 算子 | batch | 标量时间（ms）| RVV时间（ms）|
|----|----|----|----|
|Leaky Relu| 1000 |0.007|0.003|
|Leaky Relu|10000 |0.081|0.020|
|Leaky Relu|100000|0.619|0.244|
|Leaky Relu|1000000|6.002|2.689|
|GEMM 1X4| 100| 0.008| 0.005|
|GEMM 1X4|1000| 0.560| 0.235|
|GEMM 1X4|5000| 13.385| 5.317|
|GEMM 1X4|10000|40.238|21.514|
|PRelu | 1000| 0.015|0.028|
|PRelu |10000|0.408|0.253|
|PRelu |100000|3.612|2.509|
|PRelu |1000000|36.614|25.328|
|MUL |1000|0.008|0.005|
|MUL |10000|0.041|0.019|
|MUL |100000|0.859|0.502|
|MUL |1000000|8.204|4.892|
|DIV |1000|0.012|0.004|
|DIV |10000|0.070|0.021|
|DIV |100000|0.644|0.240|
|DIV |1000000|6.410|2.064|
