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
|Leaky Relu| 1000 |0.014000|0.100000|
|Leaky Relu|10000 |0.149000|0.985000|
|Leaky Relu|100000|1.484000|118.465|
|Leaky Relu|1000000|14.276| 118.465|