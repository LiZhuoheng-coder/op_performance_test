cmake_minimum_required(VERSION 3.12)


set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_CROSSCOMPILING TRUE)
if (NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_COMPILER g++-10)
    set(CMAKE_C_COMPILER  gcc-10)
else()
    set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)
    set(CMAKE_C_COMPILER  riscv64-unknown-linux-gnu-gcc)
endif()

set(CMAKE_C_FLAGS_INIT "-march=rv64imafdcv0p7_zfh_xtheadcv1p0 -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64imafdcv0p7_zfh_xtheadcv1p0 -mabi=lp64d")


project(seagull_op_test)

aux_source_directory(. DIR_SRCS)
aux_source_directory(./op DIR_SRCS)

include_directories(include)

add_executable(seagull_op_test ${DIR_SRCS})
