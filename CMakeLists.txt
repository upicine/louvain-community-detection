cmake_minimum_required(VERSION 3.16)
project(louvian CUDA)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

add_executable(louvian main.cu utils.cu utils.cuh modularity.cu modularity.cuh aggregation.cu aggregation.cuh)

set_target_properties(
        louvian
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)