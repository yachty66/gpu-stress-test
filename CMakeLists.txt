cmake_minimum_required(VERSION 3.18)
project(gpu-stress-test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(gpu-burn-pro
    src/main.cpp
    src/stress_test.cpp
    src/cuda_backend.cu
)

target_include_directories(gpu-burn-pro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu-burn-pro PRIVATE CUDA::cublas CUDA::cudart)

# Compiler flags
if(MSVC)
    target_compile_options(gpu-burn-pro PRIVATE /W4)
else()
    target_compile_options(gpu-burn-pro PRIVATE -Wall -Wextra -O3)
endif()
