cmake_minimum_required(VERSION 3.18)
project(gpu-stress-test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(gpu-stress-test
    src/main.cpp
    src/stress_test.cpp
    src/cuda_backend.cu
)

target_include_directories(gpu-stress-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu-stress-test PRIVATE CUDA::cublas CUDA::cudart)

# NVML for temperature monitoring
find_library(NVML_LIBRARY nvidia-ml
    PATHS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_LIBRARY_DIR}/stubs
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64/stubs
        /usr/local/cuda/targets/x86_64-linux/lib/stubs
)
if(NVML_LIBRARY)
    target_link_libraries(gpu-stress-test PRIVATE ${NVML_LIBRARY})
    message(STATUS "Found NVML: ${NVML_LIBRARY}")
else()
    message(WARNING "NVML not found — temperature monitoring will be disabled")
    target_compile_definitions(gpu-stress-test PRIVATE NO_NVML)
endif()

# Compiler flags
if(MSVC)
    target_compile_options(gpu-stress-test PRIVATE /W4)
else()
    target_compile_options(gpu-stress-test PRIVATE -Wall -Wextra -O3)
endif()
