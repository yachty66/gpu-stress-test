cmake_minimum_required(VERSION 3.18)
project(ZuseStressTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Packages
find_package(CUDAToolkit REQUIRED)
# find_package(Vulkan REQUIRED) # Uncomment later for Vulkan fallback

# Include Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source Files
set(SOURCES
    src/main.cpp
    src/mon/nvml_monitor.cpp
    src/core/cuda_compute.cpp
)

# Executable
add_executable(zuse_stress ${SOURCES})

# Links
target_link_libraries(zuse_stress PRIVATE CUDA::cudart)
# target_link_libraries(zuse_stress PRIVATE Vulkan::Vulkan)

# Add NVML (usually comes with driver/CUDA)
find_library(NVML_LIB NAMES nvidia-ml PATHS "/usr/local/cuda/lib64/stubs" "/usr/lib/x86_64-linux-gnu" "/usr/lib64")
if(NVML_LIB)
    target_link_libraries(zuse_stress PRIVATE ${NVML_LIB})
    target_compile_definitions(zuse_stress PRIVATE HAS_NVML)
endif()
