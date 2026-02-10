/*
 * GPU Stress Test - CUDA Backend
 *
 * Stress testing logic derived from gpu-burn by Ville Timonen
 * (https://github.com/wilicc/gpu-burn), BSD-licensed:
 *
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpu_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <chrono>

// Matrix size (same as gpu-burn)
static const int SIZE = 2048;

// Epsilon for floating point comparison (from gpu-burn)
#define EPSILON 0.001f
#define EPSILOND 0.0000001

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans) { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::string msg = std::string("CUDA Error: ") + cudaGetErrorString(code) +
                          " at " + file + ":" + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

#define CUBLAS_CHECK(ans) { cublas_assert((ans), __FILE__, __LINE__); }
inline void cublas_assert(cublasStatus_t code, const char* file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::string msg = std::string("cuBLAS Error: code ") + std::to_string((int)code) +
                          " at " + file + ":" + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// Comparison kernels (ported from gpu-burn compare.cu)
// ---------------------------------------------------------------------------
__global__ void compare_kernel_float(const float* C, const float* C_ref,
                                     int* faultyElems, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        if (fabsf(C[idx] - C_ref[idx]) > EPSILON) {
            atomicAdd(faultyElems, 1);
        }
    }
}

__global__ void compare_kernel_double(const double* C, const double* C_ref,
                                      int* faultyElems, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        if (fabs(C[idx] - C_ref[idx]) > EPSILOND) {
            atomicAdd(faultyElems, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA Backend Implementation
// ---------------------------------------------------------------------------
class CudaBackend : public GpuBackend {
private:
    struct DeviceResources {
        cublasHandle_t handle = nullptr;
        // Float buffers
        float* d_A_f = nullptr;
        float* d_B_f = nullptr;
        float* d_C_f = nullptr;
        float* d_C_ref_f = nullptr;
        // Double buffers
        double* d_A_d = nullptr;
        double* d_B_d = nullptr;
        double* d_C_d = nullptr;
        double* d_C_ref_d = nullptr;
        // Error tracking (allocated once)
        int* d_errors = nullptr;
        int matrix_size = SIZE;
        Precision precision = Precision::FLOAT;
        bool initialized = false;
    };

    std::vector<DeviceResources> resources;
    bool nvml_initialized = false;
    std::vector<nvmlDevice_t> nvml_devices;

    void init_nvml() {
        nvmlReturn_t result = nvmlInit_v2();
        if (result == NVML_SUCCESS) {
            nvml_initialized = true;
            unsigned int device_count = 0;
            nvmlDeviceGetCount_v2(&device_count);
            nvml_devices.resize(device_count);
            for (unsigned int i = 0; i < device_count; ++i) {
                nvmlDeviceGetHandleByIndex_v2(i, &nvml_devices[i]);
            }
        } else {
            std::cerr << "Warning: NVML init failed, temperature monitoring unavailable" << std::endl;
        }
    }

public:
    CudaBackend() {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        resources.resize(count);
        init_nvml();
    }

    ~CudaBackend() {
        for (size_t i = 0; i < resources.size(); ++i) {
            if (resources[i].initialized) {
                cleanup(static_cast<int>(i));
            }
        }
        if (nvml_initialized) {
            nvmlShutdown();
        }
    }

    std::vector<DeviceInfo> list_devices() override {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        std::vector<DeviceInfo> devices;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            devices.push_back({i, prop.name, total_mem, free_mem});
        }
        return devices;
    }

    bool initialize_device(int device_id, const StressConfig& config) override {
        CUDA_CHECK(cudaSetDevice(device_id));
        auto& res = resources[device_id];
        res.precision = config.precision;
        res.matrix_size = SIZE;

        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&res.handle));

        // Enable tensor cores if requested
        if (config.use_tensor_cores) {
            CUBLAS_CHECK(cublasSetMathMode(res.handle, CUBLAS_TENSOR_OP_MATH));
        }

        int n = res.matrix_size;

        // Determine how much memory to use
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        size_t use_bytes;
        if (config.memory_bytes > 0) {
            use_bytes = static_cast<size_t>(config.memory_bytes);
        } else {
            use_bytes = static_cast<size_t>(free_mem * config.memory_fraction);
        }

        if (config.precision == Precision::DOUBLE) {
            size_t elem_size = sizeof(double);
            size_t matrix_bytes = n * n * elem_size;

            if (use_bytes < 3 * matrix_bytes) {
                throw std::runtime_error("Not enough GPU memory for stress test");
            }

            // Generate random matrices on host (like gpu-burn)
            std::vector<double> h_A(n * n), h_B(n * n);
            srand(10);
            for (int i = 0; i < n * n; ++i) {
                h_A[i] = (double)(rand() % 1000000) / 100000.0;
                h_B[i] = (double)(rand() % 1000000) / 100000.0;
            }

            CUDA_CHECK(cudaMalloc(&res.d_A_d, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_B_d, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_C_d, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_C_ref_d, matrix_bytes));

            CUDA_CHECK(cudaMemcpy(res.d_A_d, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(res.d_B_d, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));

            std::cout << "  GPU " << device_id << ": " << (use_bytes / 1024 / 1024) << " MB allocated"
                      << " (" << (free_mem / 1024 / 1024) << " MB free), using DOUBLES"
                      << (config.use_tensor_cores ? ", Tensor Cores" : "") << std::endl;
        } else {
            size_t elem_size = sizeof(float);
            size_t matrix_bytes = n * n * elem_size;

            if (use_bytes < 3 * matrix_bytes) {
                throw std::runtime_error("Not enough GPU memory for stress test");
            }

            // Generate random matrices on host (like gpu-burn)
            std::vector<float> h_A(n * n), h_B(n * n);
            srand(10);
            for (int i = 0; i < n * n; ++i) {
                h_A[i] = (float)((double)(rand() % 1000000) / 100000.0);
                h_B[i] = (float)((double)(rand() % 1000000) / 100000.0);
            }

            CUDA_CHECK(cudaMalloc(&res.d_A_f, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_B_f, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_C_f, matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_C_ref_f, matrix_bytes));

            CUDA_CHECK(cudaMemcpy(res.d_A_f, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(res.d_B_f, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));

            std::cout << "  GPU " << device_id << ": " << (use_bytes / 1024 / 1024) << " MB allocated"
                      << " (" << (free_mem / 1024 / 1024) << " MB free), using FLOATS"
                      << (config.use_tensor_cores ? ", Tensor Cores" : "") << std::endl;
        }

        // Allocate error counter once
        CUDA_CHECK(cudaMalloc(&res.d_errors, sizeof(int)));

        // Compute reference result
        compute_reference(device_id);

        res.initialized = true;
        return true;
    }

    StressResult run_stress_iteration(int device_id) override {
        CUDA_CHECK(cudaSetDevice(device_id));
        auto& res = resources[device_id];
        int n = res.matrix_size;

        auto start = std::chrono::steady_clock::now();

        int host_errors = 0;

        // Reset error counter
        CUDA_CHECK(cudaMemset(res.d_errors, 0, sizeof(int)));

        if (res.precision == Precision::DOUBLE) {
            double alpha = 1.0, beta = 0.0;
            CUBLAS_CHECK(cublasDgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n, &alpha,
                                     res.d_A_d, n, res.d_B_d, n,
                                     &beta, res.d_C_d, n));
            
            // Synchronize before comparing
            CUDA_CHECK(cudaDeviceSynchronize());

            int threads = 256;
            int blocks = (n * n + threads - 1) / threads;
            compare_kernel_double<<<blocks, threads>>>(res.d_C_d, res.d_C_ref_d, res.d_errors, n);
        } else {
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n, &alpha,
                                     res.d_A_f, n, res.d_B_f, n,
                                     &beta, res.d_C_f, n));
            
            // Synchronize before comparing
            CUDA_CHECK(cudaDeviceSynchronize());

            int threads = 256;
            int blocks = (n * n + threads - 1) / threads;
            compare_kernel_float<<<blocks, threads>>>(res.d_C_f, res.d_C_ref_f, res.d_errors, n);
        }

        // Synchronize and read error count
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&host_errors, res.d_errors, sizeof(int), cudaMemcpyDeviceToHost));

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // 1 SGEMM of NxN = 2*N^3 FLOPS
        long long ops = 2LL * n * n * n;

        return {host_errors, ops, elapsed};
    }

    int get_temperature(int device_id) override {
        if (!nvml_initialized || device_id >= static_cast<int>(nvml_devices.size())) {
            return -1;
        }
        unsigned int temp = 0;
        nvmlReturn_t result = nvmlDeviceGetTemperature(nvml_devices[device_id],
                                                        NVML_TEMPERATURE_GPU, &temp);
        if (result != NVML_SUCCESS) {
            return -1;
        }
        return static_cast<int>(temp);
    }

    void cleanup(int device_id) override {
        CUDA_CHECK(cudaSetDevice(device_id));
        auto& res = resources[device_id];
        if (res.d_A_f) { cudaFree(res.d_A_f); res.d_A_f = nullptr; }
        if (res.d_B_f) { cudaFree(res.d_B_f); res.d_B_f = nullptr; }
        if (res.d_C_f) { cudaFree(res.d_C_f); res.d_C_f = nullptr; }
        if (res.d_C_ref_f) { cudaFree(res.d_C_ref_f); res.d_C_ref_f = nullptr; }
        if (res.d_A_d) { cudaFree(res.d_A_d); res.d_A_d = nullptr; }
        if (res.d_B_d) { cudaFree(res.d_B_d); res.d_B_d = nullptr; }
        if (res.d_C_d) { cudaFree(res.d_C_d); res.d_C_d = nullptr; }
        if (res.d_C_ref_d) { cudaFree(res.d_C_ref_d); res.d_C_ref_d = nullptr; }
        if (res.d_errors) { cudaFree(res.d_errors); res.d_errors = nullptr; }
        if (res.handle) { cublasDestroy(res.handle); res.handle = nullptr; }
        res.initialized = false;
    }

private:
    void compute_reference(int device_id) {
        auto& res = resources[device_id];
        int n = res.matrix_size;

        if (res.precision == Precision::DOUBLE) {
            double alpha = 1.0, beta = 0.0;
            CUBLAS_CHECK(cublasDgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n, &alpha,
                                     res.d_A_d, n, res.d_B_d, n,
                                     &beta, res.d_C_ref_d, n));
        } else {
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n, &alpha,
                                     res.d_A_f, n, res.d_B_f, n,
                                     &beta, res.d_C_ref_f, n));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

std::unique_ptr<GpuBackend> create_cuda_backend() {
    return std::make_unique<CudaBackend>();
}
