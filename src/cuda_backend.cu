/*
 * GPU Stress Test - CUDA Backend
 *
 * Portions of the stress testing logic are derived from work by Ville Timonen,
 * BSD-licensed:
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

// Matrix size
static const int SIZE = 2048;

// Epsilon for floating point comparison
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
// Comparison kernels
// Compare result slot against slot 0 (the reference)
// ---------------------------------------------------------------------------
__global__ void compare_kernel_float(const float* C, int* faultyElems,
                                     size_t slot_offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        if (fabsf(C[idx] - C[slot_offset + idx]) > EPSILON) {
            atomicAdd(faultyElems, 1);
        }
    }
}

__global__ void compare_kernel_double(const double* C, int* faultyElems,
                                      size_t slot_offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        if (fabs(C[idx] - C[slot_offset + idx]) > EPSILOND) {
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
        // Input matrices (one copy each)
        void* d_A = nullptr;
        void* d_B = nullptr;
        // Result buffer: holds num_slots matrices contiguously
        // Slot 0 = reference, slots 1..num_slots-1 = results to compare
        void* d_C = nullptr;
        // Error tracking (allocated once)
        int* d_errors = nullptr;
        int matrix_size = SIZE;
        size_t num_slots = 0;       // Number of result matrix slots
        size_t matrix_elements = 0; // n * n
        size_t matrix_bytes = 0;    // n * n * sizeof(T)
        size_t total_allocated = 0; // Total bytes allocated on GPU
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
        res.matrix_elements = (size_t)n * n;

        // Determine element size based on precision
        size_t elem_size = (config.precision == Precision::DOUBLE) ? sizeof(double) : sizeof(float);
        res.matrix_bytes = res.matrix_elements * elem_size;

        // Determine how much memory to use
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        size_t use_bytes;
        if (config.memory_bytes > 0) {
            use_bytes = static_cast<size_t>(config.memory_bytes);
        } else {
            use_bytes = static_cast<size_t>(free_mem * config.memory_fraction);
        }

        // Calculate number of result slots that fit in memory
        // Memory layout: A (1 matrix) + B (1 matrix) + C (num_slots matrices)
        // So: num_slots = (use_bytes - 2 * matrix_bytes) / matrix_bytes
        if (use_bytes < 3 * res.matrix_bytes) {
            throw std::runtime_error("Not enough GPU memory for stress test");
        }
        res.num_slots = (use_bytes - 2 * res.matrix_bytes) / res.matrix_bytes;
        if (res.num_slots < 2) {
            throw std::runtime_error("Not enough GPU memory for at least 2 result slots");
        }

        // Generate random matrices on host
        srand(10);
        if (config.precision == Precision::DOUBLE) {
            std::vector<double> h_A(res.matrix_elements), h_B(res.matrix_elements);
            for (size_t i = 0; i < res.matrix_elements; ++i) {
                h_A[i] = (double)(rand() % 1000000) / 100000.0;
                h_B[i] = (double)(rand() % 1000000) / 100000.0;
            }
            CUDA_CHECK(cudaMalloc(&res.d_A, res.matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_B, res.matrix_bytes));
            CUDA_CHECK(cudaMemcpy(res.d_A, h_A.data(), res.matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(res.d_B, h_B.data(), res.matrix_bytes, cudaMemcpyHostToDevice));
        } else {
            std::vector<float> h_A(res.matrix_elements), h_B(res.matrix_elements);
            for (size_t i = 0; i < res.matrix_elements; ++i) {
                h_A[i] = (float)((double)(rand() % 1000000) / 100000.0);
                h_B[i] = (float)((double)(rand() % 1000000) / 100000.0);
            }
            CUDA_CHECK(cudaMalloc(&res.d_A, res.matrix_bytes));
            CUDA_CHECK(cudaMalloc(&res.d_B, res.matrix_bytes));
            CUDA_CHECK(cudaMemcpy(res.d_A, h_A.data(), res.matrix_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(res.d_B, h_B.data(), res.matrix_bytes, cudaMemcpyHostToDevice));
        }

        // Allocate the big result buffer: num_slots contiguous matrices
        size_t c_total_bytes = res.num_slots * res.matrix_bytes;
        CUDA_CHECK(cudaMalloc(&res.d_C, c_total_bytes));

        res.total_allocated = 2 * res.matrix_bytes + c_total_bytes;

        // Allocate error counter
        CUDA_CHECK(cudaMalloc(&res.d_errors, sizeof(int)));

        std::cout << "  GPU " << device_id << ": "
                  << (res.total_allocated / 1024 / 1024) << " MB allocated"
                  << " (" << (free_mem / 1024 / 1024) << " MB free)"
                  << ", " << res.num_slots << " result slots"
                  << ", using " << (config.precision == Precision::DOUBLE ? "DOUBLES" : "FLOATS")
                  << (config.use_tensor_cores ? ", Tensor Cores" : "") << std::endl;

        res.initialized = true;
        return true;
    }

    StressResult run_stress_iteration(int device_id) override {
        CUDA_CHECK(cudaSetDevice(device_id));
        auto& res = resources[device_id];
        int n = res.matrix_size;

        auto start = std::chrono::steady_clock::now();

        // Compute A*B into every result slot
        if (res.precision == Precision::DOUBLE) {
            double alpha = 1.0, beta = 0.0;
            double* C_base = static_cast<double*>(res.d_C);
            for (size_t s = 0; s < res.num_slots; ++s) {
                CUBLAS_CHECK(cublasDgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, n, n, &alpha,
                                         static_cast<double*>(res.d_A), n,
                                         static_cast<double*>(res.d_B), n,
                                         &beta,
                                         C_base + s * res.matrix_elements, n));
            }
        } else {
            float alpha = 1.0f, beta = 0.0f;
            float* C_base = static_cast<float*>(res.d_C);
            for (size_t s = 0; s < res.num_slots; ++s) {
                CUBLAS_CHECK(cublasSgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         n, n, n, &alpha,
                                         static_cast<float*>(res.d_A), n,
                                         static_cast<float*>(res.d_B), n,
                                         &beta,
                                         C_base + s * res.matrix_elements, n));
            }
        }

        // Synchronize before comparing
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compare all slots against slot 0 (the reference)
        CUDA_CHECK(cudaMemset(res.d_errors, 0, sizeof(int)));

        int threads = 256;
        int blocks = (n * n + threads - 1) / threads;

        if (res.precision == Precision::DOUBLE) {
            double* C_base = static_cast<double*>(res.d_C);
            for (size_t s = 1; s < res.num_slots; ++s) {
                compare_kernel_double<<<blocks, threads>>>(
                    C_base, res.d_errors, s * res.matrix_elements, n);
            }
        } else {
            float* C_base = static_cast<float*>(res.d_C);
            for (size_t s = 1; s < res.num_slots; ++s) {
                compare_kernel_float<<<blocks, threads>>>(
                    C_base, res.d_errors, s * res.matrix_elements, n);
            }
        }

        // Synchronize and read error count
        CUDA_CHECK(cudaDeviceSynchronize());
        int host_errors = 0;
        CUDA_CHECK(cudaMemcpy(&host_errors, res.d_errors, sizeof(int), cudaMemcpyDeviceToHost));

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // Total FLOPS: num_slots SGEMMs, each is 2*N^3
        long long ops = (long long)res.num_slots * 2LL * n * n * n;

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
        if (res.d_A) { cudaFree(res.d_A); res.d_A = nullptr; }
        if (res.d_B) { cudaFree(res.d_B); res.d_B = nullptr; }
        if (res.d_C) { cudaFree(res.d_C); res.d_C = nullptr; }
        if (res.d_errors) { cudaFree(res.d_errors); res.d_errors = nullptr; }
        if (res.handle) { cublasDestroy(res.handle); res.handle = nullptr; }
        res.initialized = false;
    }
};

std::unique_ptr<GpuBackend> create_cuda_backend() {
    return std::make_unique<CudaBackend>();
}
