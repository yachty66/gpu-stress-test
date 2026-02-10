#include "gpu_backend.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void compare_matrices(const float* A, const float* B, int* errors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        // Use atomicAdd to count mismatches (bit-level differences)
        if (A[idx] != B[idx]) {
            atomicAdd(errors, 1);
        }
    }
}

class CudaBackend : public GpuBackend {
private:
    struct DeviceResources {
        cublasHandle_t handle;
        float *d_A, *d_B, *d_C, *d_C_ref;
        int matrix_size;
    };
    std::vector<DeviceResources> resources;

public:
    CudaBackend() {
        int count = 0;
        cudaGetDeviceCount(&count);
        resources.resize(count, {nullptr, nullptr, nullptr, nullptr, nullptr, 0});
    }

    std::vector<DeviceInfo> list_devices() override {
        int count = 0;
        cudaGetDeviceCount(&count);
        std::vector<DeviceInfo> devices;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            size_t free, total;
            cudaSetDevice(i);
            cudaMemGetInfo(&free, &total);
            devices.push_back({i, prop.name, total, free});
        }
        return devices;
    }

    bool initialize_device(int device_id) override {
        CUDA_CHECK(cudaSetDevice(device_id));
        cublasCreate(&resources[device_id].handle);

        // Define matrix size (e.g., 2048x2048 like gpu-burn)
        int n = 2048;
        resources[device_id].matrix_size = n;
        size_t size = n * n * sizeof(float);

        CUDA_CHECK(cudaMalloc(&resources[device_id].d_A, size));
        CUDA_CHECK(cudaMalloc(&resources[device_id].d_B, size));
        CUDA_CHECK(cudaMalloc(&resources[device_id].d_C, size));
        CUDA_CHECK(cudaMalloc(&resources[device_id].d_C_ref, size));

        // Initialize with random data (simplified for now)
        CUDA_CHECK(cudaMemset(resources[device_id].d_A, 1, size));
        CUDA_CHECK(cudaMemset(resources[device_id].d_B, 1, size));
        
        return true;
    }

    int run_stress_iteration(int device_id, int iterations) override {
        cudaSetDevice(device_id);
        auto& res = resources[device_id];
        int n = res.matrix_size;
        float alpha = 1.0f;
        float beta = 0.0f;
        
        int host_errors = 0;
        int* d_errors;
        cudaMalloc(&d_errors, sizeof(int));

        // Perform first run to get a reference
        cublasSgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, res.d_A, n, res.d_B, n, &beta, res.d_C_ref, n);

        for (int i = 0; i < iterations; ++i) {
            cudaMemset(d_errors, 0, sizeof(int));
            cublasSgemm(res.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, res.d_A, n, res.d_B, n, &beta, res.d_C, n);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (n * n + threadsPerBlock - 1) / threadsPerBlock;
            compare_matrices<<<blocksPerGrid, threadsPerBlock>>>(res.d_C, res.d_C_ref, d_errors, n);
            
            int iter_errors = 0;
            cudaMemcpy(&iter_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
            host_errors += iter_errors;
        }
        
        cudaFree(d_errors);
        cudaDeviceSynchronize();
        return host_errors;
    }

    void cleanup(int device_id) override {
        cudaSetDevice(device_id);
        auto& res = resources[device_id];
        if (res.d_A) { cudaFree(res.d_A); res.d_A = nullptr; }
        if (res.d_B) { cudaFree(res.d_B); res.d_B = nullptr; }
        if (res.d_C) { cudaFree(res.d_C); res.d_C = nullptr; }
        if (res.d_C_ref) { cudaFree(res.d_C_ref); res.d_C_ref = nullptr; }
        if (res.handle) { cublasDestroy(res.handle); res.handle = nullptr; }
    }
};

std::unique_ptr<GpuBackend> create_cuda_backend() {
    return std::make_unique<CudaBackend>();
}
