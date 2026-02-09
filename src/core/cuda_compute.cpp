#include <iostream>
#include <cuda_runtime.h>

__global__ void stress_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = data[tid];
        // Heavy math loop to generate heat
        for (int i = 0; i < 1000; ++i) {
            val = sinf(val) * cosf(val) + sqrtf(val + 1.0f);
        }
        data[tid] = val;
    }
}

void run_stress_test(int duration_seconds) {
    int n = 1 << 22; // ~4M floats
    size_t size = n * sizeof(float);
    
    float *h_data, *d_data;
    h_data = (float*)malloc(size);
    for(int i=0; i<n; i++) h_data[i] = (float)i;
    
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "[CORE] Launching CUDA stress kernels..." << std::endl;
    
    // Simple loop for duration
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < duration_seconds) {
        stress_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_data);
    free(h_data);
}
