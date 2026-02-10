#pragma once

#include <string>
#include <vector>
#include <memory>

struct DeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
};

class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    virtual std::vector<DeviceInfo> list_devices() = 0;
    virtual bool initialize_device(int device_id) = 0;
    
    // Core stress operation: performs matrix multiplications and returns error count
    virtual int run_stress_iteration(int device_id, int iterations) = 0;
    
    virtual void cleanup(int device_id) = 0;
};

std::unique_ptr<GpuBackend> create_cuda_backend();
