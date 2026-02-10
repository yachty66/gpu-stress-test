/*
 * GPU Stress Test - Backend Interface
 * 
 * Portions of the stress testing logic are derived from gpu-burn
 * (https://github.com/wilicc/gpu-burn), which is BSD-licensed:
 * Copyright (c) 2022, Ville Timonen. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

enum class Precision {
    FLOAT,
    DOUBLE
};

struct DeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
};

struct StressResult {
    int errors;
    long long ops;
    double elapsed_seconds;
};

struct StressConfig {
    int duration_seconds = 60;
    Precision precision = Precision::FLOAT;
    bool use_tensor_cores = false;
    double memory_fraction = 0.9;  // Use 90% of free memory by default
    ssize_t memory_bytes = 0;      // 0 = use memory_fraction, >0 = exact bytes
    int target_device = -1;        // -1 = all devices
};

class GpuBackend {
public:
    virtual ~GpuBackend() = default;

    virtual std::vector<DeviceInfo> list_devices() = 0;
    virtual bool initialize_device(int device_id, const StressConfig& config) = 0;
    
    // Core stress operation: performs matrix multiplications and compares results
    virtual StressResult run_stress_iteration(int device_id) = 0;
    
    // Get current GPU temperature in Celsius (-1 if unavailable)
    virtual int get_temperature(int device_id) = 0;

    virtual void cleanup(int device_id) = 0;
};

std::unique_ptr<GpuBackend> create_cuda_backend();
