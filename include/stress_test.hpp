#pragma once

#include "gpu_backend.hpp"
#include <atomic>
#include <vector>
#include <thread>

class StressTest {
public:
    StressTest(std::unique_ptr<GpuBackend> backend);
    ~StressTest();

    void start(int duration_seconds);
    void stop();
    bool is_running() const { return running; }

private:
    void run_device_test(int device_id);

    std::unique_ptr<GpuBackend> backend;
    std::atomic<bool> running{false};
    std::vector<std::thread> workers;
    
    struct DeviceStats {
        int device_id;
        std::string name;
        std::atomic<long long> total_ops{0};
        std::atomic<int> errors{0};
    };
    std::vector<std::unique_ptr<DeviceStats>> stats;
};
