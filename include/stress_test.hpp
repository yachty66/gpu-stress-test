/*
 * GPU Stress Test - Test Orchestrator
 */

#pragma once

#include "gpu_backend.hpp"
#include "test_result.hpp"
#include <atomic>
#include <vector>
#include <thread>
#include <map>
#include <mutex>

class StressTest {
public:
    StressTest(std::unique_ptr<GpuBackend> backend);
    ~StressTest();

    void start(const StressConfig& config);
    void stop();
    bool is_running() const { return running; }

    // Collect final results after test completes
    std::vector<TestResult> collect_results();

    // Called from signal handlers
    static void request_stop();

private:
    void run_device_test(int device_id);
    void print_progress(int elapsed, int duration);
    void print_summary();

    std::unique_ptr<GpuBackend> backend;
    static std::atomic<bool> running;
    std::vector<std::thread> workers;
    StressConfig config;
    
    struct DeviceStats {
        int device_id;
        std::string name;
        std::atomic<long long> total_ops{0};
        std::atomic<long long> total_errors{0};
        std::atomic<int> temperature{0};
        std::atomic<int> max_temperature{0};
        std::atomic<long long> temp_sum{0};
        std::atomic<int> temp_samples{0};
        size_t memory_total_mb{0};
        size_t memory_used_mb{0};
        double gflops{0.0};
        std::mutex gflops_mutex;
        bool faulty{false};
    };
    std::vector<std::unique_ptr<DeviceStats>> stats;
    std::map<int, size_t> device_to_index;  // device_id -> stats index

    int matrix_size = 2048;  // SIZE used for GFLOPS calculation
};
