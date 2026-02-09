#include "nvml_monitor.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>

extern void run_stress_test(int duration_seconds);

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "   ZUSE GPU STRESS TEST v1.0 (NVIDIA)   " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[INFO] Mode: HEADLESS (Default)" << std::endl;
    
    NvmlMonitor monitor;
    if (!monitor.initialize()) {
        return 1;
    }
    
    std::cout << "[INFO] Target: " << monitor.getGpuName() << std::endl;
    
    // Start stress test in a separate thread if we want real-time telemetry
    // For now, let's just run them sequentially for a simple v1 check
    int test_duration = 10;
    
    std::cout << "[INFO] Starting 10s stress test..." << std::endl;
    
    std::thread telemetry_thread([&monitor, test_duration]() {
        auto start = std::chrono::steady_clock::now();
        auto duration = std::chrono::seconds(test_duration);
        while (std::chrono::steady_clock::now() - start < duration) {
            GpuStats stats = monitor.getStats();
            std::cout << "\r" << std::setw(3) << stats.temp << "C | " 
                      << std::setw(3) << stats.power << "W | " 
                      << std::setw(4) << stats.clock << "MHz | " 
                      << std::setw(2) << stats.utilization << "% | VRAM: " 
                      << stats.vram_used << "/" << stats.vram_total << "MB " << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });
    
    run_stress_test(test_duration);
    
    if (telemetry_thread.joinable()) {
        telemetry_thread.join();
    }
    
    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "[DONE] Stress test complete." << std::endl;
    
    return 0;
}
