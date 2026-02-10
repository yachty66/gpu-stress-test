#include "stress_test.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

StressTest::StressTest(std::unique_ptr<GpuBackend> b) : backend(std::move(b)) {
    auto devices = backend->list_devices();
    for (const auto& dev : devices) {
        auto s = std::make_unique<DeviceStats>();
        s->device_id = dev.id;
        s->name = dev.name;
        stats.push_back(std::move(s));
    }
}

StressTest::~StressTest() {
    stop();
}

void StressTest::start(int duration_seconds) {
    running = true;
    for (size_t i = 0; i < stats.size(); ++i) {
        backend->initialize_device(stats[i]->device_id);
        workers.emplace_back(&StressTest::run_device_test, this, stats[i]->device_id);
    }

    auto start_time = std::chrono::steady_clock::now();
    while (running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        
        if (elapsed >= duration_seconds) {
            running = false;
            break;
        }

        // Print progress
        std::cout << "\rElapsed: " << elapsed << "s | ";
        for (const auto& s : stats) {
            std::cout << "GPU " << s->device_id << " (" << s->name << "): " 
                      << "Errors: " << s->errors << " | ";
        }
        std::cout << std::flush;
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    stop();
    std::cout << "\nTest Complete." << std::endl;
}

void StressTest::stop() {
    running = false;
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    workers.clear();
    
    for (const auto& s : stats) {
        backend->cleanup(s->device_id);
    }
}

void StressTest::run_device_test(int device_id) {
    auto& s = *stats[device_id]; // Simplified mapping
    while (running) {
        int errors = backend->run_stress_iteration(device_id, 100);
        s.errors += errors;
        s.total_ops += 100;
    }
}
