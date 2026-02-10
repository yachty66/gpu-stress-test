/*
 * GPU Stress Test - Test Orchestrator
 */

#include "stress_test.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <cmath>

std::atomic<bool> StressTest::running{false};

StressTest::StressTest(std::unique_ptr<GpuBackend> b) : backend(std::move(b)) {
}

StressTest::~StressTest() {
    stop();
}

void StressTest::request_stop() {
    running = false;
}

void StressTest::start(const StressConfig& cfg) {
    config = cfg;

    // Set up signal handlers
    std::signal(SIGTERM, [](int) { StressTest::request_stop(); });
    std::signal(SIGINT, [](int) { StressTest::request_stop(); });

    // Enumerate devices
    auto devices = backend->list_devices();
    if (devices.empty()) {
        throw std::runtime_error("No CUDA devices found");
    }

    // Filter to target device if specified
    std::vector<DeviceInfo> target_devices;
    if (config.target_device >= 0) {
        bool found = false;
        for (const auto& dev : devices) {
            if (dev.id == config.target_device) {
                target_devices.push_back(dev);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Device " + std::to_string(config.target_device) + " not found");
        }
    } else {
        target_devices = devices;
    }

    // Initialize stats and device mapping
    for (size_t i = 0; i < target_devices.size(); ++i) {
        auto s = std::make_unique<DeviceStats>();
        s->device_id = target_devices[i].id;
        s->name = target_devices[i].name;
        device_to_index[target_devices[i].id] = i;
        stats.push_back(std::move(s));
    }

    // Initialize devices
    std::cout << "\nInitializing devices..." << std::endl;
    for (const auto& s : stats) {
        backend->initialize_device(s->device_id, config);
    }

    // Start worker threads
    running = true;
    for (const auto& s : stats) {
        workers.emplace_back(&StressTest::run_device_test, this, s->device_id);
    }

    std::cout << "\nRunning stress test for " << config.duration_seconds << " seconds...\n" << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    while (running) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count());

        if (elapsed >= config.duration_seconds) {
            running = false;
            break;
        }

        print_progress(elapsed, config.duration_seconds);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    stop();
    std::cout << std::endl;
    print_summary();
}

void StressTest::print_progress(int elapsed, int duration) {
    float pct = std::fmin((float)elapsed / (float)duration * 100.0f, 100.0f);

    std::cout << "\r" << std::fixed << std::setprecision(1) << pct << "%  ";

    for (const auto& s : stats) {
        // Update temperature
        int temp = backend->get_temperature(s->device_id);
        if (temp >= 0) {
            s->temperature = temp;
            int current_max = s->max_temperature.load();
            while (temp > current_max) {
                s->max_temperature.compare_exchange_weak(current_max, temp);
            }
        }

        double gflops = 0.0;
        {
            std::lock_guard<std::mutex> lock(s->gflops_mutex);
            gflops = s->gflops;
        }

        std::cout << "GPU " << s->device_id << ": ";
        std::cout << std::fixed << std::setprecision(0) << gflops << " Gflop/s";

        long long errors = s->total_errors.load();
        if (errors > 0) {
            std::cout << " ERR:" << errors;
        }

        if (temp >= 0) {
            std::cout << " " << temp << "C";
        }

        std::cout << "  ";
    }
    std::cout << std::flush;
}

void StressTest::print_summary() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Tested " << stats.size() << " GPU(s):" << std::endl;
    std::cout << "========================================" << std::endl;

    bool any_faulty = false;
    for (const auto& s : stats) {
        bool is_faulty = s->total_errors.load() > 0;
        if (is_faulty) any_faulty = true;

        double gflops;
        {
            std::lock_guard<std::mutex> lock(s->gflops_mutex);
            gflops = s->gflops;
        }

        std::cout << "  GPU " << s->device_id << " (" << s->name << "): "
                  << (is_faulty ? "FAULTY" : "OK");

        if (is_faulty) {
            std::cout << " - " << s->total_errors.load() << " errors detected";
        }

        std::cout << std::fixed << std::setprecision(1);
        std::cout << " - " << gflops << " Gflop/s";

        if (s->max_temperature.load() > 0) {
            std::cout << " - max " << s->max_temperature.load() << "C";
        }

        std::cout << " - " << s->total_ops.load() << " ops";
        std::cout << std::endl;
    }

    std::cout << "========================================" << std::endl;
    if (any_faulty) {
        std::cout << "RESULT: FAIL — Faulty GPU(s) detected!" << std::endl;
    } else {
        std::cout << "RESULT: PASS — All GPUs healthy." << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

void StressTest::stop() {
    if (!running && workers.empty()) return;
    running = false;
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    workers.clear();

    if (backend) {
        for (const auto& s : stats) {
            backend->cleanup(s->device_id);
        }
    }
}

void StressTest::run_device_test(int device_id) {
    auto it = device_to_index.find(device_id);
    if (it == device_to_index.end()) return;
    auto& s = *stats[it->second];

    auto last_report = std::chrono::steady_clock::now();
    long long ops_since_report = 0;

    while (running) {
        try {
            StressResult result = backend->run_stress_iteration(device_id);
            s.total_errors += result.errors;
            s.total_ops += 1;
            ops_since_report++;

            if (result.errors > 0) {
                s.faulty = true;
            }

            // Update GFLOPS every few iterations
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_report).count();
            if (dt >= 1.0) {
                double gflops = (double)(ops_since_report * result.ops) / dt / 1e9;
                {
                    std::lock_guard<std::mutex> lock(s.gflops_mutex);
                    s.gflops = gflops;
                }
                last_report = now;
                ops_since_report = 0;
            }
        } catch (const std::exception& e) {
            std::cerr << "\nGPU " << device_id << " error: " << e.what() << std::endl;
            s.faulty = true;
            break;
        }
    }
}
