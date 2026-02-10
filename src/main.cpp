/*
 * GPU Stress Test - Main Entry Point
 *
 * A GPU health testing tool inspired by gpu-burn.
 * https://github.com/wilicc/gpu-burn
 */

#include "gpu_backend.hpp"
#include "stress_test.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

void show_help() {
    std::cout << "GPU Burn Pro - GPU Health & Stress Testing Tool\n"
              << "Based on gpu-burn by Ville Timonen\n\n"
              << "Usage: gpu-burn-pro [OPTIONS] [TIME]\n\n"
              << "Options:\n"
              << "  -d          Use double precision (FP64)\n"
              << "  -tc         Use Tensor Cores (if available)\n"
              << "  -m X        Use X MB of GPU memory\n"
              << "  -m N%       Use N% of available GPU memory (default: 90%)\n"
              << "  -i N        Test only GPU N\n"
              << "  -l          List all GPUs\n"
              << "  -h          Show this help\n\n"
              << "Examples:\n"
              << "  gpu-burn-pro 60          # 60 second test, all GPUs, FP32\n"
              << "  gpu-burn-pro -d 3600     # 1 hour test with doubles\n"
              << "  gpu-burn-pro -tc 120     # 2 minute test with Tensor Cores\n"
              << "  gpu-burn-pro -i 0 60     # Test only GPU 0 for 60s\n"
              << "  gpu-burn-pro -m 50% 60   # Use 50% of GPU memory\n"
              << "  gpu-burn-pro -l          # List available GPUs\n";
}

void list_gpus() {
    auto backend = create_cuda_backend();
    auto devices = backend->list_devices();

    if (devices.empty()) {
        std::cerr << "No CUDA GPUs found." << std::endl;
        return;
    }

    std::cout << "Available GPUs:" << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  GPU " << dev.id << ": " << dev.name
                  << " - " << (dev.total_memory / 1024 / 1024) << " MB total"
                  << " (" << (dev.free_memory / 1024 / 1024) << " MB free)"
                  << std::endl;
    }
}

// Parse memory argument: "512" = 512 MB, "50%" = 50% of free
// Returns: >0 for bytes, <0 for percentage (negated), 0 for error
ssize_t decode_memory(const char* s) {
    char* end;
    long long val = strtoll(s, &end, 10);
    if (s == end) return 0;
    if (*end == '%') {
        return (end[1] == '\0') ? -val : 0;
    }
    return (val > 0 && *end == '\0') ? val * 1024 * 1024 : 0;
}

int main(int argc, char* argv[]) {
    StressConfig config;
    int positional_start = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help();
            return 0;
        }
        if (strcmp(argv[i], "-l") == 0) {
            list_gpus();
            return 0;
        }
        if (strcmp(argv[i], "-d") == 0) {
            config.precision = Precision::DOUBLE;
            positional_start = i + 1;
            continue;
        }
        if (strcmp(argv[i], "-tc") == 0) {
            config.use_tensor_cores = true;
            positional_start = i + 1;
            continue;
        }
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.target_device = atoi(argv[++i]);
            positional_start = i + 1;
            continue;
        }
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            ssize_t mem = decode_memory(argv[++i]);
            if (mem == 0) {
                std::cerr << "Invalid memory specification: " << argv[i] << std::endl;
                return 1;
            }
            if (mem < 0) {
                config.memory_fraction = (double)(-mem) / 100.0;
                config.memory_bytes = 0;
            } else {
                config.memory_bytes = mem;
                config.memory_fraction = 0;
            }
            positional_start = i + 1;
            continue;
        }
        // If it looks like a number, it's the duration
        char* end;
        long val = strtol(argv[i], &end, 10);
        if (end != argv[i] && *end == '\0' && val > 0) {
            config.duration_seconds = static_cast<int>(val);
            positional_start = i + 1;
            continue;
        }
        // Unknown argument
        std::cerr << "Unknown option: " << argv[i] << std::endl;
        show_help();
        return 1;
    }

    // Default duration if not specified
    if (config.duration_seconds <= 0) {
        config.duration_seconds = 60;
    }

    std::cout << "╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║       GPU Burn Pro - Stress Tester       ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Duration:     " << config.duration_seconds << " seconds" << std::endl;
    std::cout << "  Precision:    " << (config.precision == Precision::DOUBLE ? "FP64 (double)" : "FP32 (float)") << std::endl;
    std::cout << "  Tensor Cores: " << (config.use_tensor_cores ? "enabled" : "disabled") << std::endl;
    if (config.target_device >= 0) {
        std::cout << "  Target GPU:   " << config.target_device << std::endl;
    } else {
        std::cout << "  Target GPU:   all" << std::endl;
    }

    try {
        auto backend = create_cuda_backend();
        StressTest stress(std::move(backend));
        stress.start(config);
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
