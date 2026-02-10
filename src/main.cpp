/*
 * GPU Stress Test - Main Entry Point
 *
 * A GPU health testing tool.
 */

#include "gpu_backend.hpp"
#include "stress_test.hpp"
#include <iostream>
#include <string>
#include <cstring>

void show_help() {
    std::cout << "GPU Stress Test - GPU Health & Stress Testing Tool\n\n"
              << "Usage: gpu-stress-test [MODE]\n\n"
              << "Modes:\n"
              << "  --full    5 minute stress test\n"
              << "  --quick   10 second stress test\n"
              << "  -h        Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        show_help();
        return 1;
    }

    StressConfig config;

    if (strcmp(argv[1], "--full") == 0) {
        config.duration_seconds = 300;
    } else if (strcmp(argv[1], "--quick") == 0) {
        config.duration_seconds = 10;
    } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        show_help();
        return 0;
    } else {
        std::cerr << "Unknown option: " << argv[1] << std::endl;
        show_help();
        return 1;
    }

    std::cout << "╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║       GPU Stress Test                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Duration:     " << config.duration_seconds << " seconds" << std::endl;
    std::cout << "  Precision:    FP32 (float)" << std::endl;

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
