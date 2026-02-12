/*
 * GPU Stress Test - Main Entry Point
 *
 * A GPU health testing tool.
 */

#include "gpu_backend.hpp"
#include "stress_test.hpp"
#include "result_submitter.hpp"
#include "version.hpp"
#include <iostream>
#include <string>
#include <cstring>

void show_help() {
    std::cout << "GPU Stress Test v" << STRESS_TEST_VERSION << " - GPU Health & Stress Testing Tool\n\n"
              << "Usage: gpu-stress-test [MODE] [OPTIONS]\n\n"
              << "Modes:\n"
              << "  --full      5 minute stress test\n"
              << "  --quick     10 second stress test\n\n"
              << "Options:\n"
              << "  --offline   Skip online result submission\n"
              << "  -h, --help  Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        show_help();
        return 1;
    }

    StressConfig config;
    bool offline = false;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--full") == 0) {
            config.duration_seconds = 300;
            mode_set = true;
        } else if (strcmp(argv[i], "--quick") == 0) {
            config.duration_seconds = 10;
            mode_set = true;
        } else if (strcmp(argv[i], "--offline") == 0) {
            offline = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help();
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            show_help();
            return 1;
        }
    }

    if (!mode_set) {
        std::cerr << "Error: No test mode specified (use --full or --quick)\n" << std::endl;
        show_help();
        return 1;
    }

    std::cout << "╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║       GPU Stress Test v" << STRESS_TEST_VERSION
              << "              ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Duration:     " << config.duration_seconds << " seconds" << std::endl;
    std::cout << "  Precision:    FP32 (float)" << std::endl;
    std::cout << "  Submission:   " << (offline ? "offline" : "online") << std::endl;

    try {
        auto backend = create_cuda_backend();
        StressTest stress(std::move(backend));
        stress.start(config);

        // Collect and optionally submit results
        auto results = stress.collect_results();

        if (!offline) {
            std::cout << "\nSubmitting results..." << std::endl;
            bool ok = ResultSubmitter::submit(results);
            if (ok) {
                std::cout << "Results submitted successfully." << std::endl;
            }
            // On failure, submit() already printed a warning — no error thrown
        }
    } catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
