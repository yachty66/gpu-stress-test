#include "gpu_backend.hpp"
#include "stress_test.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    int duration = 60; // Default 60 seconds
    if (argc > 1) {
        duration = std::stoi(argv[1]);
    }

    std::cout << "Starting GPU Burn Pro (Simplified replica of gpu-burn)" << std::endl;
    std::cout << "Target duration: " << duration << " seconds" << std::endl;

    try {
        auto backend = create_cuda_backend();
        StressTest stress(std::move(backend));
        stress.start(duration);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
