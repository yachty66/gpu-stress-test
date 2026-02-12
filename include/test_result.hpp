/*
 * GPU Stress Test - Test Result Data Structure
 *
 * Aggregates all metrics collected during a stress test run,
 * used for summary display and online submission.
 */

#pragma once

#include <string>

struct TestResult {
    // Per-GPU metrics
    std::string gpu_name;
    int         max_temp_c       = -1;
    int         avg_temp_c       = -1;
    size_t      gpu_memory_total_mb = 0;
    size_t      gpu_memory_used_mb  = 0;
    double      gflops           = 0.0;
    long long   errors           = 0;
    bool        passed           = true;

    // System-wide context (same for all GPUs in one run)
    std::string cuda_version;   // e.g. "12.4"
    std::string country;        // ISO-3166 alpha-2, e.g. "DE"
    std::string platform;       // "linux", "windows", "macos"
    std::string provider;       // "aws", "gcp", "azure", "lambda", "runpod", "unknown"
    std::string version;        // STRESS_TEST_VERSION
};
