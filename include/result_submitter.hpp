/*
 * GPU Stress Test - Result Submitter
 *
 * POSTs test results to an online endpoint.
 * Failures are non-fatal (warning printed, no exception thrown).
 */

#pragma once

#include "test_result.hpp"
#include <vector>

namespace ResultSubmitter {
    // Submits results via HTTP POST. Returns true on success, false on failure.
    // On failure, prints a warning to stderr but never throws.
    bool submit(const std::vector<TestResult>& results);

    // Serializes results to a JSON string for submission
    std::string to_json(const std::vector<TestResult>& results);
}
