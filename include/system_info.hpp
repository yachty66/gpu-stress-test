/*
 * GPU Stress Test - System Information Detection
 *
 * Best-effort detection of platform, cloud provider, and country.
 * All functions are non-throwing and return "unknown" on failure.
 */

#pragma once

#include <string>

namespace SystemInfo {
    // Returns "linux", "windows", or "macos"
    std::string get_platform();

    // Detects cloud provider by checking environment variables
    // Returns "aws", "gcp", "azure", "lambda", "runpod", or "unknown"
    std::string get_provider();

    // Detects country via IP geolocation (http://ip-api.com)
    // Returns ISO-3166 alpha-2 code (e.g. "DE") or "unknown"
    // Uses short timeout to avoid blocking
    std::string get_country();
}
