/*
 * GPU Stress Test - System Information Detection
 *
 * Best-effort detection of platform, cloud provider, and country.
 * All functions are non-throwing and return "unknown" on failure.
 */

#include "system_info.hpp"
#include <cstdlib>
#include <cctype>
#include <cstdint>

#ifndef NO_CURL
#include <curl/curl.h>
#endif

namespace {

// Callback for libcurl to write response data into a string
#ifndef NO_CURL
size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    static_cast<std::string*>(userp)->append(static_cast<char*>(contents), total);
    return total;
}

// Simple HTTP GET with timeout. Returns response body or empty string on failure.
std::string http_get(const std::string& url, long timeout_seconds = 3) {
    std::string response;
    CURL* curl = curl_easy_init();
    if (!curl) return "";

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) return "";
    return response;
}
#endif

// Check if an environment variable is set and non-empty
bool has_env(const char* name) {
    const char* val = std::getenv(name);
    return val != nullptr && val[0] != '\0';
}

// Extract a simple JSON string value by key (very basic, no nesting)
// Looks for "key":"value" and returns value
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

// Convert a 2-letter country code (e.g. "US") to a flag emoji (e.g. 🇺🇸)
// using Unicode Regional Indicator Symbols (U+1F1E6 to U+1F1FF)
std::string country_code_to_flag(const std::string& code) {
    if (code.size() != 2) return code;

    std::string flag;
    for (char c : code) {
        char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (upper < 'A' || upper > 'Z') return code;

        uint32_t cp = 0x1F1E6 + (upper - 'A');
        // Encode as UTF-8 (4 bytes for code points in U+10000..U+1FFFF)
        flag += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
        flag += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        flag += static_cast<char>(0x80 | ((cp >> 6)  & 0x3F));
        flag += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return flag;
}

} // anonymous namespace

namespace SystemInfo {

std::string get_platform() {
#if defined(__linux__)
    return "linux";
#elif defined(_WIN32) || defined(_WIN64)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "unknown";
#endif
}

std::string get_provider() {
    // Check well-known environment variables for major cloud/GPU providers
    if (has_env("RUNPOD_POD_ID") || has_env("RUNPOD_GPU_COUNT"))
        return "runpod";
    if (has_env("LAMBDA_API_KEY") || has_env("LAMBDA_NODE_ID"))
        return "lambda";
    if (has_env("AWS_EXECUTION_ENV") || has_env("AWS_REGION") || has_env("EC2_INSTANCE_ID"))
        return "aws";
    if (has_env("GOOGLE_CLOUD_PROJECT") || has_env("GCE_METADATA_HOST") || has_env("GCLOUD_PROJECT"))
        return "gcp";
    if (has_env("MSI_ENDPOINT") || has_env("AZURE_CLIENT_ID") || has_env("WEBSITE_INSTANCE_ID"))
        return "azure";
    if (has_env("VAST_CONTAINERLABEL"))
        return "vast";
    if (has_env("COREWEAVE_ENV"))
        return "coreweave";

    return "unknown";
}

std::string get_country() {
#ifndef NO_CURL
    try {
        std::string response = http_get("http://ip-api.com/json/?fields=countryCode", 3);
        if (!response.empty()) {
            std::string code = extract_json_string(response, "countryCode");
            if (!code.empty()) return country_code_to_flag(code);
        }
    } catch (...) {
        // Fall through
    }
#endif
    return "unknown";
}

} // namespace SystemInfo
