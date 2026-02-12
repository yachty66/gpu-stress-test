/*
 * GPU Stress Test - Result Submitter
 *
 * Serializes test results to JSON and POSTs to an endpoint.
 * Uses libcurl when available, otherwise stubs out with a warning.
 */

#include "result_submitter.hpp"
#include <iostream>
#include <sstream>

#ifndef NO_CURL
#include <curl/curl.h>
#endif

// Dummy endpoint — will be replaced with Supabase URL later
static const char* SUBMIT_ENDPOINT = "https://httpbin.org/post";

namespace {

// Escape a string for JSON (handles quotes, backslashes, control chars)
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

#ifndef NO_CURL
// Callback to discard response body
size_t discard_callback(void* /*contents*/, size_t size, size_t nmemb, void* /*userp*/) {
    return size * nmemb;
}
#endif

} // anonymous namespace

namespace ResultSubmitter {

std::string to_json(const std::vector<TestResult>& results) {
    std::ostringstream ss;
    ss << "{\"results\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i > 0) ss << ",";
        ss << "{"
           << "\"gpu_name\":\"" << json_escape(r.gpu_name) << "\","
           << "\"max_temp_c\":" << r.max_temp_c << ","
           << "\"avg_temp_c\":" << r.avg_temp_c << ","
           << "\"gpu_memory_total_mb\":" << r.gpu_memory_total_mb << ","
           << "\"gpu_memory_used_mb\":" << r.gpu_memory_used_mb << ","
           << "\"gflops\":" << r.gflops << ","
           << "\"errors\":" << r.errors << ","
           << "\"passed\":" << (r.passed ? "true" : "false") << ","
           << "\"max_power_w\":" << r.max_power_w << ","
           << "\"avg_power_w\":" << r.avg_power_w << ","
           << "\"cuda_version\":\"" << json_escape(r.cuda_version) << "\","
           << "\"country\":\"" << json_escape(r.country) << "\","
           << "\"platform\":\"" << json_escape(r.platform) << "\","
           << "\"provider\":\"" << json_escape(r.provider) << "\","
           << "\"version\":\"" << json_escape(r.version) << "\""
           << "}";
    }

    ss << "]}";
    return ss.str();
}

bool submit(const std::vector<TestResult>& results) {
#ifdef NO_CURL
    std::cerr << "Warning: Result submission unavailable (built without curl support)" << std::endl;
    (void)results;
    return false;
#else
    std::string json = to_json(results);

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Warning: Could not initialize curl for result submission" << std::endl;
        return false;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, SUBMIT_ENDPOINT);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discard_callback);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Warning: Result submission failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    if (http_code < 200 || http_code >= 300) {
        std::cerr << "Warning: Result submission returned HTTP " << http_code << std::endl;
        return false;
    }

    return true;
#endif
}

} // namespace ResultSubmitter
