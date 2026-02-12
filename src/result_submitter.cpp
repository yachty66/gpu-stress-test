/*
 * GPU Stress Test - Result Submitter
 *
 * Serializes test results to JSON and POSTs to Supabase.
 * Uses libcurl when available, otherwise stubs out with a warning.
 */

#include "result_submitter.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

#ifndef NO_CURL
#include <curl/curl.h>
#endif

// Supabase REST API endpoint for the gpu_stress_test_results table
static const char* SUPABASE_URL =
    "https://dasomvwqrumtgwktapfa.supabase.co/rest/v1/gpu_stress_test_results";

// Supabase anon key — safe to embed; security is enforced by Row Level Security (RLS)
static const char* SUPABASE_ANON_KEY =
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRhc29tdndxcnVtdGd3a3RhcGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA5MDc4NTQsImV4cCI6MjA4NjQ4Mzg1NH0."
    "FWMcEQdWRWYNxbLqfPhZev64AFAOACbRD7Ll3w52BEI";

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

    // Supabase REST API accepts a JSON array of row objects for bulk insert
    ss << "[";

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

    ss << "]";
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

    // Build auth header strings
    std::string apikey_header = std::string("apikey: ") + SUPABASE_ANON_KEY;
    std::string auth_header   = std::string("Authorization: Bearer ") + SUPABASE_ANON_KEY;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, apikey_header.c_str());
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Prefer: return=minimal");

    curl_easy_setopt(curl, CURLOPT_URL, SUPABASE_URL);
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

bool save_to_file(const std::vector<TestResult>& results, const std::string& filename) {
    std::string json = to_json(results);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    file << json;
    file.close();

    if (file.fail()) {
        std::cerr << "Warning: Failed to write results to " << filename << std::endl;
        return false;
    }

    return true;
}

} // namespace ResultSubmitter
