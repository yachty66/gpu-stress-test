#pragma once
#include <string>
#include <vector>

struct GpuStats {
    unsigned int temp;
    unsigned int power;
    unsigned int clock;
    unsigned int utilization;
    unsigned long long vram_used;
    unsigned long long vram_total;
};

class NvmlMonitor {
public:
    NvmlMonitor();
    ~NvmlMonitor();

    bool initialize();
    GpuStats getStats(int deviceIndex = 0);
    std::string getGpuName(int deviceIndex = 0);

private:
    bool m_initialized = false;
};
