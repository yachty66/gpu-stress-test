#include "nvml_monitor.hpp"
#include <iostream>

#ifdef HAS_NVML
#include <nvml.h>
#endif

NvmlMonitor::NvmlMonitor() {}

NvmlMonitor::~NvmlMonitor() {
#ifdef HAS_NVML
    if (m_initialized) {
        nvmlShutdown();
    }
#endif
}

bool NvmlMonitor::initialize() {
#ifdef HAS_NVML
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result) {
        std::cerr << "[ERROR] Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    m_initialized = true;
    return true;
#else
    std::cout << "[WARN] NVML not linked. Using mock telemetry." << std::endl;
    return true;
#endif
}

GpuStats NvmlMonitor::getStats(int deviceIndex) {
    GpuStats stats = {0, 0, 0, 0, 0, 0};
    
#ifdef HAS_NVML
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    if (NVML_SUCCESS == result) {
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &stats.temp);
        
        unsigned int pwr;
        nvmlDeviceGetPowerUsage(device, &pwr);
        stats.power = pwr / 1000; // mW to W
        
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &stats.clock);
        
        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);
        stats.utilization = util.gpu;
        
        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(device, &mem);
        stats.vram_used = mem.used / (1024 * 1024);
        stats.vram_total = mem.total / (1024 * 1024);
    }
#else
    // Mock Data for testing on non-NVIDIA systems (like your Mac development environment)
    stats.temp = 45;
    stats.power = 120;
    stats.clock = 1800;
    stats.utilization = 95;
    stats.vram_used = 4096;
    stats.vram_total = 8192;
#endif

    return stats;
}

std::string NvmlMonitor::getGpuName(int deviceIndex) {
#ifdef HAS_NVML
    nvmlDevice_t device;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    if (NVML_SUCCESS == nvmlDeviceGetHandleByIndex(deviceIndex, &device)) {
        nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        return std::string(name);
    }
#endif
    return "NVIDIA GeForce Mock-RTX 4090";
}
