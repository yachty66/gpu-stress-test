# Zuse GPU Stress Test: Clean Implementation Plan (Headless-First)

The goal is to build the world's most stable and comprehensive GPU stress test, specifically designed for professional environments, data centers, and power users.

## 核心 (The Core) - Strictly Headless
Unlike gaming benchmarks, this tool is built to run without a monitor or GUI. It is a "Compute" stress test.
*   **Default Behavior**: Run in the terminal. No windowing logic. No GUI deps.
*   **Technologies**: Vulkan Compute (Primary), CUDA/ROCm (Vendor Specific), Metal Performance Shaders.

## 1. Functional Blocks to Test (NVIDIA Focus)
*   **Core Compute**: FP32/INT32 math loops to saturate CUDA cores.
*   **Ray Tracing (Headless)**: Use NVIDIA OptiX or Vulkan RTX acceleration structures.
*   **Tensor Cores**: Matrix multiplication kernels to max out Tensor core throughput.
*   **VRAM Torture**: Bandwidth saturation and pattern-based integrity checks (ECC error logging via NVML).
*   **Power Spike Simulation**: Low-latency load cycling to trigger transient power excursions.

## 2. Professional CLI & Telemetry
A terminal interface optimized for NVIDIA setups:
*   **Real-time Sampling**: 10ms-100ms interval polling via NVML (Temp, Wattage, Clock, Fan).

## 3. Market Differentiation (The X-Factor)
*   **Anti-Cheat**: Verification that the driver isn't compromising compute precision (NVIDIA-specific checks).
*   **System Health Validation**: 
    -   Automatically flag "buggy" NVIDIA driver versions.
    -   Monitor "12VHPWR" power connector health via NVML telemetry.

## 4. Implementation Roadmap (v1 CLI)
1.  **Engine Layer**: Build basic Vulkan Compute kernel to apply 100% load.
2.  **Telemetry Layer**: Integrate NVML (NVIDIA) and ADL/OD (AMD) for real-time stats.
3.  **Stress Logic**: Implement RT and Tensor core "torture" via compute kernels.
4.  **Reporting**: Finish the terminal UI and JSON logging.
