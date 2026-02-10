# GPU Stress Test

## Goal

The goal is to build a SOTA stress test tool. At the beginning only a few features are implemented, but the goal is to make it a complete and feature-rich tool.

## How it works

- Fills GPU memory with large matrices of random floats
- Repeatedly multiplies them using cuBLAS (SGEMM/DGEMM) to max out the GPU
- Computes a reference result first, then compares every subsequent result against it
- If any element differs beyond a tiny epsilon, it's counted as a hardware error
- Monitors temperature and throughput (GFLOPS) throughout the test
- A healthy GPU produces zero errors — any errors indicate faulty hardware

## Features

- **CUDA Cores (FP32)** — massive matrix multiplications stress all floating-point units
- **GPU Memory (VRAM)** — fills ~90% of memory with matrices, exposing bad memory cells
- **Memory Bus** — constant read/write of large buffers tests bandwidth and data integrity
- **Temperature & Thermal Throttling** — monitors GPU temp throughout the test via NVML
- **Computational Correctness** — compares every result against a reference to catch silent data corruption
- **Sustained Load Stability** — runs for minutes to surface issues that only appear under prolonged stress

## How to run

- `mkdir -p build && cd build && cmake .. && make -j$(nproc)`
- `./gpu-burn-pro --full` — 5 minute stress test
- `./gpu-burn-pro --quick` — 10 second stress test\

## Tests

## Todo