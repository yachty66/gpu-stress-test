# GPU Burn Pro

A GPU health and stress testing tool. Performs intensive matrix multiplications (SGEMM/DGEMM) via cuBLAS and validates results to detect hardware faults.

Inspired by [gpu-burn](https://github.com/wilicc/gpu-burn) by Ville Timonen (BSD license).

## Features

- **Matrix multiplication stress test** using cuBLAS (SGEMM / DGEMM)
- **Error detection** via epsilon-based result comparison
- **Temperature monitoring** via NVML
- **GFLOPS reporting** — detect performance degradation
- **Double precision** (FP64) support
- **Tensor Core** support
- **Multi-GPU** — tests all GPUs in parallel
- **Configurable memory usage** — scale matrix size to fill GPU memory
- **Signal handling** — clean shutdown on Ctrl+C

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Requirements
- CUDA Toolkit (≥ 10.0)
- cuBLAS
- NVML (included with CUDA Toolkit / NVIDIA drivers)
- CMake ≥ 3.18
- C++17 compiler

## Usage

```bash
# Basic 60-second stress test on all GPUs
./gpu-burn-pro 60

# Use double precision (FP64)
./gpu-burn-pro -d 120

# Use Tensor Cores
./gpu-burn-pro -tc 60

# Test only GPU 0
./gpu-burn-pro -i 0 60

# Use 50% of GPU memory
./gpu-burn-pro -m 50% 60

# Use exactly 4096 MB
./gpu-burn-pro -m 4096 60

# List available GPUs
./gpu-burn-pro -l

# Show help
./gpu-burn-pro -h
```

## Output

```
╔══════════════════════════════════════════╗
║       GPU Burn Pro - Stress Tester       ║
╚══════════════════════════════════════════╝

Configuration:
  Duration:     60 seconds
  Precision:    FP32 (float)
  Tensor Cores: disabled
  Target GPU:   all

Initializing devices...
  GPU 0: 14400 MB allocated (15360 MB free), using FLOATS

Running stress test for 60 seconds...

100.0%  GPU 0: 5234 Gflop/s 72C

========================================
Tested 1 GPU(s):
========================================
  GPU 0 (Tesla T4): OK - 5234.0 Gflop/s - max 72C - 1847 ops
========================================
RESULT: PASS — All GPUs healthy.
========================================
```

## How It Works

1. **Initialize** — Allocates GPU memory and fills matrices A & B with random floats
2. **Reference** — Computes C_ref = A × B using cuBLAS
3. **Stress loop** — Repeatedly computes C = A × B and compares against C_ref
4. **Detection** — If `|C[i] - C_ref[i]| > epsilon`, the element is counted as faulty
5. **Report** — Summarizes errors, throughput (GFLOPS), and temperature per GPU

## License

This project contains code derived from [gpu-burn](https://github.com/wilicc/gpu-burn),
which is licensed under the BSD 2-Clause License. See the source files for the full
license text.
