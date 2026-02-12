#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# GPU Stress Test — One-liner installer & runner
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/yachty66/gpu-stress-test/main/run.sh | bash
#   curl -sSL https://raw.githubusercontent.com/yachty66/gpu-stress-test/main/run.sh | bash -s -- --quick
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Pre-flight checks ------------------------------------------------------

# 1. Check for NVIDIA GPU / drivers
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Please install NVIDIA drivers first."
fi
info "NVIDIA drivers detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# 2. Check for CUDA toolkit (nvcc)
if ! command -v nvcc &>/dev/null; then
    # Try common CUDA paths
    for p in /usr/local/cuda/bin /opt/cuda/bin; do
        if [ -x "$p/nvcc" ]; then
            export PATH="$p:$PATH"
            break
        fi
    done
fi
if ! command -v nvcc &>/dev/null; then
    error "CUDA toolkit (nvcc) not found. Please install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
fi
info "CUDA toolkit detected: $(nvcc --version | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"

# 3. Install build dependencies if missing
install_if_missing() {
    local cmd="$1" pkg="$2"
    if ! command -v "$cmd" &>/dev/null; then
        warn "$cmd not found, installing $pkg..."
        if command -v apt-get &>/dev/null; then
            apt-get update -qq && apt-get install -y -qq "$pkg" >/dev/null 2>&1 \
                || { sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg" >/dev/null 2>&1; }
        elif command -v yum &>/dev/null; then
            yum install -y -q "$pkg" >/dev/null 2>&1 \
                || sudo yum install -y -q "$pkg" >/dev/null 2>&1
        elif command -v dnf &>/dev/null; then
            dnf install -y -q "$pkg" >/dev/null 2>&1 \
                || sudo dnf install -y -q "$pkg" >/dev/null 2>&1
        else
            error "Could not install $pkg automatically. Please install it manually."
        fi
        command -v "$cmd" &>/dev/null || error "Failed to install $pkg"
        info "Installed $pkg"
    fi
}

install_if_missing cmake cmake
install_if_missing make make
install_if_missing g++ g++

# Install libcurl development headers (optional but nice to have)
if ! pkg-config --exists libcurl 2>/dev/null; then
    if [ -f /etc/debian_version ]; then
        warn "libcurl-dev not found, installing..."
        apt-get update -qq 2>/dev/null && apt-get install -y -qq libcurl4-openssl-dev >/dev/null 2>&1 \
            || { sudo apt-get update -qq 2>/dev/null && sudo apt-get install -y -qq libcurl4-openssl-dev >/dev/null 2>&1; } \
            || true
    elif [ -f /etc/redhat-release ]; then
        warn "libcurl-devel not found, installing..."
        yum install -y -q libcurl-devel >/dev/null 2>&1 \
            || sudo yum install -y -q libcurl-devel >/dev/null 2>&1 \
            || true
    fi
fi

# --- Clone & Build ----------------------------------------------------------

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

info "Cloning gpu-stress-test..."
git clone --depth 1 --quiet https://github.com/yachty66/gpu-stress-test.git "$WORKDIR/gpu-stress-test"

info "Building..."
mkdir -p "$WORKDIR/gpu-stress-test/build"
cd "$WORKDIR/gpu-stress-test/build"
cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
make -j"$(nproc)" >/dev/null 2>&1

info "Build complete!"
echo ""

# --- Run the test ------------------------------------------------------------

# Default to --full if no arguments provided
if [ $# -eq 0 ]; then
    ./gpu-stress-test --full
else
    ./gpu-stress-test "$@"
fi
