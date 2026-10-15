#!/usr/bin/env bash
# ==============================================================================
# 🚀 STRATA ARCH LINUX NATIVE TURBO BUILDER & KERNEL TUNER
#    Target Hardware:
#      - CPU: Intel Hybrid Alder/Raptor Lake (24 threads, native AVX2 + AVX-VNNI)
#      - GPU: NVIDIA RTX A5500 Laptop GPU (16 GB VRAM, Compute Capability sm_86)
#      - Model: Huihui Swift 1.5 (Qwen3.8-Flash-Next 125B MoE IQ3_XXS)
# ==============================================================================
set -euo pipefail

STRATA_DIR="${STRATA_DIR:-/mnt/d/03_Devops/Strata}"
if [[ ! -f "${STRATA_DIR}/CMakeLists.txt" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${SCRIPT_DIR}/CMakeLists.txt" ]]; then
        STRATA_DIR="${SCRIPT_DIR}"
    else
        echo "[X] Could not find Strata CMakeLists.txt at ${STRATA_DIR}"
        exit 1
    fi
fi

echo "================================================================================"
echo " 🚀 STRATA ARCH LINUX NATIVE AVX-VNNI + CUDA sm_86 TURBO BUILDER"
echo "    Strata Root: ${STRATA_DIR}"
echo "================================================================================"

# ------------------------------------------------------------------------------
# [1/5] Arch Linux Build Dependencies Check (pacman)
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Checking Arch Linux build tools (cmake, ninja, gcc, cuda)..."
NEED_PKGS=()
command -v cmake >/dev/null 2>&1 || NEED_PKGS+=("cmake")
command -v ninja >/dev/null 2>&1 || NEED_PKGS+=("ninja")
command -v g++   >/dev/null 2>&1 || NEED_PKGS+=("base-devel")
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x /opt/cuda/bin/nvcc ]] && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    NEED_PKGS+=("cuda")
fi

if [[ ${#NEED_PKGS[@]} -gt 0 ]]; then
    echo "      Installing missing Arch packages: ${NEED_PKGS[*]}"
    sudo pacman -S --needed --noconfirm "${NEED_PKGS[@]}"
fi

# Locate nvcc on Arch Linux (/opt/cuda/bin/nvcc is standard on Arch)
if [[ -x /opt/cuda/bin/nvcc ]]; then
    NVCC_BIN="/opt/cuda/bin/nvcc"
    export PATH="/opt/cuda/bin:${PATH}"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
    export PATH="/usr/local/cuda/bin:${PATH}"
else
    echo "[X] nvcc not found! Install CUDA on Arch: sudo pacman -S cuda"
    exit 1
fi

CUDA_VER="$("${NVCC_BIN}" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
echo "      Found NVCC: ${NVCC_BIN} (CUDA ${CUDA_VER})"
echo "      CPU Flags : $(grep -m1 -oE 'avx2|avx_vnni|bmi2|fma' /proc/cpuinfo | tr '\n' ' ')"

# ------------------------------------------------------------------------------
# [2/5] Tune Arch Linux CPU Governor, P-State & VM Subsystem for 64 GB MoE
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Applying Arch Linux Kernel & Intel P-Core/E-Core Turbo Tuning..."
if sudo -n true 2>/dev/null || sudo -v; then
    # Set all 24 logical cores to performance governor
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "${gov}" ]] && echo performance | sudo tee "${gov}" >/dev/null || true
    done
    # Set Intel P-State Energy Performance Preference (EPP) to performance
    for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
        [[ -w "${epp}" ]] && echo performance | sudo tee "${epp}" >/dev/null || true
    done
    # Ensure Intel Turbo Boost is enabled
    if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null || true
    fi
    # Memory subsystem tuning for 35.06 GiB pinned expert cache + 6.19 GiB KV cache
    [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]] && echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null || true
    sudo sysctl -w vm.max_map_count=1048576 >/dev/null 2>&1 || true
    sudo sysctl -w vm.swappiness=10 >/dev/null 2>&1 || true
    sudo sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
    # Lock NVIDIA GPU clocks to persistence mode
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
    echo "      [ok] CPU governor=performance, EPP=performance, THP=madvise, swappiness=10"
else
    echo "      [!] Skipped root sysctl tuning (no sudo)."
fi

# Ensure third_party/llama.cpp exists (uses setup.py helper if not yet unpacked)
PY_BIN="${STRATA_DIR}/.venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
    PY_BIN="$(command -v python3)"
fi

LLAMA_DIR="${STRATA_DIR}/third_party/llama.cpp"
if [[ ! -f "${LLAMA_DIR}/ggml/CMakeLists.txt" ]]; then
    echo "      Fetching pinned third_party/llama.cpp via setup.py..."
    "${PY_BIN}" -c "import sys; sys.path.insert(0, '${STRATA_DIR}'); import setup; print(setup.get_llama_cpp())"
fi

# ------------------------------------------------------------------------------
# [3/5] Configure & Compile Native Strata Binary (STRATA_PORTABLE=OFF, sm_86)
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Compiling Strata natively in RAM (/tmp/strata-build-native)..."
echo "      -DSTRATA_PORTABLE=OFF       (unlocks native Intel AVX-VNNI vpdpbusd + GGML_NATIVE=ON)"
echo "      -DCMAKE_CUDA_ARCHITECTURES=86 (pure RTX A5500 sm_86 SASS, no multi-arch fatbin bloat)"

BUILD_DIR="/tmp/strata-build-native"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Handle Arch Linux bleeding-edge GCC with NVCC via -allow-unsupported-compiler
CUDA_EXTRA_FLAGS="-allow-unsupported-compiler -O3 --use_fast_math"
CUDA_HOST_ARGS=()
if [[ -x /opt/cuda/bin/g++ ]]; then
    CUDA_HOST_ARGS+=("-DCMAKE_CUDA_HOST_COMPILER=/opt/cuda/bin/g++")
fi

cmake -G Ninja \
    -S "${STRATA_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_PORTABLE=OFF \
    -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
    -DCMAKE_CUDA_FLAGS="${CUDA_EXTRA_FLAGS}" \
    "${CUDA_HOST_ARGS[@]}" \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
    -DSTRATA_GGML_DIR="${LLAMA_DIR}"

JOBS="$(nproc || echo 16)"
if ! cmake --build "${BUILD_DIR}" --target strata -j "${JOBS}"; then
    echo "      (Retrying build once in case ptxas had a transient race...)"
    cmake --build "${BUILD_DIR}" --target strata -j "${JOBS}"
fi

# ------------------------------------------------------------------------------
# [4/5] Verify Native AVX-VNNI Instructions & Install into engine/strata
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Verifying AVX-VNNI (vpdpbusd / vpdpwssd) instructions in compiled binary..."
VNNI_COUNT="$(objdump -d "${BUILD_DIR}/strata" | grep -cE 'vpdpbusd|vpdpwssd' || true)"
echo "      Found ${VNNI_COUNT} native AVX-VNNI vector dot-product instructions in strata!"

mkdir -p "${STRATA_DIR}/engine"
if [[ -f "${STRATA_DIR}/engine/strata" && ! -f "${STRATA_DIR}/engine/strata.portable.bak" ]]; then
    cp -f "${STRATA_DIR}/engine/strata" "${STRATA_DIR}/engine/strata.portable.bak"
    echo "      Backed up portable binary -> ${STRATA_DIR}/engine/strata.portable.bak"
fi

cp -f "${BUILD_DIR}/strata" "${STRATA_DIR}/engine/strata"
chmod +x "${STRATA_DIR}/engine/strata"
echo "      Installed native binary -> ${STRATA_DIR}/engine/strata"

# Stamp BUILD.json so setup.py recognizes the custom local build and never overwrites it
"${PY_BIN}" - <<EOF
import json, pathlib, sys
sys.path.insert(0, "${STRATA_DIR}")
import setup
eng = pathlib.Path("${STRATA_DIR}/engine")
stamp = eng / "BUILD.json"
meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
cuda_Bin = pathlib.Path("${NVCC_BIN}").parent
dirs = [str(d) for d in (cuda_Bin, cuda_Bin / "x64", cuda_Bin.parent / "lib64", cuda_Bin.parent / "lib") if d.is_dir()]
meta.update({
    "source": "local",
    "version": setup.source_version(),
    "archs": [86],
    "cuda_dirs": dirs,
    "src": setup.source_hash(setup.ENGINE_SOURCES),
})
stamp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("      Updated engine/BUILD.json stamp (source=local, archs=[86], version=" + meta["version"] + ")")
EOF

# ------------------------------------------------------------------------------
# [5/5] Ready to Launch with Turbo Config
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo " ✅ NATIVE ARCH LINUX STRATA BUILD COMPLETE!"
echo "    - Binary : ${STRATA_DIR}/engine/strata (${VNNI_COUNT} AVX-VNNI instructions)"
echo "    - Config : ${STRATA_DIR}/strata-huihui-swift-linux.json"
echo ""
echo "    To start Strata immediately with unlimited pinned memory & active OpenMP spin:"
echo "      ulimit -l unlimited 2>/dev/null || true"
echo "      export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE CUDA_MODULE_LOADING=LAZY CUDA_DEVICE_MAX_CONNECTIONS=32"
echo "      cd ${STRATA_DIR} && ./.venv/bin/python serve/server.py --engine strata --config strata-huihui-swift-linux.json --port 8080"
echo "================================================================================"

if [[ "${1:-}" == "--run" ]]; then
    ulimit -l unlimited 2>/dev/null || true
    export OMP_WAIT_POLICY=ACTIVE
    export GOMP_SPINCOUNT=INFINITE
    export CUDA_MODULE_LOADING=LAZY
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    cd "${STRATA_DIR}"
    exec "${PY_BIN}" "${STRATA_DIR}/serve/server.py" \
        --engine strata \
        --config "${STRATA_DIR}/strata-huihui-swift-linux.json" \
        --port 8080
fi
