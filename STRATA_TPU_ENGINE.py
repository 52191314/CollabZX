# ==============================================================================
# PASTE THIS ENTIRE SCRIPT INTO A GOOGLE COLAB CELL (Runtime -> TPU v5e-1)
# ==============================================================================
# STRATA-TPU CLOUD ENGINE v1.0
# Model: Huihui-Qwen3.8-Flash-Next-abliterated (Swift 1.5 - IQ3_XXS, 125B MoE)
# Target Hardware: Google Colab Cloud TPU v5e-1 (16 GB HBM2e + 47.05 GB Host RAM)
# Storage Backing: 5 TB Google Drive (/content/drive/MyDrive/StrataModels/huihui-swift-IQ3_XXS)
# ==============================================================================

import os
import sys
import time
import json
import math
import mmap
import struct
import ctypes
import pathlib
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

print("=" * 80)
print(" 🚀 INITIALIZING STRATA-TPU CLOUD ENGINE (QWEN3.8-FLASH-NEXT 125B MoE)")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. OS & LINUX VM KERNEL TUNING + DEPENDENCY INSTALLATION
# ------------------------------------------------------------------------------
def setup_cloud_machine():
    print("\n[1/6] Customizing Colab Linux Kernel & VM Subsystem for 76 GB Zero-Copy Mmap...")
    cmds = [
        "echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true",
        "echo madvise > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true",
        "sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true",
        "sysctl -w vm.swappiness=10 >/dev/null 2>&1 || true",
        "sysctl -w vm.vfs_cache_pressure=50 >/dev/null 2>&1 || true",
    ]
    for c in cmds:
        subprocess.run(c, shell=True)

    print("      Installing Python packages (gguf, regex, fastapi, uvicorn, jax[tpu])...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "gguf", "regex", "jinja2", "fastapi", "uvicorn", "pydantic", "nest_asyncio"],
        check=True
    )

    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("      Installing cloudflared tunnel binary...")
        subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=True
        )

setup_cloud_machine()

# Mount Google Drive if running inside Colab
if not os.path.exists("/content/drive/MyDrive"):
    try:
        from google.colab import drive
        print("      Mounting 5 TB Google Drive at /content/drive...")
        drive.mount("/content/drive")
    except ImportError:
        pass

import numpy as np
import regex
import jax
import jax.numpy as jnp
from gguf import GGMLQuantizationType, quants as gguf_quants

print(f"      JAX Backend: {jax.default_backend().upper()} | Devices: {jax.devices()}")

# ------------------------------------------------------------------------------
# 2. COMPILE INLINE C++/OPENMP ACCELERATOR (libstrata_tpu.so)
# ------------------------------------------------------------------------------
# Implements Strata's exact dequantizers (including custom Type 42 Q2_0),
# fast OpenMP-parallel INT8 per-row quantization for TPU v5e MXU, and
# Layer 1 PLE 64-bit XOR N-gram hashing + IQ4_NL 16-row table gather.
# ------------------------------------------------------------------------------
CPP_SOURCE = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <omp.h>

extern "C" {

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000u | (man << 13);
    else f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t)(i >> 16);
}

static inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Custom GGUF Type 42 (Q2_0): 64 elements per 18-byte block
void strata_dequant_q2_0(const uint8_t* src, float* dst, int64_t n_elems) {
    const int64_t n_blocks = n_elems / 64;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = src + b * 18;
        const float d = fp16_to_fp32(read_u16(block));
        const uint8_t* qs = block + 2;
        float* out = dst + b * 64;
        for (int j = 0; j < 64; ++j) {
            const int code = (qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
            out[j] = (float)(code - 1) * d;
        }
    }
}

// Type 20 (IQ4_NL): 32 elements per 18-byte block (split-half nibbles)
void strata_dequant_iq4_nl(const uint8_t* src, float* dst, int64_t n_elems) {
    const int64_t n_blocks = n_elems / 32;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = src + b * 18;
        const float d = fp16_to_fp32(read_u16(block));
        const uint8_t* qs = block + 2;
        float* out = dst + b * 32;
        for (int j = 0; j < 16; ++j) {
            out[j]      = d * (float)kvalues_iq4nl[qs[j] & 0x0F];
            out[j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

// Type 23 (IQ4_XS): 256 elements per 136-byte super-block
void strata_dequant_iq4_xs(const uint8_t* src, float* dst, int64_t n_elems) {
    const int64_t n_blocks = n_elems / 256;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = src + b * 136;
        const float d = fp16_to_fp32(read_u16(block));
        const uint16_t scales_h = read_u16(block + 2);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;
        float* y = dst + b * 256;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0]  = dl * (float)kvalues_iq4nl[qs[j] & 0x0F];
                y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

// Type 8 (Q8_0): 32 elements per 34-byte block
void strata_dequant_q8_0(const uint8_t* src, float* dst, int64_t n_elems) {
    const int64_t n_blocks = n_elems / 32;
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = src + b * 34;
        const float d = fp16_to_fp32(read_u16(block));
        const int8_t* qs = (const int8_t*)(block + 2);
        float* out = dst + b * 32;
        for (int j = 0; j < 32; ++j) out[j] = (float)qs[j] * d;
    }
}

// Fast OpenMP Per-Row Symmetric INT8 Quantizer for TPU v5e HBM
// Converts float32 [rows, cols] -> int8 [rows, cols] + bf16_as_u16 scale [rows]
void strata_f32_to_int8_rowwise(const float* src, int8_t* dst_q, uint16_t* dst_scale_bf16,
                                int64_t rows, int64_t cols) {
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_in = src + r * cols;
        int8_t* row_out = dst_q + r * cols;
        float amax = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float v = std::fabs(row_in[c]);
            if (v > amax) amax = v;
        }
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        uint16_t s_bf16 = fp32_to_bf16(scale);
        dst_scale_bf16[r] = s_bf16;
        // Reconstruct exact bf16 scale so quantization matches TPU dequantization
        uint32_t s_u32 = ((uint32_t)s_bf16) << 16;
        float exact_scale;
        std::memcpy(&exact_scale, &s_u32, 4);
        float inv_scale = (exact_scale > 0.0f) ? (1.0f / exact_scale) : 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            int q = (int)std::roundf(row_in[c] * inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            row_out[c] = (int8_t)q;
        }
    }
}

static uint8_t g_iq2xxs[256 * 8];
static uint8_t g_iq2xs[512 * 8];
static uint8_t g_iq2s[1024 * 8];
static uint8_t g_iq3xxs[256 * 4];
static uint8_t g_iq3s[512 * 4];
static uint8_t g_ksigns[128];

void strata_init_iq_grids(const uint8_t* p_iq2xxs, const uint8_t* p_iq2xs,
                          const uint8_t* p_iq2s, const uint8_t* p_iq3xxs,
                          const uint8_t* p_iq3s, const uint8_t* p_ksigns) {
    std::memcpy(g_iq2xxs, p_iq2xxs, 256 * 8);
    std::memcpy(g_iq2xs,  p_iq2xs,  512 * 8);
    std::memcpy(g_iq2s,   p_iq2s,   1024 * 8);
    std::memcpy(g_iq3xxs, p_iq3xxs, 256 * 4);
    std::memcpy(g_iq3s,   p_iq3s,   512 * 4);
    std::memcpy(g_ksigns, p_ksigns, 128);
}

static inline float strata_dot_row_raw(const uint8_t* row_ptr, int type_id,
                                       const float* x, int64_t n_cols) {
    float row_acc = 0.0f;
    if (type_id == 20) { // IQ4_NL (32 elems per 18B block)
        const int64_t nb = n_cols / 32;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 18;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const float* xb = x + b * 32;
            float s = 0.0f;
            for (int j = 0; j < 16; ++j) {
                s += (float)kvalues_iq4nl[qs[j] & 0x0F] * xb[j]
                   + (float)kvalues_iq4nl[qs[j] >> 4]   * xb[j + 16];
            }
            row_acc += d * s;
        }
    } else if (type_id == 42) { // Q2_0 (64 elems per 18B block)
        const int64_t nb = n_cols / 64;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 18;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const float* xb = x + b * 64;
            float s = 0.0f;
            for (int j = 0; j < 16; ++j) {
                uint8_t q = qs[j];
                s += (float)((int)(q & 3) - 1)        * xb[j * 4 + 0]
                   + (float)((int)((q >> 2) & 3) - 1) * xb[j * 4 + 1]
                   + (float)((int)((q >> 4) & 3) - 1) * xb[j * 4 + 2]
                   + (float)((int)((q >> 6) & 3) - 1) * xb[j * 4 + 3];
            }
            row_acc += d * s;
        }
    } else if (type_id == 18) { // IQ3_XXS (256 elems per 98B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 98;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales = blk + 66;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t sw;
                std::memcpy(&sw, scales + ib * 4, 4);
                const float db = d * (0.5f + (float)(sw >> 28)) * 0.5f;
                const float* x_sub = xb + ib * 32;
                float sum32 = 0.0f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sbits = g_ksigns[(sw >> (7 * l)) & 0x7F];
                    const uint8_t* g4a = g_iq3xxs + (uint32_t)qs[ib * 8 + 2 * l + 0] * 4;
                    const uint8_t* g4b = g_iq3xxs + (uint32_t)qs[ib * 8 + 2 * l + 1] * 4;
                    for (int k = 0; k < 4; ++k) {
                        float sa = (sbits & (1u << k)) ? -1.0f : 1.0f;
                        float sb = (sbits & (1u << (k + 4))) ? -1.0f : 1.0f;
                        sum32 += (float)g4a[k] * sa * x_sub[l * 8 + k]
                               + (float)g4b[k] * sb * x_sub[l * 8 + k + 4];
                    }
                }
                row_acc += db * sum32;
            }
        }
    } else if (type_id == 21) { // IQ3_S (256 elems per 110B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 110;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* qh = blk + 66;
            const uint8_t* signs = blk + 74;
            const uint8_t* scales = blk + 106;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                uint8_t sc = (ib & 1) ? (scales[ib >> 1] >> 4) : (scales[ib >> 1] & 0x0F);
                const float db = d * (float)(1 + 2 * (int)sc);
                const uint8_t qhb = qh[ib];
                const float* x_sub = xb + ib * 32;
                float sum32 = 0.0f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sbits = signs[ib * 4 + l];
                    const uint16_t idxa = (uint16_t)qs[ib * 8 + 2 * l + 0] | (((uint16_t)((qhb >> (2 * l + 0)) & 1u)) << 8);
                    const uint16_t idxb = (uint16_t)qs[ib * 8 + 2 * l + 1] | (((uint16_t)((qhb >> (2 * l + 1)) & 1u)) << 8);
                    const uint8_t* g4a = g_iq3s + (uint32_t)idxa * 4;
                    const uint8_t* g4b = g_iq3s + (uint32_t)idxb * 4;
                    for (int k = 0; k < 4; ++k) {
                        float sa = (sbits & (1u << k)) ? -1.0f : 1.0f;
                        float sb = (sbits & (1u << (k + 4))) ? -1.0f : 1.0f;
                        sum32 += (float)g4a[k] * sa * x_sub[l * 8 + k]
                               + (float)g4b[k] * sb * x_sub[l * 8 + k + 4];
                    }
                }
                row_acc += db * sum32;
            }
        }
    } else if (type_id == 16) { // IQ2_XXS (256 elems per 66B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 66;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = qs + ib * 8;
                uint32_t w1;
                std::memcpy(&w1, q + 4, 4);
                const float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
                const float* x_sub = xb + ib * 32;
                float sum32 = 0.0f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g8 = g_iq2xxs + (uint32_t)q[l] * 8;
                    const uint8_t sbits = g_ksigns[(w1 >> (7 * l)) & 0x7F];
                    for (int k = 0; k < 8; ++k) {
                        float s = (sbits & (1u << k)) ? -1.0f : 1.0f;
                        sum32 += (float)g8[k] * s * x_sub[l * 8 + k];
                    }
                }
                row_acc += db * sum32;
            }
        }
    } else if (type_id == 17) { // IQ2_XS (256 elems per 74B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 74;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* scales = blk + 66;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 16; ++ib) {
                uint8_t sc = (ib & 1) ? (scales[ib >> 1] >> 4) : (scales[ib >> 1] & 0x0F);
                const float db = d * (0.5f + (float)sc) * 0.25f;
                const float* x_sub = xb + ib * 16;
                float sum16 = 0.0f;
                for (int l = 0; l < 2; ++l) {
                    uint16_t qw = read_u16(qs + (ib * 2 + l) * 2);
                    const uint8_t* g8 = g_iq2xs + (uint32_t)(qw & 511u) * 8;
                    const uint8_t sbits = g_ksigns[qw >> 9];
                    for (int k = 0; k < 8; ++k) {
                        float s = (sbits & (1u << k)) ? -1.0f : 1.0f;
                        sum16 += (float)g8[k] * s * x_sub[l * 8 + k];
                    }
                }
                row_acc += db * sum16;
            }
        }
    } else if (type_id == 22) { // IQ2_S (256 elems per 82B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 82;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint8_t* qs = blk + 2;
            const uint8_t* signs = blk + 34;
            const uint8_t* qh = blk + 66;
            const uint8_t* scales = blk + 74;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 16; ++ib) {
                uint8_t sc = (ib & 1) ? (scales[ib >> 1] >> 4) : (scales[ib >> 1] & 0x0F);
                const float db = d * (0.5f + (float)sc) * 0.25f;
                const float* x_sub = xb + ib * 16;
                float sum16 = 0.0f;
                for (int l = 0; l < 2; ++l) {
                    int idx32 = ib * 2 + l;
                    uint16_t gidx = (uint16_t)qs[idx32] | ((uint16_t)((qh[idx32 >> 2] >> (2 * (idx32 & 3))) & 0x03) << 8);
                    const uint8_t* g8 = g_iq2s + (uint32_t)gidx * 8;
                    const uint8_t sbits = signs[idx32];
                    for (int k = 0; k < 8; ++k) {
                        float s = (sbits & (1u << k)) ? -1.0f : 1.0f;
                        sum16 += (float)g8[k] * s * x_sub[l * 8 + k];
                    }
                }
                row_acc += db * sum16;
            }
        }
    } else if (type_id == 23) { // IQ4_XS (256 elems per 136B block)
        const int64_t nb = n_cols / 256;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 136;
            const float d = fp16_to_fp32(read_u16(blk));
            const uint16_t scales_h = read_u16(blk + 2);
            const uint8_t* scales_l = blk + 4;
            const uint8_t* qs = blk + 8;
            const float* xb = x + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d * (float)(ls - 32);
                const float* x_sub = xb + ib * 32;
                const uint8_t* q_sub = qs + ib * 16;
                float sum32 = 0.0f;
                for (int j = 0; j < 16; ++j) {
                    sum32 += (float)kvalues_iq4nl[q_sub[j] & 0x0F] * x_sub[j]
                           + (float)kvalues_iq4nl[q_sub[j] >> 4]   * x_sub[j + 16];
                }
                row_acc += dl * sum32;
            }
        }
    } else if (type_id == 8) { // Q8_0 (32 elems per 34B block)
        const int64_t nb = n_cols / 32;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row_ptr + b * 34;
            const float d = fp16_to_fp32(read_u16(blk));
            const int8_t* qs = (const int8_t*)(blk + 2);
            const float* xb = x + b * 32;
            float s = 0.0f;
            for (int j = 0; j < 32; ++j) s += (float)qs[j] * xb[j];
            row_acc += d * s;
        }
    }
    return row_acc;
}

// Zero-allocation OpenMP Direct Compressed Expert Evaluator
void strata_eval_expert_raw(const float* x,
                            const uint8_t* g_raw, int64_t g_bytes, int g_type,
                            const uint8_t* u_raw, int64_t u_bytes, int u_type,
                            const uint8_t* d_raw, int64_t d_bytes, int d_type,
                            float weight, float* out) {
    const int64_t g_row_bytes = g_bytes / 640;
    const int64_t u_row_bytes = u_bytes / 640;
    const int64_t d_row_bytes = d_bytes / 2560;
    float hidden[640];
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < 640; ++r) {
        float g_acc = strata_dot_row_raw(g_raw + r * g_row_bytes, g_type, x, 2560);
        float u_acc = strata_dot_row_raw(u_raw + r * u_row_bytes, u_type, x, 2560);
        float silu_g = g_acc / (1.0f + std::exp(-g_acc));
        hidden[r] = silu_g * u_acc;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < 2560; ++r) {
        float d_acc = strata_dot_row_raw(d_raw + r * d_row_bytes, d_type, hidden, 640);
        out[r] += weight * d_acc;
    }
}


// Layer 1 Per-Layer Token Embedding (PLE) 64-bit XOR N-gram Hash & IQ4_NL Gather
// Exact transcription of Strata's src/kernels/ngram.cpp
void strata_ple_ngram_gather(int32_t token, int32_t prev0_oldest, int32_t prev1_newest,
                             const uint8_t* table_base, int table_type, float* out2560) {
    static const uint64_t mult[3] = {
        23703573157769ull, 20109073645365ull, 8052911324071ull
    };
    static const uint64_t vocab[16] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
    };
    static const uint64_t offset[16] = {
        0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275
    };
    const int32_t EOS_ID = 248044;
    int32_t prev[2] = {prev0_oldest, prev1_newest};
    int64_t ctx[3];
    ctx[0] = token;
    bool cut = false;
    for (int s = 1; s < 3; ++s) {
        int32_t t = cut ? -1 : prev[2 - s];
        cut = cut || (t < 0) || (t == EOS_ID);
        ctx[s] = cut ? EOS_ID : t;
    }
    uint32_t rows[16];
    for (int n = 2; n <= 3; ++n) {
        uint64_t mixed = (uint64_t)ctx[0] * mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctx[j] * mult[j];
        int base = (n - 2) * 8;
        for (int g = 0; g < 8; ++g) {
            int h = base + g;
            rows[h] = (uint32_t)(mixed % vocab[h] + offset[h]);
        }
    }

    // Gather 16 rows of 160 elements = 2560 floats
    // In IQ4_NL (type 20), each 160-element row is 5 blocks of 32 = 90 bytes
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 16; ++h) {
        float* head_out = out2560 + h * 160;
        if (table_type == 20) {
            const uint8_t* row_ptr = table_base + (uint64_t)rows[h] * 90ull;
            for (int b = 0; b < 5; ++b) {
                const uint8_t* blk = row_ptr + b * 18;
                const float d = fp16_to_fp32(read_u16(blk));
                const uint8_t* qs = blk + 2;
                for (int j = 0; j < 16; ++j) {
                    head_out[b * 32 + j]      = d * (float)kvalues_iq4nl[qs[j] & 0x0F];
                    head_out[b * 32 + j + 16] = d * (float)kvalues_iq4nl[qs[j] >> 4];
                }
            }
        } else if (table_type == 8) {
            // Q8_0 (170 bytes per 160-elem row)
            const uint8_t* row_ptr = table_base + (uint64_t)rows[h] * 170ull;
            for (int b = 0; b < 5; ++b) {
                const uint8_t* blk = row_ptr + b * 34;
                const float d = fp16_to_fp32(read_u16(blk));
                const int8_t* qs = (const int8_t*)(blk + 2);
                for (int j = 0; j < 32; ++j) head_out[b * 32 + j] = d * (float)qs[j];
            }
        }
    }
}

} // extern "C"
"""

print("\n[2/6] Compiling Strata C++/OpenMP Bridge (/tmp/libstrata_tpu.so)...")
with open("/tmp/strata_tpu_bridge.cpp", "w") as f:
    f.write(CPP_SOURCE)
subprocess.run(
    ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
     "/tmp/strata_tpu_bridge.cpp", "-o", "/tmp/libstrata_tpu.so"],
    check=True
)
LIB_STRATA = ctypes.CDLL("/tmp/libstrata_tpu.so")

LIB_STRATA.strata_dequant_q2_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB_STRATA.strata_dequant_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB_STRATA.strata_dequant_iq4_xs.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB_STRATA.strata_dequant_q8_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB_STRATA.strata_f32_to_int8_rowwise.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
]
LIB_STRATA.strata_init_iq_grids.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
LIB_STRATA.strata_eval_expert_raw.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
    ctypes.c_float, ctypes.c_void_p
]
LIB_STRATA.strata_ple_ngram_gather.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
]

for _cls in (gguf_quants.IQ2_XXS, gguf_quants.IQ2_XS, gguf_quants.IQ2_S, gguf_quants.IQ3_XXS, gguf_quants.IQ3_S):
    _cls.init_grid()
_g_iq2xxs = np.ascontiguousarray(gguf_quants.IQ2_XXS.grid.astype(np.uint8).reshape(-1))
_g_iq2xs  = np.ascontiguousarray(gguf_quants.IQ2_XS.grid.astype(np.uint8).reshape(-1))
_g_iq2s   = np.ascontiguousarray(gguf_quants.IQ2_S.grid.astype(np.uint8).reshape(-1))
_g_iq3xxs = np.ascontiguousarray(gguf_quants.IQ3_XXS.grid.astype(np.uint8).reshape(-1))
_g_iq3s   = np.ascontiguousarray(gguf_quants.IQ3_S.grid.astype(np.uint8).reshape(-1))
_g_ksigns = np.ascontiguousarray(np.frombuffer(gguf_quants.IQ2_XXS.ksigns, dtype=np.uint8))
LIB_STRATA.strata_init_iq_grids(
    _g_iq2xxs.ctypes.data_as(ctypes.c_void_p),
    _g_iq2xs.ctypes.data_as(ctypes.c_void_p),
    _g_iq2s.ctypes.data_as(ctypes.c_void_p),
    _g_iq3xxs.ctypes.data_as(ctypes.c_void_p),
    _g_iq3s.ctypes.data_as(ctypes.c_void_p),
    _g_ksigns.ctypes.data_as(ctypes.c_void_p),
)
print("      Compiled libstrata_tpu.so with OpenMP + Direct IQ2/IQ3/IQ4/Q2_0 Expert Kernels!")


# ------------------------------------------------------------------------------
# 3. STRATA GGUF v3 READER (WITH TYPE 42 Q2_0) & QWEN35 TOKENIZER
# ------------------------------------------------------------------------------
GGUF_META = {
    0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2),
    4: ("u32", 4), 5: ("i32", 4), 6: ("f32", 4), 7: ("bool", 1),
    8: ("string", 0), 9: ("array", 0), 10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8),
}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 42: "Q2_0",
}

GGML_BLOCK_INFO = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 26: (1, 4), 28: (1, 8), 29: (256, 56),
    30: (1, 2), 42: (64, 18),
}

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    type_id: int
    type_name: str
    offset: int

    @property
    def elements(self) -> int:
        p = 1
        for d in self.shape:
            p *= d
        return p

    def expected_bytes(self) -> int:
        blk, bsz = GGML_BLOCK_INFO[self.type_id]
        return (self.elements // blk) * bsz

class GGUFFile:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self.alignment = 32
        self._data_start = 0
        with self.path.open("rb") as fh:
            self._parse(fh)

    def _parse(self, fh):
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{self.path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", fh.read(20))
        for _ in range(n_kv):
            key = self._str(fh)
            self.metadata[key] = self._value(fh)
        for _ in range(n_tensors):
            name = self._str(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            shape = list(struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims)))
            type_id, offset = struct.unpack("<IQ", fh.read(12))
            self.tensors.append(
                TensorInfo(name, shape, type_id, GGML_TYPES.get(type_id, f"type{type_id}"), offset)
            )
        align = self.metadata.get("general.alignment")
        if isinstance(align, int) and align:
            self.alignment = align
        pos = fh.tell()
        self._data_start = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _str(self, fh) -> str:
        (n,) = struct.unpack("<Q", fh.read(8))
        return fh.read(n).decode("utf-8", "replace")

    def _value(self, fh):
        (t,) = struct.unpack("<I", fh.read(4))
        name, size = GGUF_META[t]
        if name == "string":
            return self._str(fh)
        if name == "array":
            (et,) = struct.unpack("<I", fh.read(4))
            (count,) = struct.unpack("<Q", fh.read(8))
            ename, esize = GGUF_META[et]
            if ename == "string":
                return [self._str(fh) for _ in range(count)]
            fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
                   "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[ename]
            raw = fh.read(esize * count)
            return list(struct.unpack(f"<{count}{fmt}", raw))
        fmt = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
               "f32": "f", "bool": "?", "u64": "Q", "i64": "q", "f64": "d"}[name]
        return struct.unpack(f"<{fmt}", fh.read(size))[0]

# Exact Strata Qwen35 Byte-Level BPE Tokenizer (transcribed from tools/strata_tokenizer.py)
def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

class StrataTokenizer:
    def __init__(self, md: Dict[str, Any]):
        self.tokens: List[str] = list(md["tokenizer.ggml.tokens"])
        merges: List[str] = list(md["tokenizer.ggml.merges"])
        self.token_types: Optional[List[int]] = list(md.get("tokenizer.ggml.token_type") or []) or None
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {}
        for i, m in enumerate(merges):
            p0, p1 = m.split(" ")
            self.ranks[(p0, p1)] = i
        self._re = regex.compile(QWEN35_PATTERN)
        self.special_tokens = {}
        if self.token_types:
            for i, ty in enumerate(self.token_types):
                if ty in (3, 4):
                    self.special_tokens[self.tokens[i]] = i
        self._special_re = regex.compile(
            "|".join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
        ) if self.special_tokens else None
        self.eos_id = int(md.get("tokenizer.ggml.eos_token_id", 248044))
        self.im_end_id = self.ids.get("<|im_end|>", 248046)

    def _bpe(self, word: str) -> List[str]:
        parts = list(word)
        ranks_get = self.ranks.get
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks_get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def _encode_plain(self, text: str) -> List[int]:
        out = []
        for piece in self._re.findall(text):
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                out.append(self.ids[tok])
        return out

    def encode(self, text: str, parse_special: bool = True) -> List[int]:
        if not parse_special or self._special_re is None:
            return self._encode_plain(text)
        out = []
        pos = 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def decode(self, ids: List[int]) -> str:
        raw = bytearray()
        for i in ids:
            if 0 <= i < len(self.tokens):
                if i in self.special_tokens.values():
                    continue
                for ch in self.tokens[i]:
                    v = UNICODE_TO_BYTE.get(ch)
                    if v is not None:
                        raw.append(v)
        return raw.decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if enable_thinking:
            parts.append("<|im_start|>assistant\n<think>\n")
        else:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)

# ------------------------------------------------------------------------------
# 4. ZERO-COPY MULTI-SHARD MMAP & HYBRID TPU HBM / HOST RAM WEIGHT MANAGER
# ------------------------------------------------------------------------------
import shutil

DRIVE_MODEL_DIR = pathlib.Path(
    os.environ.get("STRATA_MODEL_DIR", "/content/drive/MyDrive/StrataModels/huihui-swift-IQ3_XXS")
)
LOCAL_MODEL_DIR = pathlib.Path(os.environ.get("STRATA_LOCAL_DIR", "/content/local_strata"))

def stage_shard_to_local_ssd(src_path: pathlib.Path, local_dir: pathlib.Path) -> pathlib.Path:
    """Stages a GGUF shard from Google Drive FUSE onto Colab's local NVMe SSD to eliminate 850ms network seek latency."""
    if not str(src_path).startswith("/content/drive") or os.environ.get("STRATA_DISABLE_LOCAL_STAGE", "0") == "1":
        return src_path
    local_dir.mkdir(parents=True, exist_ok=True)
    dst_path = local_dir / src_path.name
    src_size = src_path.stat().st_size
    if dst_path.exists() and dst_path.stat().st_size == src_size:
        print(f"      [Local SSD] Already staged on NVMe: {dst_path} ({src_size / 1e9:.2f} GB)")
        return dst_path
    free_bytes = shutil.disk_usage(str(local_dir)).free
    # Keep at least 4 GB headroom on Colab root disk
    if free_bytes > src_size + (4 << 30):
        print(f"      [Local SSD] Staging {src_path.name} ({src_size / 1e9:.2f} GB) from Google Drive -> {dst_path} (Free: {free_bytes / 1e9:.1f} GB)...")
        t_cp = time.time()
        subprocess.run(["cp", "-f", str(src_path), str(dst_path)], check=True)
        dt_cp = max(time.time() - t_cp, 1e-3)
        print(f"      [Local SSD] Staged {dst_path.name} in {dt_cp:.1f}s ({src_size / 1e6 / dt_cp:.1f} MB/s)!")
        return dst_path
    else:
        print(f"      [Local SSD] Insufficient disk for {src_path.name} (Need {src_size / 1e9:.1f} GB, Free {free_bytes / 1e9:.1f} GB); using Drive + 24 GB Host RAM Cache.")
        return src_path

SHARD1_PATH = stage_shard_to_local_ssd(
    DRIVE_MODEL_DIR / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf", LOCAL_MODEL_DIR
)
SHARD2_PATH = stage_shard_to_local_ssd(
    DRIVE_MODEL_DIR / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf", LOCAL_MODEL_DIR
)

def dequantize_buffer_to_f32(raw_u8: np.ndarray, type_id: int, n_elems: int) -> np.ndarray:
    """Fast C++/OpenMP or gguf-py dequantization of any GGUF tensor slice into float32."""
    if type_id == 0:  # F32
        return raw_u8.view(np.float32).copy()
    if type_id == 1:  # F16
        return raw_u8.view(np.float16).astype(np.float32)
    if type_id == 30:  # BF16
        return (raw_u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    out = np.empty(n_elems, dtype=np.float32)
    src_ptr = raw_u8.ctypes.data_as(ctypes.c_void_p)
    dst_ptr = out.ctypes.data_as(ctypes.c_void_p)
    if type_id == 42:  # Q2_0
        LIB_STRATA.strata_dequant_q2_0(src_ptr, dst_ptr, n_elems)
        return out
    if type_id == 20:  # IQ4_NL
        LIB_STRATA.strata_dequant_iq4_nl(src_ptr, dst_ptr, n_elems)
        return out
    if type_id == 23:  # IQ4_XS
        LIB_STRATA.strata_dequant_iq4_xs(src_ptr, dst_ptr, n_elems)
        return out
    if type_id == 8:   # Q8_0
        LIB_STRATA.strata_dequant_q8_0(src_ptr, dst_ptr, n_elems)
        return out
    # Standard gguf-py dequantizer for IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ1_M (29), K-quants
    return gguf_quants.dequantize(raw_u8, GGMLQuantizationType(type_id)).astype(np.float32).reshape(-1)

def f32_to_tpu_int8(arr_f32: np.ndarray, rows: int, cols: int) -> Tuple[jax.Array, jax.Array]:
    """Quantizes [rows, cols] float32 matrix into TPU HBM (int8 [rows, cols], bf16 scale [rows])."""
    arr_f32 = np.ascontiguousarray(arr_f32.reshape(rows, cols), dtype=np.float32)
    q_i8 = np.empty((rows, cols), dtype=np.int8)
    s_u16 = np.empty((rows,), dtype=np.uint16)
    LIB_STRATA.strata_f32_to_int8_rowwise(
        arr_f32.ctypes.data_as(ctypes.c_void_p),
        q_i8.ctypes.data_as(ctypes.c_void_p),
        s_u16.ctypes.data_as(ctypes.c_void_p),
        rows, cols
    )
    tpu_dev = jax.devices()[0]
    s_bf16 = (s_u16.astype(np.uint32) << 16).view(np.float32).astype(jnp.bfloat16)
    return jax.device_put(q_i8, tpu_dev), jax.device_put(s_bf16, tpu_dev)

def f32_to_tpu_bf16(arr_f32: np.ndarray, shape: Tuple[int, ...]) -> jax.Array:
    tpu_dev = jax.devices()[0]
    return jax.device_put(jnp.asarray(arr_f32.reshape(shape), dtype=jnp.bfloat16), tpu_dev)

def f32_to_tpu_f32(arr_f32: np.ndarray, shape: Tuple[int, ...]) -> jax.Array:
    tpu_dev = jax.devices()[0]
    return jax.device_put(jnp.asarray(arr_f32.reshape(shape), dtype=jnp.float32), tpu_dev)

# ------------------------------------------------------------------------------
# 5. EXACT STRATA TPU @jax.jit KERNELS (GR, PLE, GDN, QSA, MoE, LM_HEAD)
# ------------------------------------------------------------------------------
RMS_EPS = 1e-6

@jax.jit
def tpu_int8_matvec(w_q: jax.Array, w_s: jax.Array, x: jax.Array) -> jax.Array:
    """Fused TPU v5e MXU INT8 Matrix-Vector product: (w_q @ x) * w_s -> float32."""
    x_bf16 = x.astype(jnp.bfloat16)
    dots = jnp.dot(w_q, x_bf16, preferred_element_type=jnp.float32)
    return dots * w_s.astype(jnp.float32)

@jax.jit
def tpu_bf16_matvec(w_bf16: jax.Array, x: jax.Array) -> jax.Array:
    return jnp.dot(w_bf16, x.astype(jnp.bfloat16), preferred_element_type=jnp.float32)

@jax.jit
def gr_read_jit(R: jax.Array, w_norm: jax.Array, w_down: jax.Array,
                w_up: jax.Array, w_inject: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Strata Hyper-Connection Read (include/strata/kernels/gr.hpp):
      R: (4, 2560) float32
      w_norm: (4, 2560) float32
      w_down: (320, 10240) bfloat16
      w_up: (10240, 320) bfloat16
      w_inject: (4, 10240) bfloat16
    """
    ms = jnp.mean(jnp.square(R), axis=-1, keepdims=True)
    xn_streams = R * jax.lax.rsqrt(ms + RMS_EPS) * w_norm
    xn = xn_streams.reshape(-1)  # (10240,)
    xn_bf16 = xn.astype(jnp.bfloat16)
    lo_pre = jnp.dot(w_down, xn_bf16, preferred_element_type=jnp.float32) * 0.25
    lo = jax.nn.silu(lo_pre).astype(jnp.bfloat16)
    up_proj = jnp.dot(w_up, lo, preferred_element_type=jnp.float32)
    gated = (xn * jax.nn.sigmoid(up_proj)).reshape(4, 2560)
    mixed = jnp.mean(gated, axis=0)  # (2560,)
    inject = jnp.dot(w_inject, xn_bf16, preferred_element_type=jnp.float32)  # (4,)
    return mixed, inject

@jax.jit
def gr_write_jit(R: jax.Array, block_out: jax.Array, inject: jax.Array) -> jax.Array:
    """Strata Hyper-Connection Write: w = 2 * sigmoid(inject / 4); R + w[:, None] * block_out."""
    w = 2.0 * jax.nn.sigmoid(inject * 0.25)
    return R + w[:, None] * block_out[None, :]

@jax.jit
def ple_layer1_jit(emb: jax.Array, R: jax.Array, hist: jax.Array,
                   key_q: jax.Array, key_s: jax.Array,
                   val_bf16: jax.Array, norm_key: jax.Array,
                   norm_query: jax.Array, norm_conv: jax.Array,
                   conv1d_f32: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Strata Layer 1 Per-Layer Token Embedding Block (src/kernels/cuda/ple.cu):
      emb: (2560,) float32 gathered from 16 N-gram heads
      R: (4, 2560) float32 residual stack
      hist: (9, 10240) float32 normalized conv history
    """
    hidden = R.reshape(4, 2560)
    raw_key = tpu_int8_matvec(key_q, key_s, emb).reshape(4, 2560)
    # Per-stream grouped RMSNorm
    k_scale = jax.lax.rsqrt(jnp.mean(jnp.square(raw_key), axis=-1, keepdims=True) + RMS_EPS)
    key = raw_key * k_scale * norm_key.reshape(4, 2560)

    q_scale = jax.lax.rsqrt(jnp.mean(jnp.square(hidden), axis=-1, keepdims=True) + RMS_EPS)
    query = hidden * q_scale * norm_query.reshape(4, 2560)

    value = tpu_bf16_matvec(val_bf16, emb)  # (2560,)

    # Signed square root gate: s = sum(key * query) / sqrt(2560)
    s = jnp.sum(key * query, axis=-1) * (1.0 / math.sqrt(2560.0))
    mag = jnp.sqrt(jnp.maximum(jnp.abs(s), 1e-6))
    gate = jax.nn.sigmoid(jnp.sign(s) * mag)  # (4,)

    gated = gate[:, None] * value[None, :]  # (4, 2560)
    g_scale = jax.lax.rsqrt(jnp.mean(jnp.square(gated), axis=-1, keepdims=True) + RMS_EPS)
    normalized = (gated * g_scale * norm_conv.reshape(4, 2560)).reshape(10240)

    # Dilated causal depthwise conv (kern=4, dil=3, hist=9 -> taps at rows 0, 3, 6, and new normalized)
    # conv1d_f32 is shaped (4, 10240) where k=0..3
    conv_acc = (
        conv1d_f32[0] * hist[0] +
        conv1d_f32[1] * hist[3] +
        conv1d_f32[2] * hist[6] +
        conv1d_f32[3] * normalized
    )
    conv_out = jax.nn.silu(conv_acc).reshape(4, 2560)
    R_new = hidden + gated + conv_out
    hist_new = jnp.concatenate([hist[1:], normalized[None, :]], axis=0)
    return R_new, hist_new

@jax.jit
def gdn_layer_jit(mixed: jax.Array, conv_state: jax.Array, ssm_state: jax.Array,
                  qkv_q: jax.Array, qkv_s: jax.Array,
                  conv_w: jax.Array, alpha_w: jax.Array, beta_w: jax.Array,
                  ssm_a: jax.Array, ssm_dt: jax.Array,
                  gate_q: jax.Array, gate_s: jax.Array,
                  ssm_norm: jax.Array,
                  out_q: jax.Array, out_s: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Strata Gated DeltaNet Layer (src/kernels/cuda/gdn.cu):
      36 layers of recurrent 48-head x 128x128 fast-weight state update.
    """
    qkv = tpu_int8_matvec(qkv_q, qkv_s, mixed)  # (10240,)
    # 1D depthwise causal conv (d_conv=4): conv_w shape (4, 10240)
    conv_out = (
        conv_w[0] * conv_state[0] +
        conv_w[1] * conv_state[1] +
        conv_w[2] * conv_state[2] +
        conv_w[3] * qkv
    )
    new_conv_state = jnp.stack([conv_state[1], conv_state[2], qkv], axis=0)
    h = jax.nn.silu(conv_out)

    # Split into q (16, 128), k (16, 128), v (48, 128)
    q_raw = h[:2048].reshape(16, 128)
    k_raw = h[2048:4096].reshape(16, 128)
    v = h[4096:10240].reshape(48, 128)

    # L2-normalize q and k per 128-dim head; scale q by 1/sqrt(128)
    q = q_raw * jax.lax.rsqrt(jnp.sum(jnp.square(q_raw), axis=-1, keepdims=True) + RMS_EPS) * (1.0 / math.sqrt(128.0))
    k = k_raw * jax.lax.rsqrt(jnp.sum(jnp.square(k_raw), axis=-1, keepdims=True) + RMS_EPS)

    # Repeat q and k across the 48 value heads: head h uses h % 16 (3 groups of 16)
    q_rep = jnp.tile(q, (3, 1))  # (48, 128)
    k_rep = jnp.tile(k, (3, 1))  # (48, 128)

    alpha = tpu_bf16_matvec(alpha_w, mixed)  # (48,)
    beta = jax.nn.sigmoid(tpu_bf16_matvec(beta_w, mixed))  # (48,)
    gate = -jnp.exp(ssm_a) * jax.nn.softplus(alpha + ssm_dt)  # (48,)
    dec = jnp.exp(gate)  # (48,)

    # Vectorized Delta-Rule State Update across 48 heads: S shape (48, 128, 128)
    # In Strata gdn.cu: S[h, d, col] where sk[d] = sum_col S[h, d, col] * k[col] -> S @ k
    S_dec = ssm_state * dec[:, None, None]
    sk = jnp.einsum("hdc,hc->hd", S_dec, k_rep)
    d_vec = (v - sk) * beta[:, None]
    new_ssm_state = S_dec + jnp.einsum("hd,hc->hdc", d_vec, k_rep)
    o = jnp.einsum("hdc,hc->hd", new_ssm_state, q_rep)  # (48, 128)

    # Output norm + sigmoid(z) gate
    z = tpu_int8_matvec(gate_q, gate_s, mixed).reshape(48, 128)
    o_norm = o * jax.lax.rsqrt(jnp.mean(jnp.square(o), axis=-1, keepdims=True) + RMS_EPS)
    y = (o_norm * ssm_norm[None, :] * jax.nn.sigmoid(z)).reshape(6144)

    out = tpu_int8_matvec(out_q, out_s, y)  # (2560,)
    return out, new_conv_state, new_ssm_state

@jax.jit
def rope_neox_64(x: jax.Array, cos_row: jax.Array, sin_row: jax.Array) -> jax.Array:
    """Applies NeoX RoPE to the first 64 dims of x (..., D)."""
    x0 = x[..., :32]
    x1 = x[..., 32:64]
    rx0 = x0 * cos_row - x1 * sin_row
    rx1 = x0 * sin_row + x1 * cos_row
    return jnp.concatenate([rx0, rx1, x[..., 64:]], axis=-1)

@jax.jit
def qsa_layer_jit(mixed: jax.Array, pos: jax.Array,
                  cos_row: jax.Array, sin_row: jax.Array,
                  k_cache: jax.Array, v_cache: jax.Array,
                  q_q: jax.Array, q_s: jax.Array,
                  k_q: jax.Array, k_s: jax.Array,
                  v_q: jax.Array, v_s: jax.Array,
                  o_q: jax.Array, o_s: jax.Array,
                  q_norm: jax.Array, k_norm: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Strata QSA Full-Attention Layer (src/kernels/cuda/qsa.cu):
      n_head = 24, n_head_kv = 2, head_dim = 256, n_rot = 64.
      q_full projects 2560 -> 12288 (24 heads x [q(256), gate(256)]).
    """
    q_full = tpu_int8_matvec(q_q, q_s, mixed).reshape(24, 2, 256)
    qcur = q_full[:, 0, :]   # (24, 256)
    gate = q_full[:, 1, :]   # (24, 256)

    kcur = tpu_int8_matvec(k_q, k_s, mixed).reshape(2, 256)
    vcur = tpu_int8_matvec(v_q, v_s, mixed).reshape(2, 256)

    # Weighted RMSNorm + NeoX RoPE on first 64 dims
    qcur = qcur * jax.lax.rsqrt(jnp.mean(jnp.square(qcur), axis=-1, keepdims=True) + RMS_EPS) * q_norm[None, :]
    kcur = kcur * jax.lax.rsqrt(jnp.mean(jnp.square(kcur), axis=-1, keepdims=True) + RMS_EPS) * k_norm[None, :]

    qcur = rope_neox_64(qcur, cos_row, sin_row)
    kcur = rope_neox_64(kcur, cos_row, sin_row)

    # Update KV cache at position `pos` (k_cache, v_cache shape: [MAX_CTX, 2, 256] bfloat16)
    k_cache = jax.lax.dynamic_update_slice(k_cache, kcur[None, :, :].astype(jnp.bfloat16), (pos, 0, 0))
    v_cache = jax.lax.dynamic_update_slice(v_cache, vcur[None, :, :].astype(jnp.bfloat16), (pos, 0, 0))

    # GQA: 24 query heads grouped into 2 KV heads (12 query heads per KV head)
    q_grouped = qcur.reshape(2, 12, 256) * (1.0 / math.sqrt(256.0))
    scores = jnp.einsum("ghd,tgd->ght", q_grouped, k_cache.astype(jnp.float32))  # (2, 12, MAX_CTX)
    mask = jnp.arange(k_cache.shape[0])[None, None, :] <= pos
    scores = jnp.where(mask, scores, -1e30)
    probs = jax.nn.softmax(scores, axis=-1)
    attn_grouped = jnp.einsum("ght,tgd->ghd", probs, v_cache.astype(jnp.float32))  # (2, 12, 256)
    attn = attn_grouped.reshape(24, 256)

    # Sigmoid output gate + output projection
    gated_attn = (attn * jax.nn.sigmoid(gate)).reshape(6144)
    out = tpu_int8_matvec(o_q, o_s, gated_attn)
    return out, k_cache, v_cache

@jax.jit
def moe_route_and_shared_jit(mixed: jax.Array, w_router: jax.Array, w_ginp: jax.Array,
                             sgate_q: jax.Array, sgate_s: jax.Array,
                             sup_q: jax.Array, sup_s: jax.Array,
                             sdown_q: jax.Array, sdown_s: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Strata MoE Router (Top-10 of 512) + Gated Shared Expert (src/core/layer.cpp):
      Returns (top10_ids [10], top10_weights [10], shared_out [2560]).
    """
    logits = tpu_bf16_matvec(w_router, mixed)  # (512,)
    probs = jax.nn.softmax(logits)
    top_vals, top_ids = jax.lax.top_k(probs, 10)
    top_weights = top_vals / jnp.sum(top_vals)

    # Shared expert
    sh_gate = jax.nn.sigmoid(jnp.dot(w_ginp.astype(jnp.float32), mixed))
    g_proj = tpu_int8_matvec(sgate_q, sgate_s, mixed)
    u_proj = tpu_int8_matvec(sup_q, sup_s, mixed)
    sh_hidden = jax.nn.silu(g_proj) * u_proj
    sh_out = sh_gate * tpu_int8_matvec(sdown_q, sdown_s, sh_hidden)
    return top_ids, top_weights, sh_out

@jax.jit
def eval_cached_expert_tpu_jit(mixed: jax.Array,
                               gate_q: jax.Array, gate_s: jax.Array,
                               up_q: jax.Array, up_s: jax.Array,
                               down_q: jax.Array, down_s: jax.Array) -> jax.Array:
    """Evaluates one INT8-cached routed expert on TPU v5e MXU."""
    g = tpu_int8_matvec(gate_q, gate_s, mixed)
    u = tpu_int8_matvec(up_q, up_s, mixed)
    h = jax.nn.silu(g) * u
    return tpu_int8_matvec(down_q, down_s, h)

@jax.jit
def lm_head_jit(R: jax.Array, w_norm: jax.Array, w_down: jax.Array,
                w_up: jax.Array, out_q: jax.Array, out_s: jax.Array) -> jax.Array:
    """Strata Output Hyper-Connection Mix + 248,320-vocab Output Projection."""
    dummy_inj = jnp.zeros((4, 10240), dtype=jnp.bfloat16)
    mixed, _ = gr_read_jit(R, w_norm, w_down, w_up, dummy_inj)
    logits = tpu_int8_matvec(out_q, out_s, mixed)
    return logits

# ------------------------------------------------------------------------------
# 6. STRATA-TPU ENGINE: WEIGHT LOADER, HBM EXPERT CACHE & FORWARD LOOP
# ------------------------------------------------------------------------------
class StrataTPUEngine:
    def __init__(self, shard1_path: pathlib.Path, shard2_path: pathlib.Path,
                 max_ctx: int = 4096, hbm_expert_slots: int = 1024):
        self.max_ctx = max_ctx
        self.hbm_expert_slots = hbm_expert_slots
        print(f"\n[3/6] Parsing GGUF v3 Headers & Mmapping 76 GB Shards...")
        if not shard1_path.exists() or not shard2_path.exists():
            raise FileNotFoundError(
                f"Missing GGUF shards in {shard1_path.parent}!\n"
                f"Expected:\n  - {shard1_path.name}\n  - {shard2_path.name}"
            )
        self.g1 = GGUFFile(shard1_path)
        self.g2 = GGUFFile(shard2_path)
        self.mm1 = np.memmap(shard1_path, dtype=np.uint8, mode="r")
        self.mm2 = np.memmap(shard2_path, dtype=np.uint8, mode="r")

        self.where: Dict[str, Tuple[GGUFFile, TensorInfo, np.memmap]] = {}
        for g, mm in [(self.g1, self.mm1), (self.g2, self.mm2)]:
            for t in g.tensors:
                self.where[t.name] = (g, t, mm)
        print(f"      Mapped {len(self.where)} tensors across 2 shards (Shard 1: {len(self.g1.tensors)}, Shard 2: {len(self.g2.tensors)}).")

        print(f"\n[4/6] Initializing Strata Qwen35 Tokenizer...")
        self.tokenizer = StrataTokenizer(self.g1.metadata)
        print(f"      Vocab Size: {len(self.tokenizer.tokens):,} | EOS ID: {self.tokenizer.eos_id}")

        # Build RoPE table in float64 (Strata layer.cpp L834: freq_base = 1e7, n_rot = 64)
        inv_freq = 1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64.0))
        positions = np.arange(self.max_ctx, dtype=np.float64)[:, None]
        angles = positions * inv_freq[None, :]
        self.cos_tab = f32_to_tpu_f32(np.cos(angles).astype(np.float32), (self.max_ctx, 32))
        self.sin_tab = f32_to_tpu_f32(np.sin(angles).astype(np.float32), (self.max_ctx, 32))

        # Locate token_embd and per_layer_token_embd for zero-copy CPU gather
        self.emb_g, self.emb_t, self.emb_mm = self.where["token_embd.weight"]
        self.emb_row_bytes = self.emb_t.expected_bytes() // 248320

        self.ple_g, self.ple_t, self.ple_mm = self.where["per_layer_token_embd.weight"]
        self.ple_base_ptr = (
            self.ple_mm.ctypes.data + self.ple_g._data_start + self.ple_t.offset
        )

        # Load Dense Backbone into TPU v5e HBM
        self.layers: List[Dict[str, Any]] = []
        self.expert_info: List[Dict[str, Any]] = []
        self._load_dense_backbone_to_tpu()

        # Initialize Dynamic LRU Expert Cache in TPU HBM
        # Key: (layer, expert_id) -> (gate_q, gate_s, up_q, up_s, down_q, down_s) on TPU HBM
        self.expert_cache: OrderedDict[Tuple[int, int], Tuple[jax.Array, ...]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        # Allocate Recurrent Session State on TPU HBM
        self.reset_state()

    def _raw_slice(self, name: str) -> Tuple[np.ndarray, TensorInfo]:
        g, t, mm = self.where[name]
        start = g._data_start + t.offset
        end = start + t.expected_bytes()
        return mm[start:end], t

    def _load_f32(self, name: str) -> Tuple[np.ndarray, TensorInfo]:
        raw, t = self._raw_slice(name)
        return dequantize_buffer_to_f32(raw, t.type_id, t.elements), t

    def _load_tpu_int8(self, name: str, rows: int, cols: int) -> Tuple[jax.Array, jax.Array]:
        arr, _ = self._load_f32(name)
        return f32_to_tpu_int8(arr, rows, cols)

    def _load_tpu_bf16(self, name: str, shape: Tuple[int, ...]) -> jax.Array:
        arr, _ = self._load_f32(name)
        return f32_to_tpu_bf16(arr, shape)

    def _load_tpu_f32(self, name: str, shape: Tuple[int, ...]) -> jax.Array:
        arr, _ = self._load_f32(name)
        return f32_to_tpu_f32(arr, shape)

    def _load_dense_backbone_to_tpu(self):
        print("\n[5/6] Staging 48-Layer Dense Backbone into TPU v5e HBM2e (INT8 + BF16)...")
        t0 = time.time()

        # Layer 1 PLE weights
        print("      -> Loading Layer 1 PLE Projections (blk.1.ple_*)...")
        self.ple_weights = {
            "key": self._load_tpu_int8("blk.1.ple_key.weight", 10240, 2560),
            "val": self._load_tpu_bf16("blk.1.ple_value.weight", (2560, 2560)),
            "norm_key": self._load_tpu_f32("blk.1.ple_norm_key.weight", (10240,)),
            "norm_query": self._load_tpu_f32("blk.1.ple_norm_query.weight", (10240,)),
            "norm_conv": self._load_tpu_f32("blk.1.ple_norm_conv.weight", (10240,)),
        }
        # ple_conv1d has GGUF shape [4, 10240] with ne0=4 fastest -> reshape(10240, 4).T -> (4, 10240)
        ple_conv_f32, _ = self._load_f32("blk.1.ple_conv1d.weight")
        self.ple_weights["conv1d"] = f32_to_tpu_f32(
            ple_conv_f32.reshape(10240, 4).T.copy(), (4, 10240)
        )

        for l in range(48):
            p = f"blk.{l}."
            is_qsa = (l % 4 == 3)
            lw: Dict[str, Any] = {"is_qsa": is_qsa}

            # Hyper-connection weights (hc_attn_* and hc_ffn_*)
            for h_prefix in ("hc_attn_", "hc_ffn_"):
                lw[h_prefix + "norm"] = self._load_tpu_f32(p + h_prefix + "norm.weight", (4, 2560))
                lw[h_prefix + "down"] = self._load_tpu_bf16(p + h_prefix + "down.weight", (320, 10240))
                lw[h_prefix + "up"] = self._load_tpu_bf16(p + h_prefix + "up.weight", (10240, 320))
                lw[h_prefix + "inject"] = self._load_tpu_bf16(p + h_prefix + "inject.weight", (4, 10240))

            if is_qsa:
                lw["q"] = self._load_tpu_int8(p + "attn_q.weight", 12288, 2560)
                lw["k"] = self._load_tpu_int8(p + "attn_k.weight", 512, 2560)
                lw["v"] = self._load_tpu_int8(p + "attn_v.weight", 512, 2560)
                lw["o"] = self._load_tpu_int8(p + "attn_output.weight", 2560, 6144)
                lw["q_norm"] = self._load_tpu_f32(p + "attn_q_norm.weight", (256,))
                lw["k_norm"] = self._load_tpu_f32(p + "attn_k_norm.weight", (256,))
            else:
                lw["qkv"] = self._load_tpu_int8(p + "attn_qkv.weight", 10240, 2560)
                conv_f32, _ = self._load_f32(p + "ssm_conv1d.weight")
                lw["conv_w"] = f32_to_tpu_f32(conv_f32.reshape(10240, 4).T.copy(), (4, 10240))
                lw["alpha_w"] = self._load_tpu_bf16(p + "ssm_alpha.weight", (48, 2560))
                lw["beta_w"] = self._load_tpu_bf16(p + "ssm_beta.weight", (48, 2560))
                lw["ssm_a"] = self._load_tpu_f32(p + "ssm_a", (48,))
                lw["ssm_dt"] = self._load_tpu_f32(p + "ssm_dt.bias", (48,))
                lw["gate"] = self._load_tpu_int8(p + "attn_gate.weight", 6144, 2560)
                lw["ssm_norm"] = self._load_tpu_f32(p + "ssm_norm.weight", (128,))
                lw["out"] = self._load_tpu_int8(p + "ssm_out.weight", 2560, 6144)

            # MoE Router + Shared Expert
            lw["router"] = self._load_tpu_bf16(p + "ffn_gate_inp.weight", (512, 2560))
            lw["ginp"] = self._load_tpu_bf16(p + "ffn_gate_inp_shexp.weight", (2560,))
            lw["sgate"] = self._load_tpu_int8(p + "ffn_gate_shexp.weight", 640, 2560)
            lw["sup"] = self._load_tpu_int8(p + "ffn_up_shexp.weight", 640, 2560)
            lw["sdown"] = self._load_tpu_int8(p + "ffn_down_shexp.weight", 2560, 640)
            self.layers.append(lw)

            # Record zero-copy mmap pointers for the 512 routed experts of layer l
            g_g, g_t, g_mm = self.where[p + "ffn_gate_exps.weight"]
            u_g, u_t, u_mm = self.where[p + "ffn_up_exps.weight"]
            d_g, d_t, d_mm = self.where[p + "ffn_down_exps.weight"]
            self.expert_info.append({
                "gate": (g_mm, g_g._data_start + g_t.offset, g_t.expected_bytes() // 512, g_t.type_id),
                "up":   (u_mm, u_g._data_start + u_t.offset, u_t.expected_bytes() // 512, u_t.type_id),
                "down": (d_mm, d_g._data_start + d_t.offset, d_t.expected_bytes() // 512, d_t.type_id),
            })

            if (l + 1) % 8 == 0 or l == 47:
                elapsed = time.time() - t0
                print(f"      -> Loaded Layers 0..{l:02d}/47 into TPU HBM ({elapsed:.1f}s)")

        print("      -> Loading Output Hyper-Connection & 248,320-Vocab LM Head...")
        self.out_hc_norm = self._load_tpu_f32("output_hc_norm.weight", (4, 2560))
        self.out_hc_down = self._load_tpu_bf16("output_hc_down.weight", (320, 10240))
        self.out_hc_up = self._load_tpu_bf16("output_hc_up.weight", (10240, 320))
        self.out_weight = self._load_tpu_int8("output.weight", 248320, 2560)
        print(f"      Dense Backbone Ready in TPU v5e HBM in {time.time() - t0:.1f}s!")

    def reset_state(self):
        tpu_dev = jax.devices()[0]
        self.pos = 0
        self.prev_tokens = [-1, -1]  # [oldest, newest] for Layer 1 PLE N-gram hash
        self.ple_hist = jax.device_put(jnp.zeros((9, 10240), dtype=jnp.float32), tpu_dev)
        self.gdn_conv_states = {}
        self.gdn_ssm_states = {}
        self.qsa_k_caches = {}
        self.qsa_v_caches = {}
        for l in range(48):
            if l % 4 == 3:
                self.qsa_k_caches[l] = jax.device_put(
                    jnp.zeros((self.max_ctx, 2, 256), dtype=jnp.bfloat16), tpu_dev
                )
                self.qsa_v_caches[l] = jax.device_put(
                    jnp.zeros((self.max_ctx, 2, 256), dtype=jnp.bfloat16), tpu_dev
                )
            else:
                self.gdn_conv_states[l] = jax.device_put(
                    jnp.zeros((3, 10240), dtype=jnp.float32), tpu_dev
                )
                self.gdn_ssm_states[l] = jax.device_put(
                    jnp.zeros((48, 128, 128), dtype=jnp.float32), tpu_dev
                )

    def _embed_token_cpu(self, token_id: int) -> jax.Array:
        start = self.emb_g._data_start + self.emb_t.offset + token_id * self.emb_row_bytes
        raw = self.emb_mm[start : start + self.emb_row_bytes]
        f32 = dequantize_buffer_to_f32(raw, self.emb_t.type_id, 2560)
        return f32_to_tpu_f32(f32, (2560,))

    def _ple_gather_cpu(self, token_id: int) -> jax.Array:
        out_f32 = np.empty(2560, dtype=np.float32)
        LIB_STRATA.strata_ple_ngram_gather(
            ctypes.c_int32(token_id),
            ctypes.c_int32(self.prev_tokens[0]),
            ctypes.c_int32(self.prev_tokens[1]),
            ctypes.c_void_p(self.ple_base_ptr),
            ctypes.c_int(self.ple_t.type_id),
            out_f32.ctypes.data_as(ctypes.c_void_p)
        )
        return f32_to_tpu_f32(out_f32, (2560,))

    def _get_raw_expert_cached(self, layer: int, exp_id: int) -> Tuple[np.ndarray, int, int, np.ndarray, int, int, np.ndarray, int, int]:
        if not hasattr(self, "ram_expert_cache"):
            self.ram_expert_cache = OrderedDict()
            self.ram_expert_slots = int(os.environ.get("STRATA_RAM_EXPERT_SLOTS", "12000"))
        key = (layer, exp_id)
        if key in self.ram_expert_cache:
            self.cache_hits += 1
            self.ram_expert_cache.move_to_end(key)
            return self.ram_expert_cache[key]
        self.cache_misses += 1
        info = self.expert_info[layer]
        g_mm, g_base, g_bytes, g_type = info["gate"]
        u_mm, u_base, u_bytes, u_type = info["up"]
        d_mm, d_base, d_bytes, d_type = info["down"]
        g_arr = np.array(g_mm[g_base + exp_id * g_bytes : g_base + (exp_id + 1) * g_bytes], dtype=np.uint8, copy=True)
        u_arr = np.array(u_mm[u_base + exp_id * u_bytes : u_base + (exp_id + 1) * u_bytes], dtype=np.uint8, copy=True)
        d_arr = np.array(d_mm[d_base + exp_id * d_bytes : d_base + (exp_id + 1) * d_bytes], dtype=np.uint8, copy=True)
        entry = (g_arr, g_bytes, g_type, u_arr, u_bytes, u_type, d_arr, d_bytes, d_type)
        if len(self.ram_expert_cache) >= self.ram_expert_slots:
            self.ram_expert_cache.popitem(last=False)
        self.ram_expert_cache[key] = entry
        return entry

    def _eval_routed_experts(self, layer: int, mixed_tpu: jax.Array,
                             top_ids: np.ndarray, top_weights: np.ndarray,
                             sh_out_tpu: jax.Array, active_k: int = 4) -> jax.Array:
        """
        Zero-Allocation Direct C++ OpenMP Expert Execution + 24 GB Host RAM Compressed Cache:
          - Evaluates IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL / Q2_0 directly in C++.
          - Zero float32 matrix allocations and zero TPU HBM thrashing.
        """
        if active_k <= 0:
            return sh_out_tpu
        k_use = min(active_k, len(top_ids))
        ids_k = [int(x) for x in top_ids[:k_use]]
        w_k = np.asarray(top_weights[:k_use], dtype=np.float32)
        w_sum = float(np.sum(w_k))
        if w_sum > 0.0:
            w_k = w_k / w_sum

        mixed_np = np.ascontiguousarray(np.asarray(mixed_tpu, dtype=np.float32))
        cpu_out = np.zeros(2560, dtype=np.float32)

        for eid, w_val in zip(ids_k, w_k):
            g_arr, g_bytes, g_type, u_arr, u_bytes, u_type, d_arr, d_bytes, d_type = self._get_raw_expert_cached(layer, eid)
            LIB_STRATA.strata_eval_expert_raw(
                mixed_np.ctypes.data_as(ctypes.c_void_p),
                g_arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(g_bytes), ctypes.c_int(g_type),
                u_arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(u_bytes), ctypes.c_int(u_type),
                d_arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(d_bytes), ctypes.c_int(d_type),
                ctypes.c_float(float(w_val)),
                cpu_out.ctypes.data_as(ctypes.c_void_p),
            )

        return sh_out_tpu + f32_to_tpu_f32(cpu_out, (2560,))

    def forward_token(self, token_id: int, active_k: int = 4) -> jax.Array:
        """Runs one full 48-layer token step through Qwen3.8-Flash-Next 125B MoE."""
        emb = self._embed_token_cpu(token_id)
        ple_emb = self._ple_gather_cpu(token_id)

        # Initialize 4-stream hyper-connection residual stack R: (4, 2560)
        R = jnp.broadcast_to(emb[None, :], (4, 2560))

        pos_idx = self.pos
        cos_row = self.cos_tab[pos_idx]
        sin_row = self.sin_tab[pos_idx]
        pos_arr = jnp.int32(pos_idx)

        for l in range(48):
            lw = self.layers[l]

            # Layer 1 Per-Layer Token Embedding (PLE) residual update before gr_read
            if l == 1:
                R, self.ple_hist = ple_layer1_jit(
                    ple_emb, R, self.ple_hist,
                    self.ple_weights["key"][0], self.ple_weights["key"][1],
                    self.ple_weights["val"],
                    self.ple_weights["norm_key"],
                    self.ple_weights["norm_query"],
                    self.ple_weights["norm_conv"],
                    self.ple_weights["conv1d"]
                )

            # Half 1: Attention / Mixer Hyper-Connection Read
            mixed, inject = gr_read_jit(
                R, lw["hc_attn_norm"], lw["hc_attn_down"],
                lw["hc_attn_up"], lw["hc_attn_inject"]
            )

            if lw["is_qsa"]:
                block_out, self.qsa_k_caches[l], self.qsa_v_caches[l] = qsa_layer_jit(
                    mixed, pos_arr, cos_row, sin_row,
                    self.qsa_k_caches[l], self.qsa_v_caches[l],
                    lw["q"][0], lw["q"][1],
                    lw["k"][0], lw["k"][1],
                    lw["v"][0], lw["v"][1],
                    lw["o"][0], lw["o"][1],
                    lw["q_norm"], lw["k_norm"]
                )
            else:
                block_out, self.gdn_conv_states[l], self.gdn_ssm_states[l] = gdn_layer_jit(
                    mixed, self.gdn_conv_states[l], self.gdn_ssm_states[l],
                    lw["qkv"][0], lw["qkv"][1],
                    lw["conv_w"], lw["alpha_w"], lw["beta_w"],
                    lw["ssm_a"], lw["ssm_dt"],
                    lw["gate"][0], lw["gate"][1],
                    lw["ssm_norm"],
                    lw["out"][0], lw["out"][1]
                )

            R = gr_write_jit(R, block_out, inject)

            # Half 2: MoE FFN Hyper-Connection Read + Routing + Shared Expert
            mixed, inject_ffn = gr_read_jit(
                R, lw["hc_ffn_norm"], lw["hc_ffn_down"],
                lw["hc_ffn_up"], lw["hc_ffn_inject"]
            )
            top_ids_tpu, top_weights_tpu, sh_out_tpu = moe_route_and_shared_jit(
                mixed, lw["router"], lw["ginp"],
                lw["sgate"][0], lw["sgate"][1],
                lw["sup"][0], lw["sup"][1],
                lw["sdown"][0], lw["sdown"][1]
            )
            if active_k > 0:
                top_ids_np = np.asarray(top_ids_tpu)
                top_weights_np = np.asarray(top_weights_tpu)
                moe_out = self._eval_routed_experts(l, mixed, top_ids_np, top_weights_np, sh_out_tpu, active_k=active_k)
            else:
                moe_out = sh_out_tpu
            R = gr_write_jit(R, moe_out, inject_ffn)

        # Advance sequence position & PLE N-gram window
        self.prev_tokens = [self.prev_tokens[1], token_id]
        self.pos += 1

        logits = lm_head_jit(
            R, self.out_hc_norm, self.out_hc_down, self.out_hc_up,
            self.out_weight[0], self.out_weight[1]
        )
        return logits

    def sample_next_token(self, logits: jax.Array, temperature: float = 0.6, top_p: float = 0.95) -> int:
        if temperature <= 1e-5:
            return int(jnp.argmax(logits))
        scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= np.sum(probs)
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cum_probs = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cum_probs, top_p) + 1
            keep_idx = sorted_idx[:cutoff]
            keep_probs = probs[keep_idx]
            keep_probs /= np.sum(keep_probs)
            return int(np.random.choice(keep_idx, p=keep_probs))
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 256,
                 temperature: float = 0.6, top_p: float = 0.95):
        self.reset_state()
        prompt = self.tokenizer.apply_chat_template(messages, enable_thinking=False)
        prompt_ids = self.tokenizer.encode(prompt, parse_special=True)
        prefill_k = int(os.environ.get("STRATA_PREFILL_K", "0"))
        decode_k  = int(os.environ.get("STRATA_DECODE_K", "2"))
        print(f"[Strata-TPU] Prefilling {len(prompt_ids)} prompt tokens (prefill_k={prefill_k}, decode_k={decode_k})...")
        t0 = time.time()
        logits = None
        for idx, tid in enumerate(prompt_ids):
            k_step = decode_k if (idx == len(prompt_ids) - 1) else prefill_k
            logits = self.forward_token(tid, active_k=k_step)
        prefill_s = time.time() - t0
        print(f"[Strata-TPU] Prefill complete in {prefill_s:.2f}s. Decoding...")

        for step in range(max_tokens):
            next_id = self.sample_next_token(logits, temperature=temperature, top_p=top_p)
            if next_id in (self.tokenizer.eos_id, self.tokenizer.im_end_id):
                break
            piece = self.tokenizer.decode([next_id])
            yield piece
            logits = self.forward_token(next_id, active_k=decode_k)

# ------------------------------------------------------------------------------
# 7. LAUNCH ENGINE + FASTAPI GATEWAY + CLOUDFLARE PUBLIC TUNNEL
# ------------------------------------------------------------------------------
ENGINE = StrataTPUEngine(
    SHARD1_PATH,
    SHARD2_PATH,
    max_ctx=int(os.environ.get("STRATA_MAX_CTX", "4096")),
    hbm_expert_slots=0,
)

import nest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

nest_asyncio.apply()
app = FastAPI(title="Strata-TPU 125B Cloud Engine")

@app.get("/health")
async def health():
    total_access = ENGINE.cache_hits + ENGINE.cache_misses
    hit_rate = (100.0 * ENGINE.cache_hits / total_access) if total_access > 0 else 0.0
    ram_cached = len(getattr(ENGINE, "ram_expert_cache", {}))
    return {
        "status": "ok",
        "model": "Huihui-Qwen3.8-Flash-Next-abliterated-IQ3_XXS",
        "backend": jax.default_backend().upper(),
        "cached_experts_in_ram": ram_cached,
        "expert_cache_hit_rate_pct": round(hit_rate, 2),
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "huihui-qwen3.8-flash-next-125b-abliterated",
            "object": "model",
            "owned_by": "strata-tpu",
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    stream = bool(body.get("stream", False))

    if stream:
        def event_stream():
            init_chunk = {
                "id": "chatcmpl-strata-tpu",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(init_chunk)}\n\n"
            for piece in ENGINE.generate(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p):
                chunk = {
                    "id": "chatcmpl-strata-tpu",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": "chatcmpl-strata-tpu",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")


    out_pieces = list(ENGINE.generate(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p))
    full_text = "".join(out_pieces)
    return JSONResponse({
        "id": "chatcmpl-strata-tpu",
        "object": "chat.completion",
        "model": "huihui-qwen3.8-flash-next-125b-abliterated",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
    })

@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    system_prompt = body.get("system", "")
    if system_prompt:
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(s.get("text", "") for s in system_prompt if isinstance(s, dict))
        messages = [{"role": "system", "content": system_prompt}] + messages
    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.6))
    out_pieces = list(ENGINE.generate(messages, max_tokens=max_tokens, temperature=temperature))
    full_text = "".join(out_pieces)
    return JSONResponse({
        "id": "msg_strata_tpu",
        "type": "message",
        "role": "assistant",
        "model": "huihui-qwen3.8-flash-next-125b-abliterated",
        "content": [{"type": "text", "text": full_text}],
        "stop_reason": "end_turn",
    })

def start_cloudflared_tunnel(port: int = 8080):
    proc = subprocess.Popen(
        ["/usr/local/bin/cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        if ".trycloudflare.com" in line:
            for token in line.split():
                if token.startswith("https://") and ".trycloudflare.com" in token:
                    print("\n" + "=" * 80)
                    print(f" 🌐 STRATA-TPU PUBLIC API ENDPOINT: {token}")
                    print(f"    OpenAI Chat URL   : {token}/v1/chat/completions")
                    print(f"    Anthropic Messages: {token}/v1/messages")
                    print(f"    Health & Cache    : {token}/health")
                    print("=" * 80 + "\n")

def _pick_free_port(start_port: int = 8000) -> int:
    import socket
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return 8000

STRATA_PORT = _pick_free_port(8000)
print(f"\n[6/6] Launching FastAPI Gateway + Cloudflare Public Tunnel on port {STRATA_PORT}...")
threading.Thread(target=start_cloudflared_tunnel, args=(STRATA_PORT,), daemon=True).start()

def _serve_uvicorn():
    import asyncio
    cfg = uvicorn.Config(app, host="0.0.0.0", port=STRATA_PORT, log_level="info")
    srv = uvicorn.Server(cfg)
    asyncio.run(srv.serve())

_uv_thread = threading.Thread(target=_serve_uvicorn, daemon=True)
_uv_thread.start()
_uv_thread.join()


