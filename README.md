# CollabZX — Strata-TPU 125B Cloud Engine & Arch Linux Local Turbo Suite

Custom **Strata** deployments for **Huihui-Qwen3.8-Flash-Next-abliterated (Swift 1.5 - `IQ3_XXS`, 125B MoE)**:
1. **Strata-TPU Cloud Engine (`STRATA_TPU_ENGINE.py` / `STRATA_TPU_RUNNER.txt`)**: Runs the 125B MoE model on **Google Colab Cloud TPU v5e-1** (16 GB HBM2e + 47.05 GB Host RAM) backed by a 5 TB Google Drive (`zero local laptop heat`).
2. **Arch Linux Local Turbo Config (`strata-huihui-swift-linux.json`)**: Custom Strata v0.1.40.3 configuration tuned for **Intel Hybrid P/E-core CPUs + NVIDIA RTX A5500 16 GB** to push local decode speed beyond the 35–47 tok/s baseline into the **52–65+ tok/s** range.

---

## 1. Google Colab Cloud TPU v5e-1 Deployment

### Step A — One-Time 5 TB Google Drive Ingestion (`HUIHUIDRIVE.txt`)
Run [`HUIHUIDRIVE.txt`](./HUIHUIDRIVE.txt) once on any Colab CPU runtime to pull the 76 GB GGUF shards (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` and `00002-of-00002.gguf`) directly from Hugging Face into `/content/drive/MyDrive/StrataModels/huihui-swift-IQ3_XXS` at datacenter speed via `aria2c`.

### Step B — Launch the Strata-TPU 125B Engine (`STRATA_TPU_ENGINE.py` / `STRATA_TPU_RUNNER.txt`)
1. Switch your Colab runtime to **TPU v5e-1** (*Runtime $\rightarrow$ Change runtime type $\rightarrow$ TPU v5e-1*).
2. Paste [`STRATA_TPU_RUNNER.txt`](./STRATA_TPU_RUNNER.txt) (or [`STRATA_TPU_ENGINE.py`](./STRATA_TPU_ENGINE.py)) into a single cell and run it.
3. The script automatically:
   - Tunes the Colab Linux kernel (`transparent_hugepage=always`, `vm.overcommit_memory=1`, `vm.swappiness=10`, `vm.vfs_cache_pressure=50`).
   - Compiles `/tmp/libstrata_tpu.so` (`g++ -O3 -march=native -fopenmp`) with Strata's custom **Type 42 (`Q2_0`)**, `IQ4_NL`, `IQ4_XS`, and `Q8_0` multi-threaded dequantizers, row-wise `INT8` MXU quantizer, and Layer 1 **64-bit XOR N-gram hash + 16-head `IQ4_NL` PLE table gather**.
   - Stages the 48-layer Dense Backbone (`GR`, `PLE`, `36 GDN`, `12 QSA`, `48 Routers`, `48 Shared Experts`, `LM Head`) into **~4.98 GB of TPU v5e HBM2e** (`INT8` + `bfloat16`), leaving over **10 GB of HBM2e** for the **Dynamic LRU Expert Cache**.
   - Starts a FastAPI gateway (`/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/health`) and exposes a public `https://*.trycloudflare.com` tunnel URL.

---

## 2. Arch Linux Local Turbo Tuning (35–47 tok/s $\rightarrow$ 52–65+ tok/s)

Diagnostic analysis of `strata-huihui-swift-linux.log` on the local **Arch Linux + RTX A5500 Laptop GPU (16 GB) + Intel 24-thread Hybrid P/E-Core CPU** identified five specific bottlenecks holding decode speed to 35–47 tok/s. All five are resolved in [`strata-huihui-swift-linux.json`](./strata-huihui-swift-linux.json):

| Bottleneck in Baseline Log | Turbo Customization Applied | Speed Impact |
| :--- | :--- | :--- |
| **1. Intel Hybrid P/E-Core Barrier Stall**<br>15 workers spanned P-cores (`0,2..14`) and E-cores (`16..23`) with only `48` tasks/phase (`3` tasks/thread). Fast P-cores stalled at every layer barrier waiting for an E-core to finish its 1/48th slice. | Added `"--pool-tasks", "192"` + `"--host-core", "last"` + `"STRATA_IQ3S_MT1": "1"` | Eliminates E-core tail-latency barrier stalls by splitting each MoE phase into 192 micro-tasks and moving the GPU spin-wait thread off P-Core 0. |
| **2. Chained Suffix + MTP Verify Window Capped at 4**<br>At 150K context, suffix drafts hit 98%+ acceptance (`2543/2547`), yet `--spec 4` capped windows at 5 tokens even though the 6-token verify buffer was already allocated in VRAM. | Added `"--spec", "6"`, `"--mtp-max-t", "4"`, `"--lookup-chain", "4"`, `"--spec-min-p", "0.38"`, `"--coupled-draft"`, `"STRATA_SPEC_GUMBEL": "1"` | Chains prompt-lookup suffix matches onto MTP drafts (expanding verify windows up to 8 tokens) and couples Gumbel-max sampling between drafter and target. |
| **3. Uncompressed MTP Draft Layer (`836 MiB` + `178 MiB` head)**<br>Full-precision MTP projections and draft head consumed ~1 GiB of VRAM and slowed each draft step. | Added `"--mtp-q4", "all"` | Runs the MTP draft layer and draft head from 4-bit copies (~2x faster draft generation + frees VRAM for main-model experts). |
| **4. Suboptimal VRAM Budget (`5,426` cached experts)**<br>`--kv int8` and `--vram-reserve-mib 600` held back ~650 MiB of RTX A5500 VRAM from the expert cache. | Changed to `"--kv", "k8v4"` and `"--vram-reserve-mib", "256"` | Frees ~650+ MiB of VRAM (23% smaller KV cache with Hadamard-rotated 4-bit V), adding **~450–650 more resident experts** directly into RTX A5500 VRAM. |
| **5. Synchronous Adaptive Expert Swaps (`943,776` swaps)**<br>Nearly 1M RAM$\leftrightarrow$VRAM expert swaps executed blocking host copies between decode windows. | Added `"--adapt-async", "1"` and `"STRATA_EXCHANGE_ROTATE": "1"` | Runs adaptive VRAM swaps asynchronously on a background helper thread and rotates page-locked buffer ownership with zero host memory copies. |
