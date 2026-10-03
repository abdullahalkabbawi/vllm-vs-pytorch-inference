# Eager PyTorch vs vLLM: Qwen inference benchmark

Same model, same prompts, same number of generated tokens. Two engines:

| Run | What it is |
|---|---|
| `hf-eager` | Hugging Face `transformers` + `model.generate()`: plain eager PyTorch, the baseline |
| `vllm` | vLLM with all its tricks: PagedAttention, continuous batching, CUDA graphs |
| `vllm-eager` | vLLM with CUDA graphs off (`--enforce-eager`), to show how much CUDA graphs alone add |

## What gets measured

**Latency** (one request on its own, median of 5 runs)
- **TTFT** (time to first token): how long until the user sees the first word. Mostly *prefill*, meaning the prompt gets processed.
- **TPOT** (time per output token): how fast each later token arrives. This is *decode*, and it's limited by memory bandwidth.

**Throughput** (64 requests in total)
- Output tokens per second across all requests. This shows how many users one GPU can serve.
- HF runs requests in fixed batches (1, 8, 16). vLLM gets all 64 at once and its scheduler decides how to batch them.

Prompts are random token IDs, each exactly `--input-len` tokens long, and every request produces exactly
`--output-len` tokens (EOS is ignored). That way both engines do identical work.

## Where to run it

vLLM needs **Linux + an NVIDIA GPU**. It does not run natively on Windows.

### Option A: cloud GPU (recommended for the 7B model)
Qwen2.5-7B in fp16 needs ~15 GB just for weights, so you need a **24 GB+ GPU** (L4, A10G, RTX 4090, A100...).
A Colab T4 (16 GB) is too small. Colab Pro with an L4, RunPod, Lambda, or Vast.ai all work.

```bash
python3 -m venv venv && source venv/bin/activate
pip install uv
uv pip install vllm matplotlib --torch-backend=auto
bash run_all.sh        # runs the three benchmarks, then compare.py
```

**Gotcha: `ImportError: libcudart.so.13`.** The default vLLM wheel on PyPI is built for CUDA 13, which needs
NVIDIA driver 580+. Check yours with `nvidia-smi`. On an older driver (e.g. 570), install the CUDA 12.9 build
from the vLLM GitHub release that matches your version:
```bash
uv pip install --reinstall-package vllm \
  "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --torch-backend=cu129
```

### Option B: your laptop (RTX 4060, 8 GB) through WSL2
The 7B model **won't fit** in 8 GB at fp16. Use the 1.5B model so the comparison stays fair (same precision on both sides):

```powershell
wsl --install   # in an admin PowerShell, then reboot
```
Then inside the Ubuntu terminal:
```bash
python3 -m venv venv && source venv/bin/activate
pip install vllm matplotlib
M=Qwen/Qwen2.5-1.5B-Instruct
python bench_hf.py   --model $M --batch-sizes 1,8,16,32
python bench_vllm.py --model $M
python bench_vllm.py --model $M --enforce-eager
python compare.py
```

Results go to `results/*.json` and `results/comparison.png`.

## Useful knobs
- `--input-len / --output-len`: try a long prompt with a short answer (`--input-len 2048 --output-len 64`) and the reverse. The first stresses prefill (TTFT); the second stresses decode (TPOT).
- `--num-prompts 256`: more requests means more for vLLM's scheduler to work with, so the throughput gap usually grows.
- `--batch-sizes 1,8,16,32,64`: find where HF runs out of memory. vLLM's paged KV cache keeps going much further.

## What to expect (and why)
- **Single-request latency:** vLLM is faster, mostly from CUDA graphs (compare `vllm` with `vllm-eager`)
  and fused kernels. Eager PyTorch launches hundreds of small GPU kernels per token, and at batch size 1 the GPU sits waiting on Python.
- **Throughput:** this is where vLLM pulls far ahead. HF pre-allocates padded KV cache and waits for the slowest
  request in each batch. vLLM stores KV cache in pages (no waste), so it fits many more requests in memory at once.
  It also adds new requests the moment a slot frees up (continuous batching).
