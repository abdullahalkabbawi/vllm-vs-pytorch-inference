#!/usr/bin/env bash
# Run the whole comparison on a Linux GPU box. Usage: bash run_all.sh [extra args, e.g. --model Qwen/Qwen2.5-1.5B-Instruct]
set -e
cd "$(dirname "$0")"

python bench_hf.py "$@"
python bench_vllm.py "$@"
python bench_vllm.py --enforce-eager "$@"
python compare.py
