"""Shared settings so both engines run the exact same workload."""
import argparse
import json
import os
import random
import statistics

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_args(engine_name):
    p = argparse.ArgumentParser(description=f"Benchmark {engine_name}")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="float16", help="float16 works on every GPU; bfloat16 on Ampere+")
    p.add_argument("--input-len", type=int, default=512, help="prompt length in tokens")
    p.add_argument("--output-len", type=int, default=256, help="generated tokens per request")
    p.add_argument("--num-prompts", type=int, default=64, help="requests in the throughput test")
    p.add_argument("--latency-runs", type=int, default=5, help="repeats for the single-request latency test")
    p.add_argument("--tag", default=engine_name, help="name used for the results file")
    return p


def make_prompts(n, input_len, vocab_size, seed=0):
    """Random token IDs: every prompt is unique (no prefix-cache shortcuts) and exactly input_len long.
    The text is gibberish, but speed doesn't depend on meaning, so this is the standard way to benchmark."""
    rng = random.Random(seed)
    hi = min(vocab_size, 100_000) - 1
    return [[rng.randint(1000, hi) for _ in range(input_len)] for _ in range(n)]


def latency_summary(ttfts, e2es, output_len):
    ttft = statistics.median(ttfts)
    e2e = statistics.median(e2es)
    tpot = (e2e - ttft) / (output_len - 1)  # time per output token after the first
    return {
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "e2e_s": e2e,
        "single_stream_tok_per_s": output_len / e2e,
    }


def save(tag, args, latency, throughput):
    os.makedirs("results", exist_ok=True)
    out = {"tag": tag, "config": vars(args), "latency": latency, "throughput": throughput}
    path = os.path.join("results", f"{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved {path}")
