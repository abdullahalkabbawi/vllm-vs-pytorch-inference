"""vLLM: PagedAttention + continuous batching + CUDA graphs (unless --enforce-eager)."""
import time

from vllm import LLM, SamplingParams

from common import get_args, latency_summary, make_prompts, save


def main():
    p = get_args("vllm")
    p.add_argument("--enforce-eager", action="store_true",
                   help="disable CUDA graphs, to see how much they contribute")
    p.add_argument("--gpu-mem", type=float, default=0.90, help="fraction of VRAM vLLM may use")
    args = p.parse_args()
    if args.enforce_eager and args.tag == "vllm":
        args.tag = "vllm-eager"

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.input_len + args.output_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,  # prompts are unique anyway; keep it a fair fight
        seed=0,
    )

    def generate(batch, n_tokens):
        params = SamplingParams(temperature=0, max_tokens=n_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": ids} for ids in batch], params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) == n_tokens for o in outs)
        return elapsed

    prompts = make_prompts(args.num_prompts, args.input_len, len(llm.get_tokenizer()))

    print("Warming up...")
    generate(prompts[:1], 16)

    # --- Latency: one request at a time ---
    ttfts, e2es = [], []
    for i in range(args.latency_runs):
        ttfts.append(generate([prompts[i]], 1))
        e2es.append(generate([prompts[i]], args.output_len))
        print(f"  latency run {i + 1}: ttft={ttfts[-1] * 1000:.0f} ms, e2e={e2es[-1]:.2f} s")
    latency = latency_summary(ttfts, e2es, args.output_len)

    # --- Throughput: hand vLLM every prompt at once; its scheduler batches continuously ---
    total = generate(prompts, args.output_len)
    throughput = [{
        "mode": "continuous batching",
        "total_s": total,
        "output_tok_per_s": len(prompts) * args.output_len / total,
        "requests_per_s": len(prompts) / total,
    }]
    print(f"  all {len(prompts)} at once: {throughput[0]['output_tok_per_s']:.0f} tok/s")

    save(args.tag, args, latency, throughput)


if __name__ == "__main__":
    main()
