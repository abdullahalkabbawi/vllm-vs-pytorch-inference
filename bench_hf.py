"""Baseline: plain Hugging Face transformers, eager PyTorch, model.generate()."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import get_args, latency_summary, make_prompts, save


def main():
    p = get_args("hf-eager")
    p.add_argument("--batch-sizes", default="1,8,16", help="static batch sizes to try for throughput")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cuda"
    ).eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def generate(batch, n_tokens):
        ids = torch.tensor(batch, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,  # never stop early at EOS: fixed work per request
                do_sample=False,
                pad_token_id=pad_id,
            )
        torch.cuda.synchronize()
        assert out.shape[1] - ids.shape[1] == n_tokens
        return time.perf_counter() - t0

    prompts = make_prompts(args.num_prompts, args.input_len, len(tok))

    print("Warming up...")
    generate(prompts[:1], 16)

    # --- Latency: one request at a time ---
    ttfts, e2es = [], []
    for i in range(args.latency_runs):
        ttfts.append(generate([prompts[i]], 1))
        e2es.append(generate([prompts[i]], args.output_len))
        print(f"  latency run {i + 1}: ttft={ttfts[-1] * 1000:.0f} ms, e2e={e2es[-1]:.2f} s")
    latency = latency_summary(ttfts, e2es, args.output_len)

    # --- Throughput: all prompts, processed in fixed-size batches ---
    throughput = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        try:
            total = sum(generate(prompts[i:i + bs], args.output_len) for i in range(0, len(prompts), bs))
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs}: out of memory, skipping")
            torch.cuda.empty_cache()
            continue
        r = {
            "mode": f"static batch {bs}",
            "total_s": total,
            "output_tok_per_s": len(prompts) * args.output_len / total,
            "requests_per_s": len(prompts) / total,
        }
        print(f"  batch {bs}: {r['output_tok_per_s']:.0f} tok/s")
        throughput.append(r)

    save(args.tag, args, latency, throughput)


if __name__ == "__main__":
    main()
