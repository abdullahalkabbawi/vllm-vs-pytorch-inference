"""Read results/*.json, print a comparison table, and save results/comparison.png."""
import glob
import json

import matplotlib.pyplot as plt

# Fixed color per engine (blue, orange, aqua), so a missing run never repaints the others.
COLORS = {"hf-eager": "#2a78d6", "vllm": "#eb6834", "vllm-eager": "#1baf7a"}
ORDER = ["hf-eager", "vllm-eager", "vllm"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

runs = {}
for path in glob.glob("results/*.json"):
    with open(path) as f:
        r = json.load(f)
    runs[r["tag"]] = r
tags = [t for t in ORDER if t in runs] + sorted(t for t in runs if t not in ORDER)
if not tags:
    raise SystemExit("No results yet: run bench_hf.py and bench_vllm.py first.")

# --- Table ---
print(f"\n{'run':<34}{'tok/s':>10}{'TTFT ms':>10}{'TPOT ms':>10}{'e2e s':>8}")
print("-" * 72)
for t in tags:
    lat = runs[t]["latency"]
    for i, thr in enumerate(runs[t]["throughput"]):
        lat_cols = f"{lat['ttft_ms']:>10.1f}{lat['tpot_ms']:>10.1f}{lat['e2e_s']:>8.2f}" if i == 0 else ""
        print(f"{t + ' / ' + thr['mode']:<34}{thr['output_tok_per_s']:>10.0f}{lat_cols}")

best = {t: max(x["output_tok_per_s"] for x in runs[t]["throughput"]) for t in tags}
if "hf-eager" in best:
    for t in tags:
        if t != "hf-eager":
            print(f"\n{t}: {best[t] / best['hf-eager']:.1f}x the throughput of the best HF batch size, "
                  f"{runs['hf-eager']['latency']['tpot_ms'] / runs[t]['latency']['tpot_ms']:.1f}x faster per token")

# --- Chart: three small panels, one measure each ---
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE,
                         gridspec_kw={"width_ratios": [2, 1, 1]})

def panel(ax, labels, values, colors, title, unit):
    ax.set_facecolor(SURFACE)
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:,.0f}" if v >= 100 else f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color=INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(unit)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.15)

labels, values, colors = [], [], []
for t in tags:
    for thr in runs[t]["throughput"]:
        labels.append(t + "\n" + thr["mode"].replace(" ", "\n", 1))
        values.append(thr["output_tok_per_s"])
        colors.append(COLORS.get(t, MUTED))
panel(axes[0], labels, values, colors, "Throughput\n(higher is better)", "output tokens / s")

cols = [COLORS.get(t, MUTED) for t in tags]
panel(axes[1], tags, [runs[t]["latency"]["ttft_ms"] for t in tags], cols,
      "Time to first token\n(lower is better)", "ms")
panel(axes[2], tags, [runs[t]["latency"]["tpot_ms"] for t in tags], cols,
      "Time per output token\n(lower is better)", "ms")

cfg = runs[tags[0]]["config"]
fig.suptitle(f"{cfg['model']}  ·  {cfg['input_len']} in / {cfg['output_len']} out tokens  ·  "
             f"{cfg['num_prompts']} requests", x=0.01, ha="left", color=MUTED)
fig.tight_layout()
fig.savefig("results/comparison.png", dpi=150, facecolor=SURFACE)
print("\nSaved results/comparison.png")
