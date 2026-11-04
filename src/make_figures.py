"""Generate all paper figures from files in results/ -> paper/figs/*.pdf (+png).
Usage: python src/make_figures.py   (skips figures whose inputs do not exist yet)"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "paper" / "figs"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout(); fig.savefig(F / f"{name}.pdf"); fig.savefig(F / f"{name}.png", dpi=200); plt.close(fig)
    print("wrote", name)


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def training_curves(run="lora_final_src"):
    runs = [d for d in R.iterdir() if d.is_dir() and (d / "train_log.csv").exists() and d.name.startswith("lora_")]
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for d in sorted(runs):
        rows = read_csv(d / "train_log.csv")
        tr = [(float(r["step"]), float(r["train_loss"])) for r in rows if r["train_loss"]]
        va = [(float(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
        if tr:
            ax[0].plot(*zip(*tr), label=d.name)
        if va:
            ax[1].plot(*zip(*va), marker="o", label=d.name)
    ax[0].set(xlabel="optimizer step", ylabel="train loss (answer tokens)", title="Training loss", yscale="log")
    ax[1].set(xlabel="optimizer step", ylabel="validation loss", title="Validation loss (epoch ends)")
    ax[0].legend(); ax[1].legend()
    save(fig, "training_curves")


def confusion(tag, split, name):
    p = R / f"eval_{tag}_{split}.json"
    if not p.exists():
        return
    rep = json.load(open(p))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, part in zip(axs, ("inclusion", "exclusion")):
        cm = rep[part]["classification"]["confusion_matrix"]
        m = np.array(cm["matrix"]); rows, cols = cm["rows_gold"], cm["cols_pred"]
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("not enough information", "NEI").replace("not applicable", "N/A") for c in cols], rotation=45, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([c.replace("not enough information", "NEI").replace("not applicable", "N/A") for c in rows])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set(xlabel="predicted", ylabel="gold", title=f"{part} ({tag}, {split})"); ax.grid(False)
    save(fig, name)


def kv_bench():
    p = R / "kv_benchmark.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    cfgs = sorted({(int(r["prompt_len"]), int(r["new_tokens"])) for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    x = np.arange(len(cfgs)); w = 0.38
    for k, (metric, lab) in enumerate((("decode_tok_per_s", "decode tokens/s"), ("prefill_ms", "prefill (ms)"), ("peak_mem_MiB", "peak memory (MiB)"))):
        for j, mode in enumerate(("no_cache", "kv_cache")):
            v = [float(next(r for r in rows if r["mode"] == mode and (int(r["prompt_len"]), int(r["new_tokens"])) == c)[metric]) for c in cfgs]
            ax[k].bar(x + (j - 0.5) * w, v, w, label=mode)
        ax[k].set_xticks(x); ax[k].set_xticklabels([f"{p}+{n}" for p, n in cfgs]); ax[k].set(ylabel=lab, xlabel="prompt+new tokens")
    ax[0].legend(); save(fig, "kv_benchmark")


def attention_bench():
    p = R / "attention_benchmark.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    for impl in ("reference", "tiled_64", "tiled_256", "sdpa"):
        rs = [r for r in rows if r["impl"] == impl]
        L = [int(r["L"]) for r in rs]
        ax[0].plot(L, [float(r["median_ms"]) for r in rs], marker="o", label=impl)
        ax[1].plot(L, [float(r["peak_extra_MiB"]) for r in rs], marker="o", label=impl)
    for a, yl in zip(ax, ("median time (ms)", "peak extra memory (MiB)")):
        a.set(xscale="log", yscale="log", xlabel="sequence length", ylabel=yl)
    ax[0].legend(); save(fig, "attention_benchmark")


def spec_bench():
    p = R / "spec_benchmark_final.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    works = sorted({r["workload"] for r in rows})
    fig, ax = plt.subplots(1, len(works), figsize=(7, 2.6), squeeze=False)
    for a, w in zip(ax[0], works):
        rs = [r for r in rows if r["workload"] == w and r["method"] == "speculative"]
        ks = [int(r["k"]) for r in rs]
        a.plot(ks, [float(r["speedup"]) for r in rs], marker="o", label="speedup vs greedy-KV")
        a.axhline(1.0, color="k", ls="--", lw=0.8)
        a2 = a.twinx(); a2.grid(False)
        a2.plot(ks, [float(r["acceptance_rate"]) for r in rs], marker="s", color="C1", label="acceptance rate")
        a.set(xlabel="draft length k", ylabel="speedup", title=w[:28]); a2.set(ylabel="acceptance")
    save(fig, "spec_benchmark")


if __name__ == "__main__":
    training_curves()
    for tag in ("base_zeroshot", "lora_final"):
        confusion(tag, "test", f"confusion_{tag}_test")
    kv_bench(); attention_bench(); spec_bench()
