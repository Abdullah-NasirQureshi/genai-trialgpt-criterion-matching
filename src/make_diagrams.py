"""Architecture / algorithm diagrams for the paper (matplotlib only) -> paper/figs/*.pdf|png
  arch_overview   system + application workflow
  arch_kvcache    decoder-only Transformer with KV cache: prefill vs decode step
  arch_tiled      tiled online-softmax attention (blocks, causal skip, running statistics)
  arch_spec       one round of greedy speculative decoding with cache synchronisation
Usage: python src/make_diagrams.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "figure.dpi": 150})
C = {"in": "#dbe9f6", "proc": "#e8f1de", "model": "#fde9c9", "out": "#f3dcdc", "cache": "#e6dff0", "gray": "#eeeeee"}


def box(ax, x, y, w, h, text, fc="#ffffff", fs=7, bold=False, ec="#444444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, color="#333333", rad=0.0, fs=6, dy=0.08):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def canvas(w, h, xmax, ymin, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(ymin, ymax); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220); plt.close(fig)
    print("wrote", name)


def overview():
    fig, ax = canvas(7.2, 3.4, 14.4, 0.0, 7.0)
    box(ax, 0.1, 4.5, 2.5, 1.3, "Patient note\n(free text or\nnumbered sentences)", C["in"])
    box(ax, 0.1, 2.6, 2.5, 1.3, "Trial criteria\n(inclusion / exclusion,\none per line)", C["in"])
    box(ax, 0.1, 0.7, 2.5, 1.3, "Inference mode:\nno cache | KV cache |\nspeculative (k)", C["in"])
    box(ax, 3.4, 3.2, 2.4, 1.6, "Input validation\n(empty, too long,\n>30 criteria ...)", C["proc"])
    box(ax, 6.5, 3.2, 2.6, 1.6, "Prompt builder\nIDs + type + criterion\n+ allowed labels", C["proc"])
    box(ax, 6.5, 0.5, 2.6, 2.0, "Qwen2.5-1.5B-Instruct\n+ LoRA adapter\n(target, fp16)\n\ndraft: Qwen2.5-0.5B\n+ LoRA (speculative only)", C["model"])
    box(ax, 9.9, 3.2, 2.1, 1.6, "Output parser\n{label, evidence IDs}\ninvalid -> flagged", C["proc"])
    box(ax, 9.9, 0.5, 2.1, 1.9, "Evidence lookup:\nexact sentence text\n+ template\nexplanation", C["proc"])
    box(ax, 12.4, 3.2, 1.9, 1.6, "Rule-based\naggregation\n(overall outcome)", C["proc"])
    box(ax, 12.4, 0.5, 1.9, 2.1, "Result table +\nHTML report\n(provenance,\ndisclaimer) +\nlive benchmark", C["out"])
    arrow(ax, 2.6, 5.1, 3.4, 4.4); arrow(ax, 2.6, 3.3, 3.4, 3.9); arrow(ax, 2.6, 1.4, 6.5, 1.6)
    arrow(ax, 5.8, 4.0, 6.5, 4.0); arrow(ax, 7.8, 3.2, 7.8, 2.5, "prompt")
    arrow(ax, 9.1, 1.5, 9.9, 3.2, "tokens", rad=-0.15); arrow(ax, 12.0, 4.0, 12.4, 4.0)
    arrow(ax, 10.95, 3.2, 10.95, 2.4); arrow(ax, 12.0, 1.5, 12.4, 1.5)
    arrow(ax, 13.35, 3.2, 13.35, 2.6)
    ax.text(7.2, 6.75, "Educational screening prototype - not a clinical enrollment decision", ha="center", fontsize=7.5, fontweight="bold", color="#8a1c1c")
    ax.text(7.2, 6.25, "Per criterion: one prompt -> one generated JSON answer (greedy). Overall outcome = explicit rules, not a ground-truth label.", ha="center", fontsize=6.3, color="#444444")
    save(fig, "arch_overview")


def kvcache():
    fig, ax = canvas(7.2, 3.3, 14.4, -0.7, 6.6)
    ax.text(3.6, 6.3, "(a) Prefill: whole prompt, once", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(10.8, 6.3, "(b) Decode step t: only the newest token", ha="center", fontsize=7.5, fontweight="bold")
    notes = {0.0: "writes K,V of all P prompt positions\ninto the cache (every layer)",
             7.2: "appends K,V of the new token (+1 position);\nthe query attends to ALL cached keys"}
    for off, ntok in ((0.0, "P prompt tokens"), (7.2, "1 token (position_id = cache length)")):
        box(ax, off + 0.2, 5.1, 6.4, 0.8, ntok, C["in"])
        box(ax, off + 0.2, 4.2, 6.4, 0.6, "Embedding  (+ RoPE angles from absolute position_ids)", C["gray"], fs=6.5)
        ax.add_patch(Rectangle((off + 0.2, 1.0), 6.4, 2.9, fc="#fbfbfb", ec="#444444", lw=0.8, ls="--"))
        ax.text(off + 0.35, 3.62, "x 28 layers", fontsize=6.5, style="italic")
        box(ax, off + 0.5, 2.85, 2.9, 0.55, "RMSNorm", C["gray"], fs=6.5)
        box(ax, off + 0.5, 1.8, 2.9, 0.85, "GQA self-attention\n12 query / 2 KV heads", C["model"], fs=6.5)
        box(ax, off + 3.8, 2.85, 2.6, 0.55, "RMSNorm", C["gray"], fs=6.5)
        box(ax, off + 3.8, 1.8, 2.6, 0.85, "SwiGLU MLP", C["model"], fs=6.5)
        box(ax, off + 0.5, 1.12, 5.9, 0.5, notes[off], C["cache"], fs=5.8, ec="#5b3a8c")
        box(ax, off + 0.2, 0.15, 6.4, 0.6, "final RMSNorm + LM head -> logits of the LAST position -> argmax", C["out"], fs=6.3)
        arrow(ax, off + 3.4, 5.1, off + 3.4, 4.8); arrow(ax, off + 3.4, 4.2, off + 3.4, 3.95); arrow(ax, off + 3.4, 1.0, off + 3.4, 0.78)
    ax.text(7.2, -0.45, "KV cache size per token = 2 (K and V) x 28 layers x 2 KV heads x 128 dims x 2 bytes = 28 KiB;  masking: causal triangle in prefill, all-ones in decode",
            ha="center", fontsize=6.2, color="#5b3a8c")
    save(fig, "arch_kvcache")


def tiled():
    fig, ax = canvas(7.2, 3.1, 14.4, 0.0, 6.2)
    nq, nk, s = 4, 4, 0.9
    x0, y0 = 1.3, 0.9
    for i in range(nq):
        for j in range(nk):
            fc = "#e4e4e4" if j > i else ("#9fcf90" if j == i else "#c9e3c0")
            ax.add_patch(Rectangle((x0 + j * s, y0 + (nq - 1 - i) * s), s, s, fc=fc, ec="#555555", lw=0.6))
            ax.text(x0 + j * s + s / 2, y0 + (nq - 1 - i) * s + s / 2, "S%d%d" % (i, j) if j <= i else "skip", ha="center", va="center", fontsize=5.5)
    ax.text(x0 + 2 * s, y0 + nq * s + 0.3, "key/value blocks K_j, V_j", ha="center", fontsize=6.3)
    ax.text(0.45, y0 + 2 * s, "query blocks Q_i", rotation=90, ha="center", va="center", fontsize=6.3)
    ax.text(x0 + 2 * s, 0.4, "blocks above the diagonal are never computed;\ndiagonal blocks (dark) are masked element-wise", ha="center", va="center", fontsize=5.8)
    box(ax, 5.4, 3.35, 8.8, 2.55,
        "For one query block Q_i sweep key blocks j = 0..i, keeping per query row\n"
        "m = running max,  l = running sum of exp,  o = running unnormalised output\n"
        "S = Q_i K_j^T / sqrt(d)   (+ causal mask)\n"
        "m' = max(m, rowmax S),   a = exp(m - m'),   P = exp(S - m')\n"
        "l' = a * l + rowsum(P),   o' = a * o + P V_j      ;      at the end:  output = o / l", C["proc"], fs=6.0)
    box(ax, 5.4, 0.5, 4.2, 2.5, "Memory: only one B_q x B_k tile\nof scores exists at a time\n(never the full L x L matrix):\n59-158x less extra memory\nat L = 4096", C["cache"], fs=6.0)
    box(ax, 10.0, 0.5, 4.2, 2.5, "Exact softmax attention\n(no approximation):\nmax error 4e-5 vs float64 (fp32).\nA plain PyTorch loop is slower\nthan the dense and fused versions", C["out"], fs=6.0)
    save(fig, "arch_tiled")


def spec():
    fig, ax = canvas(7.2, 3.3, 14.4, -0.3, 6.6)
    ax.text(7.2, 6.3, "One round of greedy speculative decoding (k = 3 shown); both caches hold c tokens, `last` is not cached", ha="center", fontsize=7.2, fontweight="bold")
    box(ax, 0.2, 4.6, 3.6, 1.2, "1. DRAFT (0.5B)\n k sequential steps:\n last -> d1 -> d2 -> d3", C["model"])
    box(ax, 4.6, 4.6, 5.0, 1.2, "2. VERIFY (1.5B), ONE forward pass\n input [last, d1, d2, d3]  (positions c .. c+3)\n -> target greedy picks p0 p1 p2 p3", C["model"])
    box(ax, 10.4, 4.6, 3.8, 1.2, "3. ACCEPT longest prefix\n d1==p0 ?  d2==p1 ?  d3==p2 ?\n m = number of matches", C["proc"])
    arrow(ax, 3.8, 5.2, 4.6, 5.2); arrow(ax, 9.6, 5.2, 10.4, 5.2)
    rows = [("all match (m = 3)", "emit d1 d2 d3 + p3 (bonus)", "target cache: keep c+4;  draft cache: feed d3, keep c+4"),
            ("first mismatch at d2 (m = 1)", "emit d1 + p1 (correction)", "target cache: crop to c+2;  draft cache: crop to c+2"),
            ("mismatch at d1 (m = 0)", "emit p0 (correction)", "target cache: crop to c+1;  draft cache: crop to c+1")]
    ax.text(0.2, 3.85, "4. SYNCHRONISE caches and continue (new `last` = last emitted token):", fontsize=7, fontweight="bold")
    for r, (a, b, c) in enumerate(rows):
        y = 2.7 - r * 1.0
        box(ax, 0.2, y, 3.8, 0.75, a, C["gray"], fs=6.5)
        box(ax, 4.3, y, 3.6, 0.75, b, C["out"], fs=6.5)
        box(ax, 8.2, y, 6.0, 0.75, c, C["cache"], fs=6.5)
        arrow(ax, 4.0, y + 0.37, 4.3, y + 0.37); arrow(ax, 7.9, y + 0.37, 8.2, y + 0.37)
    ax.text(7.2, -0.1, "Every emitted token is the target model's own greedy choice -> output identical to plain greedy decoding; 1 to k+1 tokens per target pass.", ha="center", fontsize=6.3, color="#444444")
    save(fig, "arch_spec")


if __name__ == "__main__":
    overview(); kvcache(); tiled(); spec()
