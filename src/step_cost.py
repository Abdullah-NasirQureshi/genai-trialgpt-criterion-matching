"""Measure the per-step cost of target and draft and predict the speculative-decoding speedup.

For one round with draft length k:
    time(round)  = k * t_draft(1 token) + t_target(k+1 tokens)        [+ cache crop ~0]
    tokens(round)= measured tokens per target forward (accepted + 1)
    predicted speedup = tokens(round) * t_target(1 token) / time(round)
Writes results/spec_step_cost.csv.   python src/step_cost.py
"""
import csv
import statistics as st
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import load_model  # noqa: E402
from kv_decode import make_prompt_of_length  # noqa: E402
from bench_kv import notes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def step_times(model, ids, ks, reps=15):
    """median ms for an n-token forward at the end of a cached prompt (n=1 and n=k+1)."""
    P = ids.shape[1]
    cache = DynamicCache()
    pos = torch.arange(P, device="cuda")
    model(input_ids=ids, position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
    out = {}
    for n in sorted({1, *[k + 1 for k in ks]}):
        x = ids[:, -n:]
        ts = []
        for r in range(reps + 3):
            p = torch.arange(P, P + n, device="cuda")
            sync(); t0 = time.perf_counter()
            model(input_ids=x, attention_mask=torch.ones((1, P + n), device="cuda", dtype=torch.long),
                  position_ids=p[None], cache_position=p, past_key_values=cache, use_cache=True)
            sync(); ts.append((time.perf_counter() - t0) * 1000)
            cache.crop(P)
        out[n] = st.median(ts[3:])
    return out


if __name__ == "__main__":
    ks = [2, 4, 6, 8]
    pool = notes()
    tok, tgt = load_model(str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"), str(ROOT / "results" / "lora_final" / "adapter"))
    ids = make_prompt_of_length(tok, pool, 256).cuda()
    tt = step_times(tgt, ids, ks)
    del tgt; torch.cuda.empty_cache()
    _, drf = load_model(str(ROOT / "models" / "Qwen2.5-0.5B-Instruct"), str(ROOT / "results" / "lora_draft" / "adapter"))
    td = step_times(drf, ids, ks)
    c = td[1] / tt[1]
    print(f"target 1-token {tt[1]:.1f} ms, draft 1-token {td[1]:.1f} ms, cost ratio c = {c:.2f}")
    meas = {}
    p = ROOT / "results" / "spec_benchmark_final.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["method"] == "speculative" and r["workload"].startswith("task"):
                meas[int(r["k"])] = (float(r["acceptance_rate"]), float(r["tokens_per_target_forward"]), float(r["speedup"]))
    rows = []
    for k in ks:
        t_round = k * td[1] + tt[k + 1]
        a, tpr, sp = meas.get(k, (float("nan"),) * 3)
        rows.append(dict(k=k, target_1tok_ms=round(tt[1], 2), target_verify_ms=round(tt[k + 1], 2),
                         draft_1tok_ms=round(td[1], 2), cost_ratio_c=round(c, 3), round_ms=round(t_round, 1),
                         measured_acceptance=a, measured_tokens_per_round=tpr,
                         predicted_speedup=round(tpr * tt[1] / t_round, 3) if tpr == tpr else "",
                         measured_speedup=sp))
        print(rows[-1])
    with open(ROOT / "results" / "spec_step_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
