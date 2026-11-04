"""Task 3 tests + benchmarks.  python src/bench_kv.py [--adapter results/lora_r16/adapter]
Outputs: results/kv_correctness.csv, results/kv_numerics.csv, results/kv_benchmark.csv"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import load_model  # noqa: E402
from kv_decode import greedy_kv, greedy_nocache, make_prompt_of_length  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCAL = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
SMALL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def notes():
    rows = [json.loads(l) for l in open(ROOT / "data" / "val.jsonl", encoding="utf-8")]
    seen, out = set(), []
    for r in rows:
        if r["patient_id"] not in seen:
            seen.add(r["patient_id"]); out.append(r["note"])
    return out


def dump(rows, name):
    with open(ROOT / "results" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def correctness(tok, model, pool, tag):
    rows = []
    for P, N in [(32, 16), (128, 64), (256, 128), (512, 64), (700, 32)]:
        ids = make_prompt_of_length(tok, pool, P).cuda()
        a = greedy_nocache(model, ids, N)
        b = greedy_kv(model, ids, N, trace=True)
        mism = [i for i, (x, y) in enumerate(zip(a.tokens, b.tokens)) if x != y]
        rows.append(dict(model=tag, prompt_len=P, new_tokens=N, identical=(a.tokens == b.tokens),
                         first_mismatch=mism[0] if mism else -1, n_mismatch=len(mism),
                         cache_len_final=b.cache_len_trace[-1], expected_cache_len=P + N - 1,
                         cache_MiB=round(b.cache_bytes / 2**20, 2)))
        print(rows[-1], flush=True)
    return rows


@torch.no_grad()
def numerics(tok, model, pool, tag, P=256, N=64):
    """Teacher-forced comparison of logits: full forward vs prefill+incremental cached steps."""
    from transformers import DynamicCache
    ids = make_prompt_of_length(tok, pool, P).cuda()
    gen = greedy_nocache(model, ids, N).tokens
    full = torch.cat([ids, torch.tensor([gen], device="cuda")], dim=1)  # [1, P+N]
    ref = model(input_ids=full).logits[0, P - 1:P + N - 1].float()        # logits predicting gen[0..N-1]
    cache = DynamicCache()
    pos = torch.arange(P, device="cuda")
    o = model(input_ids=ids, position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)
    outs = [o.logits[0, -1].float()]
    for i in range(N - 1):
        L = cache.get_seq_length(); p = torch.tensor([L], device="cuda")
        o = model(input_ids=full[:, P + i:P + i + 1], attention_mask=torch.ones((1, L + 1), device="cuda", dtype=torch.long),
                  position_ids=p[None], cache_position=p, past_key_values=cache, use_cache=True)
        outs.append(o.logits[0, -1].float())
    got = torch.stack(outs)
    d = (ref - got).abs()
    top2 = ref.topk(2, dim=-1).values
    return dict(model=tag, dtype=str(next(model.parameters()).dtype).split(".")[-1], prompt_len=P, steps=N,
                max_abs_logit_diff=d.max().item(), mean_abs_logit_diff=d.mean().item(),
                argmax_agreement=(ref.argmax(-1) == got.argmax(-1)).float().mean().item(),
                min_top1_top2_margin=(top2[:, 0] - top2[:, 1]).min().item())


def timed(fn, reps):
    fn()  # warm-up (kernels, allocator)
    rs = [fn() for _ in range(reps)]
    return rs


def benchmark(tok, model, pool, tag, reps=3):
    rows = []
    for P, N in [(128, 32), (128, 128), (512, 32), (512, 128)]:
        ids = make_prompt_of_length(tok, pool, P).cuda()
        for name, fn in (("no_cache", lambda: greedy_nocache(model, ids, N)),
                         ("kv_cache", lambda: greedy_kv(model, ids, N))):
            rs = timed(fn, reps)
            pre = st.median(r.prefill_s for r in rs); dec = st.median(r.decode_s for r in rs)
            rows.append(dict(model=tag, mode=name, prompt_len=P, new_tokens=N, repeats=reps,
                             prefill_ms=round(pre * 1000, 1), decode_ms=round(dec * 1000, 1),
                             decode_tok_per_s=round((N - 1) / dec, 2), total_tok_per_s=round(N / (pre + dec), 2),
                             peak_mem_MiB=round(max(r.peak_mem_MiB for r in rs), 0)))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    pool = notes()
    tag = "finetuned-1.5B" if a.adapter else "base-1.5B"
    tok, model = load_model(LOCAL, a.adapter)
    corr = correctness(tok, model, pool, tag)
    num = [numerics(tok, model, pool, tag)]
    bench = benchmark(tok, model, pool, tag)
    del model; torch.cuda.empty_cache()
    # fp32 vs fp16 numerics on the small model (1.5B in fp32 does not fit in 6 GB)
    for dt in (torch.float32, torch.float16):
        _, m = load_model(SMALL, None, dt)
        num.append(numerics(tok, m, pool, "base-0.5B", P=256, N=64))
        print(num[-1], flush=True)
        if dt == torch.float32:
            corr += correctness(tok, m, pool, "base-0.5B-fp32")
        del m; torch.cuda.empty_cache()
    dump(corr, "kv_correctness.csv"); dump(num, "kv_numerics.csv"); dump(bench, "kv_benchmark.csv")
    print("DONE")
