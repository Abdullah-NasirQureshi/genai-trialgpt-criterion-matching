"""Task 5 tests + benchmarks.
python src/bench_spec.py --target-adapter results/lora_r16/adapter [--draft-adapter results/lora_draft/adapter]
Outputs results/spec_correctness.csv, results/spec_benchmark.csv"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_model import load_model  # noqa: E402
from kv_decode import greedy_kv, make_prompt_of_length  # noqa: E402
from prompts import encode_prompt, split_sentences  # noqa: E402
from speculative import speculative_greedy  # noqa: E402
from bench_kv import notes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIG = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
SMALL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def task_prompts(tok, n):
    rows = [json.loads(l) for l in open(ROOT / "data" / "val.jsonl", encoding="utf-8")]
    rows = rows[::max(1, len(rows) // n)][:n]
    return [tok(encode_prompt(tok, split_sentences(r["note"]), r["criterion_type"], r["criterion_text"]),
                return_tensors="pt", add_special_tokens=False).input_ids.cuda() for r in rows]


def run_workload(name, target, draft, prompts, N, eos, ks, reps, rows, corr_rows, tag):
    # --- plain greedy (cached) baseline: reference tokens + timing
    ref = []
    for ids in prompts:
        ref.append(greedy_kv(target, ids, N, eos_id=eos))
    for _ in range(1):  # warm-up already done by the line above
        pass
    def time_plain():
        ts, toks = [], 0
        for ids in prompts:
            r = greedy_kv(target, ids, N, eos_id=eos)
            ts.append(r.prefill_s + r.decode_s); toks += len(r.tokens)
        return sum(ts), toks
    plain = [time_plain() for _ in range(reps)]
    sec = st.median(p[0] for p in plain); toks = plain[0][1]
    pk = max(r.peak_mem_MiB for r in ref)
    rows.append(dict(tag=tag, workload=name, method="greedy_kv", k=0, prompts=len(prompts), tokens=toks,
                     seconds=round(sec, 3), tok_per_s=round(toks / sec, 2), speedup=1.0, acceptance_rate="",
                     tokens_per_target_forward="", peak_mem_MiB=round(pk)))
    print(rows[-1], flush=True)
    for k in ks:
        eq = 0; first_mis = []
        res = [speculative_greedy(target, draft, ids, N, k=k, eos_id=eos) for ids in prompts]
        for r, g in zip(res, ref):
            same = r.tokens == g.tokens
            eq += same
            if not same:
                first_mis.append(next(i for i, (x, y) in enumerate(zip(r.tokens, g.tokens)) if x != y)
                                 if any(x != y for x, y in zip(r.tokens, g.tokens)) else min(len(r.tokens), len(g.tokens)))
        corr_rows.append(dict(tag=tag, workload=name, k=k, prompts=len(prompts), identical=eq,
                              identical_frac=round(eq / len(prompts), 3), first_mismatch_positions=str(first_mis[:10])))
        print(corr_rows[-1], flush=True)
        def time_spec():
            rs = [speculative_greedy(target, draft, ids, N, k=k, eos_id=eos) for ids in prompts]
            return rs, sum(r.seconds for r in rs)
        runs = [time_spec() for _ in range(reps)]
        sec_s = st.median(x[1] for x in runs); rs = runs[0][0]
        drafted = sum(r.drafted for r in rs); acc = sum(r.accepted for r in rs)
        rounds = sum(r.rounds for r in rs); toks_s = sum(len(r.tokens) for r in rs)
        rows.append(dict(tag=tag, workload=name, method="speculative", k=k, prompts=len(prompts), tokens=toks_s,
                         seconds=round(sec_s, 3), tok_per_s=round(toks_s / sec_s, 2), speedup=round(sec / sec_s, 3),
                         acceptance_rate=round(acc / max(1, drafted), 3),
                         tokens_per_target_forward=round((acc + rounds) / max(1, rounds), 2),
                         peak_mem_MiB=round(max(r.peak_mem_MiB for r in rs))))
        print(rows[-1], flush=True)


def hf_reference_check(target, draft, prompts, N, eos):
    """Our speculative output vs HF built-in assisted generation (reference test only)."""
    same = 0
    for ids in prompts:
        ours = speculative_greedy(target, draft, ids, N, k=4, eos_id=eos).tokens
        g = target.generate(ids, attention_mask=torch.ones_like(ids), assistant_model=draft, do_sample=False,
                            max_new_tokens=N, eos_token_id=eos, pad_token_id=eos)[0, ids.shape[1]:].tolist()
        if eos in g:
            g = g[:g.index(eos) + 1]
        same += (ours == g[:len(ours)] and len(ours) == len(g))
    return same, len(prompts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-adapter", default=None)
    ap.add_argument("--draft-adapter", default=None)
    ap.add_argument("--tag", default="spec")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    tok, target = load_model(BIG, a.target_adapter)
    _, draft = load_model(SMALL, a.draft_adapter)
    assert tok.get_vocab() == __import__("transformers").AutoTokenizer.from_pretrained(SMALL).get_vocab()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    rows, corr = [], []
    tp = task_prompts(tok, a.n_prompts)
    run_workload("task_answers(EOS stop,N=48)", target, draft, tp, 48, eos, [2, 4, 6, 8], a.reps, rows, corr, a.tag)
    pool = notes()
    lp = [make_prompt_of_length(tok, pool, 128).cuda() for _ in range(4)]
    run_workload("long_free_generation(N=128,no EOS)", target, draft, lp, 128, None, [2, 4, 8], a.reps, rows, corr, a.tag)
    same, n = hf_reference_check(target, draft, tp[:6], 48, eos)
    corr.append(dict(tag=a.tag, workload="HF assisted_generation reference", k=4, prompts=n, identical=same,
                     identical_frac=round(same / n, 3), first_mismatch_positions=""))
    print(corr[-1])
    for name, data in (("spec_benchmark", rows), ("spec_correctness", corr)):
        with open(ROOT / "results" / f"{name}_{a.tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print("DONE")
