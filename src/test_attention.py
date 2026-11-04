"""Correctness + benchmark for Task 4. Run:  python src/test_attention.py
Writes results/attention_correctness.csv and results/attention_benchmark.csv."""
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention import reference_attention, tiled_attention, sdpa_attention  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def err(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def correctness():
    rows = []
    torch.manual_seed(0)
    # (B,H,Lq,Lk,D, causal, block_q, block_k, dtype)
    cases = [
        (1, 1, 5, 5, 8, True, 2, 2, torch.float32),     # tiny, hand-checkable, uneven tiles
        (1, 2, 7, 7, 16, False, 3, 4, torch.float32),   # non-causal
        (2, 4, 64, 64, 32, True, 16, 16, torch.float32),
        (2, 4, 100, 100, 64, True, 32, 48, torch.float32),   # non-divisible lengths
        (1, 8, 257, 257, 64, True, 64, 64, torch.float32),
        (1, 4, 1, 129, 64, True, 64, 64, torch.float32),     # single-token decode vs cache
        (1, 4, 16, 144, 64, True, 8, 32, torch.float32),     # Lq < Lk (cached prefill chunk)
        (1, 4, 128, 128, 64, True, 64, 64, torch.float16),
        (1, 4, 128, 128, 64, True, 64, 64, torch.bfloat16),
        (1, 2, 64, 64, 32, True, 16, 16, torch.float32),     # large-magnitude scores (stability)
    ]
    for n, (B, H, Lq, Lk, D, causal, bq, bk, dt) in enumerate(cases):
        q = torch.randn(B, H, Lq, D, device=dev, dtype=dt)
        k = torch.randn(B, H, Lk, D, device=dev, dtype=dt)
        v = torch.randn(B, H, Lk, D, device=dev, dtype=dt)
        if n == len(cases) - 1:
            q, k = q * 30, k * 30  # scores in the hundreds -> naive exp() would overflow
        # independent float64 ground truth (so the 'reference' row is not a self-comparison)
        ref32 = reference_attention(q.double(), k.double(), v.double(), causal)
        r = reference_attention(q, k, v, causal)
        t = tiled_attention(q, k, v, causal, bq, bk)
        s = sdpa_attention(q, k, v, causal)
        for name, o in (("reference", r), ("tiled", t), ("sdpa", s)):
            mx, mn = err(o, ref32)
            rows.append(dict(case=n, B=B, H=H, Lq=Lq, Lk=Lk, D=D, causal=causal, block_q=bq,
                             block_k=bk, dtype=str(dt).split(".")[-1], impl=name,
                             max_abs_err_vs_fp64_ref=mx, mean_abs_err=mn,
                             has_nan=bool(torch.isnan(o).any())))
    return rows


def timeit(fn, reps=10):
    for _ in range(3):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def benchmark():
    rows = []
    B, H, D = 1, 12, 64
    for L in (256, 512, 1024, 2048, 4096):
        q, k, v = (torch.randn(B, H, L, D, device=dev, dtype=torch.float16) for _ in range(3))
        for name, fn in (("reference", lambda: reference_attention(q, k, v, True)),
                         ("tiled_64", lambda: tiled_attention(q, k, v, True, 64, 64)),
                         ("tiled_256", lambda: tiled_attention(q, k, v, True, 256, 256)),
                         ("sdpa", lambda: sdpa_attention(q, k, v, True))):
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            try:
                ms = timeit(fn)
                peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev == "cuda" else float("nan")
            except torch.cuda.OutOfMemoryError:
                ms, peak = float("nan"), float("nan")
            rows.append(dict(L=L, heads=H, head_dim=D, dtype="float16", impl=name,
                             median_ms=round(ms, 3), peak_extra_MiB=round(peak, 1)))
    return rows


def dump(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    c = correctness()
    dump(c, ROOT / "results" / "attention_correctness.csv")
    print(f"{'case':>4} {'dtype':>9} {'Lq':>4} {'Lk':>4} {'impl':>9} {'max_err':>10} {'mean_err':>10}")
    for r in c:
        print(f"{r['case']:>4} {r['dtype']:>9} {r['Lq']:>4} {r['Lk']:>4} {r['impl']:>9} "
              f"{r['max_abs_err_vs_fp64_ref']:>10.2e} {r['mean_abs_err']:>10.2e}  nan={r['has_nan']}")
    b = benchmark()
    dump(b, ROOT / "results" / "attention_benchmark.csv")
    print("\nBenchmark (device=%s)" % dev)
    for r in b:
        print(r)
