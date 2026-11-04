"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the TrialGPT train split.

Objective : next-token cross-entropy on the answer tokens only (prompt tokens masked
            with -100), answer = '{"label": "...", "evidence": [...]}' + <|im_end|>.
Selection : best validation loss (val split only; test is never read here).
Memory    : fp16 frozen base + fp32 LoRA weights, autocast fp16, gradient checkpointing.
Outputs   : results/<run>/{config.json, train_log.csv, adapter/ (best), adapter_last/}
Usage     : python src/train_lora.py --run lora_r16 --epochs 3 --lr 2e-4 --rank 16
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import encode_prompt, split_sentences, target_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")  # default; override with --model
MAX_LEN = 1024  # no example exceeds 555 tokens (results/token_stats.json) -> no truncation


def build_examples(split, tok):  # uses the global MAX_LEN
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{split}.jsonl", encoding="utf-8")]
    ex = []
    for r in rows:
        p = encode_prompt(tok, split_sentences(r["note"]), r["criterion_type"], r["criterion_text"])
        a = target_text(r["expert_eligibility"], r["expert_sentences"]) + "<|im_end|>"
        pi = tok(p, add_special_tokens=False).input_ids
        ai = tok(a, add_special_tokens=False).input_ids
        ids = (pi + ai)[:MAX_LEN]
        labels = ([-100] * len(pi) + ai)[:MAX_LEN]
        ex.append((ids, labels))
    return ex


def collate(batch, pad_id):
    L = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):  # right padding for training
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()



def answer_loss(model, ids, lab, att, reduction="mean"):
    """Cross-entropy on answer tokens only. Runs the transformer body, then applies the
    (huge, 151936-way) output layer ONLY at positions that predict an answer token, so we
    never materialise [batch, seq, vocab] logits (would not fit in 6 GB)."""
    inner = model.base_model.model  # Qwen2ForCausalLM
    with torch.autocast("cuda", dtype=torch.float16):
        h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
        tgt = lab[:, 1:]
        sel = tgt != -100
        logits = inner.lm_head(h[sel])
    return torch.nn.functional.cross_entropy(logits.float(), tgt[sel], reduction=reduction), int(sel.sum())

@torch.no_grad()
def eval_loss(model, data, pad_id, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(data), bs):
        ids, lab, att = collate(data[s:s + bs], pad_id)
        loss, cnt = answer_loss(model, ids, lab, att, reduction="sum")
        tot += loss.item()
        n += cnt
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    model_path = a.model

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = ROOT / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(a), "model": Path(a.model).name, "max_len": MAX_LEN,
               "target_modules": "q,k,v,o,gate,up,down", "precision": "fp16 base + fp32 LoRA, autocast"},
              open(out / "config.json", "w"), indent=2)

    tok = AutoTokenizer.from_pretrained(model_path)
    train, val = build_examples("train", tok), build_examples("val", tok)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 trainable weights, fp16 autocast compute
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = (len(train) + a.micro_bs * a.grad_accum - 1) // (a.micro_bs * a.grad_accum)
    total = steps_per_epoch * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + np.cos(np.pi * min(1.0, s / total)))))
    scaler = torch.amp.GradScaler('cuda')

    log = open(out / "train_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "gpu_peak_GiB", "elapsed_s"])
    best, step, t0 = float("inf"), 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    pad = tok.pad_token_id
    v0 = eval_loss(model, val, pad)
    w.writerow([0, 0, "", f"{v0:.4f}", a.lr, "", 0]); log.flush()
    print(f"step 0 val_loss {v0:.4f}")

    for ep in range(a.epochs):
        order = np.random.permutation(len(train)).tolist()
        mb = a.micro_bs
        chunks = [order[i:i + mb] for i in range(0, len(order), mb)]
        run_loss, run_n = 0.0, 0
        for ci, chunk in enumerate(chunks):
            ids, lab, att = collate([train[i] for i in chunk], pad)
            loss, _ = answer_loss(model, ids, lab, att)
            scaler.scale(loss / a.grad_accum).backward()
            run_loss += loss.item(); run_n += 1
            if (ci + 1) % a.grad_accum == 0 or ci == len(chunks) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step % 5 == 0:
                    w.writerow([step, f"{ep + (ci + 1) / len(chunks):.2f}", f"{run_loss / run_n:.4f}", "",
                                f"{sched.get_last_lr()[0]:.2e}", f"{torch.cuda.max_memory_allocated() / 2**30:.2f}",
                                f"{time.time() - t0:.0f}"]); log.flush()
                    run_loss, run_n = 0.0, 0
        vl = eval_loss(model, val, pad)
        w.writerow([step, ep + 1, "", f"{vl:.4f}", f"{sched.get_last_lr()[0]:.2e}",
                    f"{torch.cuda.max_memory_allocated() / 2**30:.2f}", f"{time.time() - t0:.0f}"]); log.flush()
        print(f"epoch {ep + 1} step {step} val_loss {vl:.4f} peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"elapsed {time.time() - t0:.0f}s", flush=True)
        if vl < best:
            best = vl
            model.save_pretrained(out / "adapter")
            print("  saved best adapter", flush=True)
    model.save_pretrained(out / "adapter_last")
    json.dump({"best_val_loss": best, "base_val_loss": v0, "train_seconds": time.time() - t0,
               "gpu_peak_GiB": torch.cuda.max_memory_allocated() / 2**30}, open(out / "summary.json", "w"), indent=2)
    print("done; best val loss", best)


if __name__ == "__main__":
    main()
