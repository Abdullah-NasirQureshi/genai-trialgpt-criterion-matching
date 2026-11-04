"""Evaluate a base model (optionally + LoRA adapter) on a split with greedy decoding.

Usage:
  python src/eval_model.py --split val --tag base_zeroshot
  python src/eval_model.py --split val --adapter results/lora_run1/adapter --tag lora_run1
Writes results/eval_<tag>_<split>.json (+ .preds.jsonl with every prediction).
The test split is refused unless --allow-test is passed (final evaluation only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import encode_prompt, parse_output, split_sentences  # noqa: E402
from metrics import full_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-1.5B-Instruct")  # downloaded via results/dl_curl.sh


def load_split(name):
    return [json.loads(l) for l in open(ROOT / "data" / f"{name}.jsonl", encoding="utf-8")]


def load_model(model_name, adapter=None, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).cuda().eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return tok, model


@torch.no_grad()
def run(rows, tok, model, batch_size=8, max_new_tokens=48):
    out = []
    prompts = [encode_prompt(tok, split_sentences(r["note"]), r["criterion_type"], r["criterion_text"])
               for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # length-bucketed batches
    texts = {}
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            texts[i] = tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
    for i, r in enumerate(rows):
        n_sent = len(split_sentences(r["note"]))
        label, ev, ok = parse_output(texts[i], r["criterion_type"], n_sent)
        out.append({"annotation_id": r["annotation_id"], "patient_id": r["patient_id"],
                    "criterion_type": r["criterion_type"], "gold_label": r["expert_eligibility"],
                    "gold_evidence": r["expert_sentences"], "pred_label": label,
                    "pred_evidence": ev, "valid": ok, "raw_output": texts[i]})
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("Refusing to evaluate on the test split without --allow-test (final evaluation only).")

    rows = load_split(a.split)[: a.limit]
    tok, model = load_model(a.model, a.adapter)
    preds, secs = run(rows, tok, model, a.batch_size)
    rep = full_report(preds)
    rep["meta"] = {"model": a.model, "adapter": a.adapter, "split": a.split, "n": len(preds),
                   "decoding": "greedy, max_new_tokens=48, fp16, batch=%d, left-padded" % a.batch_size,
                   "seconds": secs, "invalid_outputs": sum(not p["valid"] for p in preds)}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    json.dump(rep, open(res / f"eval_{a.tag}_{a.split}.json", "w"), indent=2)
    with open(res / f"eval_{a.tag}_{a.split}.preds.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    c = rep["overall"]["classification"]
    e = rep["overall"]["evidence"]
    print(f"[{a.tag}/{a.split}] n={len(preds)} acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} "
          f"invalid={rep['meta']['invalid_outputs']} "
          f"evidence microF1(all)={e['all_rows'].get('micro_f1', float('nan')):.3f} "
          f"microF1(gold-only)={e['rows_with_gold_evidence'].get('micro_f1', float('nan')):.3f} "
          f"time={secs:.0f}s")


if __name__ == "__main__":
    main()
