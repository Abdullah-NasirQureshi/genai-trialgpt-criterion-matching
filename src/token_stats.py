"""Token-length statistics for the full prompt (+ gold completion) per split.

Documents truncation: we compare against the model's context window (32768 for
Qwen2.5-1.5B-Instruct) and against our training max length (MAX_LEN), and count how
many examples would lose patient evidence sentences if truncated at MAX_LEN.
Writes results/token_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import encode_prompt, split_sentences, target_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 1024  # planned training/inference cap; chosen from these statistics

tok = AutoTokenizer.from_pretrained(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
draft_tok = AutoTokenizer.from_pretrained(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
out = {"max_len_cap": MAX_LEN, "context_window_model": 32768,
       "tokenizers_identical_vocab": tok.get_vocab() == draft_tok.get_vocab(), "splits": {}}
for split in ("train", "val", "test"):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{split}.jsonl", encoding="utf-8")]
    plen, tlen, nlen = [], [], []
    over = 0
    for r in rows:
        sents = split_sentences(r["note"])
        p = encode_prompt(tok, sents, r["criterion_type"], r["criterion_text"])
        pl = len(tok(p).input_ids)
        tl = len(tok(target_text(r["expert_eligibility"], r["expert_sentences"])).input_ids)
        plen.append(pl); tlen.append(tl); nlen.append(len(tok(r["note"]).input_ids))
        over += (pl + tl) > MAX_LEN
    a = np.array(plen)
    out["splits"][split] = {
        "n": len(rows), "prompt_tokens_mean": float(a.mean()), "prompt_tokens_p95": float(np.percentile(a, 95)),
        "prompt_tokens_max": int(a.max()), "note_tokens_max": int(max(nlen)),
        "target_tokens_mean": float(np.mean(tlen)), "target_tokens_max": int(max(tlen)),
        "examples_exceeding_cap": int(over)}
json.dump(out, open(ROOT / "results" / "token_stats.json", "w"), indent=2)
print(json.dumps(out, indent=2))
