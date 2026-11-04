# WORKLOG – what was done, in simple words

(Newest steps at the bottom. Written by Claude while working alone.)

## Setup (Nov 5)
1. **Read the assignment PDF.** It asks for: fine-tune a small LLM on clinical-trial data (TrialGPT), then build
   KV-cache decoding, tiled attention, speculative decoding, a browser app, and an IEEE LaTeX report.
2. **Made a plan** (saved in the Claude plans folder) and picked models: Qwen2.5-1.5B-Instruct (main) and
   Qwen2.5-0.5B-Instruct (small "draft" model for speculative decoding). Reason: same tokenizer, Apache-2.0
   license, and they fit in the 6 GB RTX 4050.
3. **Installed Python packages** in a private folder `.venv` (had to pin older versions of transformers/peft/
   datasets/gradio because they clashed with PyTorch 2.5.1).
4. **Downloaded the dataset** and looked at it: 1015 rows, 53 patients, 8 label types. Found that 60% of rows have
   *no* evidence sentences, and some classes are very rare (e.g. only 10 "excluded" rows in the whole dataset).
5. **Made the split** (`data/prepare_data.py`): sorted patient IDs, shuffled with numpy seed 42, 37 train / 8 val /
   8 test patients (603 / 206 / 206 rows). Saved `data/split_manifest.json`. Checked no patient is in two splits.
   Dropped all GPT-4 columns so they can never leak into the model.
6. **Wrote the prompt format + output parser** (`src/prompts.py`). Model answer format:
   `{"label": "...", "evidence": [sentence ids]}`.
7. **Wrote the metrics code** (`src/metrics.py`): accuracy, macro-F1, per-class scores, confusion matrix,
   evidence precision/recall/F1 with clear rules for empty evidence. Checked by hand on a tiny example.
8. **Measured token lengths** (`src/token_stats.py`): longest prompt is 555 tokens, so *nothing needs truncating*.
9. **Models were hard to download** (slow, stalled connection). Fixed by using resumable `curl`
   (`results/dl_curl.sh`). Both models are now in `models/`.
10. **Task 4 (attention) done** (`src/attention.py`, `src/test_attention.py`): normal attention vs tiled
    "online softmax" attention vs PyTorch's built-in. All agree to ~1e-6 (float32). Tiled uses far less memory but
    is slower in plain PyTorch (explained in the report). Results: `results/attention_*.csv`.
11. **Zero-shot baseline ("before" fine-tuning)** on validation: accuracy 0.068, macro-F1 0.081 (very bad: it says
    "excluded" almost always and never says "not enough information").
12. **LoRA fine-tuning started** (`src/train_lora.py`). First attempt got stuck because my loss code built a huge
    logits tensor that didn't fit in 6 GB; fixed by computing the output layer only on the answer tokens.
    Second attempt runs fine (peak 3.9 GB, ~16 s per step, ~35 min in total).

## Next steps (will be appended below as they happen)
