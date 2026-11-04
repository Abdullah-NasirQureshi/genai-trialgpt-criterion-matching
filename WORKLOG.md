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
13. **Git history (local).** You asked me to push to GitHub as I go. I made a local git repo and committed the work in
    small logical steps (see `git log`). **Creating the GitHub repo and pushing was blocked by Claude Code's
    permission check, so I did NOT push** and did not try to get around it. To publish later yourself:
    `gh repo create <name> --public --source=. --push` (or add a remote and `git push -u origin main`).
    Large files (models, adapters, .venv) are git-ignored; the adapter should go in a GitHub Release / HF Hub.
14. Added a `--model` option to `train_lora.py` so the small 0.5B "draft" model can also be fine-tuned later
    (a draft that knows the task agrees with the big model more often, which helps speculative decoding).
15. **Wrote the KV-cache decoder** (`src/kv_decode.py`): a hand-written loop. "No-cache" re-reads the whole text for every
    new token; "KV-cache" reads the prompt once, then only feeds the newest token, passing the cache, the exact
    position number and the attention mask ourselves. `src/bench_kv.py` will prove both give identical tokens at
    5 length settings, compare logits (fp16 vs fp32) and time prefill/decode/tokens-per-second/peak memory.
16. **Wrote the speculative decoder** (`src/speculative.py`): small model guesses k tokens, big model checks them all in
    one pass, we keep the matching ones, fix the first wrong one, and trim both caches so they stay in sync.
    `src/bench_spec.py` checks it matches normal greedy output, also against Hugging Face's own assisted generation.
17. LoRA training epoch 1 finished: validation loss dropped from 0.658 to 0.148 (good sign).
18. **GitHub connected (you asked for it).** The earlier push was blocked, but after your explicit request I created the
    **private** repo https://github.com/Abdullah-NasirQureshi/genai-trialgpt-criterion-matching and pushed all
    commits. (Entry 13 above is now outdated.) It is private: **make it public before submitting** so graders can open it
    (GitHub > Settings > Danger zone > Change visibility). I will keep pushing after each logical commit.
19. **LoRA run 1 finished** (rank 16, lr 2e-4, 3 epochs, 35 min, 3.9 GB GPU). Best validation loss 0.1333 at epoch 2
    (0.658 before training). On the validation split: accuracy 0.068 -> **0.728**, macro-F1 0.081 -> **0.467**.
    Problem found: for exclusion criteria the model says "not excluded" 139 times out of 140 (same as always guessing the
    majority class) and never predicts "not enough information" / "not applicable"; it also misses all "not included".
20. **Trying one fix: class-balanced training** (`--balance`: repeat rare labels, e.g. "excluded" x6). Trained on train
    only, judged on validation only, 2 epochs (run `lora_bal`). The test set is still untouched.
21. Wrote `src/error_analysis.py`: picks diverse mistakes (false certainty, missed information, flipped polarity,
    unsupported answers, wrong evidence) and prints them with the patient sentences quoted. I will run it on the
    TEST predictions at the end and write the explanations by hand in the paper.
22. **Started the paper** (`paper/`): bibliography with 16 real references (`refs.bib`), the Introduction + Related Work
    (`sec_intro_related.tex`) and the full Materials & Methods (`sec_methods.tex`: dataset/split, model comparison table,
    prompt format, LoRA setup, KV-cache, tiled attention equations, speculative algorithm, app workflow). Architecture
    numbers (layers, heads, KV bytes per token) were read from the real config files. Results sections come after the
    experiments finish.
23. **No LaTeX compiler on this PC.** I tried to download a portable one (Tectonic); Claude Code's permission check
    blocked running a downloaded program, so I did not do it. The paper stays as `.tex` source in `paper/` -- upload
    the `paper/` folder to Overleaf (it compiles with the IEEEtran class, which Overleaf has) to get the PDF.
    I will check the .tex files for syntax mistakes with my own script instead. `main.tex` has TODO markers for your
    name, university, student ID, video link.
24. Wrote `README.md` (setup, start the app, reproduce every experiment, repo map) and `src/check_tex.py`, a checker that
    catches LaTeX mistakes without a compiler (unbalanced braces, missing citations/labels, figures never discussed).
25. **LoRA run 2 (class-balanced) finished** (rank 16, 2 epochs, 31 min). Validation: accuracy 0.733, macro-F1 **0.534**
    (run 1: 0.728 / 0.467). The balanced model now predicts "not enough information" and "not applicable" for exclusion
    criteria (exclusion macro-F1 0.377 -> 0.567); inclusion is unchanged. Chosen as the **final adapter** using
    VALIDATION only (copied to `results/lora_final/`). Test set still untouched.
26. Started fine-tuning the small 0.5B **draft** model with the same recipe (it will agree with the big model more often
    than the untuned draft, which is what speculative decoding needs).
27. **Draft model fine-tuned** (0.5B, balanced, 2 epochs, 11 min, only 1.5 GB GPU; val loss 0.1536).
28. **KV-cache tests and benchmark done** (`results/kv_*.csv`).
    * Speed: with the cache the 1.5B model decodes ~13.6-16 tokens/s; without it only 5.6-6.3 (128-token prompt) and
      2.3 (512-token prompt) -> 2.2x to 7x faster. Memory is the same (~3.0 GB, the cache only adds 5-20 MiB).
    * Correctness: in fp32 (0.5B model) cached and uncached output are **identical at all 5 length settings** (logits differ by
      only 3e-5). In fp16 two of five settings diverge after ~25-44 tokens; I traced this to rounding: the two top
      logits are sometimes exactly tied in fp16 (smallest gap 0.0), so the tiny numeric difference between the two
      code paths can flip the greedy choice. Not a bug.
