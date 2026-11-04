# Defense notes - what to be able to explain (and where in the code)

Simple, short answers. Numbers are from the final runs in `results/`.

## 1. The task and data
* **Task:** given a patient note (numbered sentences) + one criterion (inclusion/exclusion) -> label (8 expert categories) + cited sentence IDs.
* **Data:** TrialGPT Criterion Annotations, 1015 rows, 53 patients. **Split:** sort patient IDs, `default_rng(42)` shuffle, 37/8/8 (`data/prepare_data.py`). Manifest in `data/split_manifest.json`.
* **No leakage:** GPT-4 columns dropped in `prepare_data.py`; only note, criterion, expert label/evidence remain. Test run once (`eval_model.py` refuses test without `--allow-test`).
* **Quirks:** 60% of rows have empty gold evidence; test has no "excluded" example; strong class imbalance.

## 2. Model and fine-tuning
* **Qwen2.5-1.5B-Instruct:** 28 layers, hidden 1536, 12 query heads / 2 KV heads (GQA), RoPE, 32k context, Apache-2.0, 3.1 GB in fp16. Fits 6 GB.
* **LoRA:** freeze the base, add low-rank matrices A,B to every linear layer: W + (alpha/r)BA, r=16 -> 18.5M trainable params (1.18%).
* **Loss:** next-token cross-entropy on the answer tokens only (prompt masked). `train_lora.py: answer_loss` computes the 152k-way output layer only at answer positions (saves memory).
* **Why balanced oversampling:** plain run collapsed to "not excluded" on exclusion criteria. `--balance` repeats rare labels (sqrt rule).
* **Results:** test acc 0.058 -> 0.718, macro-F1 0.084 -> 0.604; exclusion accuracy (0.679) is slightly below the majority baseline (0.695) - say this yourself.

## 3. KV cache (`src/kv_decode.py`)
* **Idea:** attention at step t needs K,V of all earlier tokens; they never change, so store them. Each step feeds only the new token.
* **What the code does:** `greedy_nocache` re-runs the whole sequence each step. `greedy_kv` prefills once with a `DynamicCache`, then passes `past_key_values`, the exact `position_ids`/`cache_position` (= cache length) and an all-ones mask of length cache+1.
* **Positions:** RoPE rotates q,k by the absolute position; cached keys are already rotated, the new token must get its true position.
* **Masking:** prefill uses the causal triangle; a single new query is the last position, so it may see everything cached.
* **Cache growth:** +1 K and V per layer per token; for this model 28 KiB/token.
* **Numerical differences:** fp32 identical; fp16 two settings diverge because top-2 logits can be exactly tied (min gap 0.0) and the two code paths round differently (`results/kv_numerics.csv`).
* **Numbers:** 2.2x (128-token prompt) to 7x (512) faster decode; memory unchanged (weights dominate).

## 4. Tiled attention (`src/attention.py`)
* **Reference:** softmax(QK^T/sqrt(d))V with causal mask and row-max subtraction.
* **Tiled online softmax:** per query block keep running max m, normaliser l, output o; for a new key block: m' = max(m, m_tile); alpha = exp(m - m'); l' = alpha*l + sum exp(S - m'); o' = alpha*o + exp(S - m')V; at the end o/l. Exact, never builds LxL.
* **Causal skip:** key blocks entirely in the future are skipped; fully masked rows stay at -inf (guard against inf-inf).
* **Results:** matches float64 to ~4e-5 (fp32); memory 59-158x lower at L=4096; **slower** than dense and 18-303x slower than the fused kernel, because many small kernel launches and no SRAM tiling.

## 5. Speculative decoding (`src/speculative.py`)
* **Round:** draft proposes k tokens -> target verifies [last, d1..dk] in one pass -> accept longest matching prefix m -> emit d1..dm and the target's own next token (correction or bonus) -> crop target cache to c+m+1; draft cache cropped too (if m==k feed d_k first).
* **Why output equals greedy:** every emitted token is the target's argmax given the same prefix.
* **Results:** identical outputs everywhere (and equal to HF assisted generation); acceptance 90-95% (tuned draft) but speed 0.71-0.82x. **Why:** draft step 57 ms vs target 71 ms (c=0.81) - eager PyTorch per-layer overhead dominates; draft must also prefill; short answers (16 tokens). Untuned draft is worse (acceptance 0.62, speed 0.49-0.63x).

## 6. App (`app/app.py`)
* Gradio; models load lazily; input validation in `validate()`; decoding mode chosen per run; overall verdict = rules in `aggregate()` (not ground truth); explanation = template quoting only cited sentences; report = HTML with provenance.
* Test: `python src/test_app.py` (18 checks).

## 7. Likely "change this live" requests
| Request | Where |
|---|---|
| Change draft length k | slider in app; `k=` in `speculative_greedy` |
| Change tile size | `block_q`, `block_k` in `tiled_attention` |
| Add a new criterion / note | paste in the app |
| Evaluate another split/adapter | `python src/eval_model.py --split val --adapter <dir> --tag x` |
| Show cache growth | `greedy_kv(..., trace=True)` returns `cache_len_trace` |
| Change aggregation rule | `aggregate()` in `app/app.py` |
| Change evidence empty-set rule | `evidence_row_scores` in `src/metrics.py` |

## 8. Weaknesses to admit (shows understanding)
8 test patients -> wide intervals; single seed; only Qwen trained; exclusion NEI recall 3/28; explanations are templates; timings from one laptop GPU; speculative slowdown is hardware specific.
