# Clinical-trial criterion matching with a fine-tuned small Transformer

GenAI Assignment 2. A Qwen2.5-1.5B-Instruct model (LoRA fine-tuned on the TrialGPT Criterion Annotations) that, given a
patient note and one trial inclusion/exclusion criterion, returns an eligibility label (8 expert categories) and the
supporting sentence IDs. The repo also contains from-scratch implementations of **KV-cached decoding**, **tiled
online-softmax attention** and **greedy speculative decoding**, and a **browser app** that ties them together.

> **Educational screening prototype - not a clinical enrollment decision tool.**

Tested on: Windows 11, Python 3.11, RTX 4050 Laptop GPU (6 GB), CUDA 12.1. Work log of every step: [WORKLOG.md](WORKLOG.md).
Other documents: IEEE paper source in [paper/](paper/) (compile `main.tex` on Overleaf or with `pdflatex`+`bibtex`),
[DEMO_SCRIPT.md](DEMO_SCRIPT.md) (video plan), [DEFENSE_NOTES.md](DEFENSE_NOTES.md) (what to explain).

## Results at a glance (held-out test split: 8 patients, 206 criteria; all numbers in `results/`)
| | Accuracy | Macro-F1 | Evidence micro-F1 (all rows / rows with gold evidence) |
|---|---|---|---|
| Majority label per criterion type | 0.626 | 0.272 | 0.000 / 0.000 |
| Zero-shot Qwen2.5-1.5B-Instruct | 0.058 | 0.084 | 0.122 / 0.398 |
| **LoRA fine-tuned (class-balanced)** | **0.718** | **0.604** | **0.531 / 0.636** |

Weak spot (be aware): on exclusion criteria the model's accuracy (0.679) is slightly below always answering "not excluded" (0.695), and it
finds only 3 of 28 "not enough information" exclusion cases. The test split has no "excluded" example.

| Mechanism | Result |
|---|---|
| KV cache vs no cache (1.5B, fp16) | 2.2x (128-token prompt) to 7x (512) faster decoding, same memory; identical tokens in fp32, 2 of 5 fp16 settings diverge because of exact logit ties |
| Tiled online-softmax attention | matches a float64 reference to 4e-5 (fp32); 59-158x less memory at L=4096, but slower than dense (34x) and fused SDPA (up to 303x) in plain PyTorch |
| Greedy speculative decoding (0.5B draft) | output identical to greedy; 90-95% acceptance; **0.71-0.82x speed** (slower; draft is only 19% cheaper per step) |

## 1. Setup
```bash
python -m venv .venv
.venv\Scripts\activate                      # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Download the models (≈ 4 GB, one time)
Base models from Hugging Face (Apache-2.0). Any download method works; this one resumes after interruptions:
```bash
bash results/dl_curl.sh        # downloads Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct into models/
```
### Download the trained LoRA adapters
The adapters are attached to the GitHub release of this repository (`adapters.zip`). Unzip into `results/`:
```
results/lora_final/adapter/   # target model adapter (1.5B)
results/lora_draft/adapter/   # draft model adapter (0.5B)
```
Or retrain them (steps below). Download with `gh` (inside this repo) and unzip with Python (works on every OS):
```bash
gh release download v1.0 --pattern adapters.zip
python -m zipfile -e adapters.zip results
```
(Or open the release page in a browser, download `adapters.zip` and extract it into `results/`.)

## 2. Start the application (no notebook / IDE needed)
```bash
python app/app.py
```
Open http://127.0.0.1:7860. Click **Run screening**; the models load on first use (~30 s). Tabs:
* **Screening** - paste a patient note, enter inclusion/exclusion criteria (one per line), choose *No cache / KV cache /
  Speculative* decoding, inspect every decision with its exact evidence sentences, download the HTML report.
* **Inference benchmark (live)** - runs the three decoders on *your* input and compares latency, tokens/s, peak memory and
  token equality.
* **Saved experiment results** - benchmark tables recorded during the project.

Invalid input (empty note, no criteria, too long, …) shows a message instead of crashing. Environment variables
`TARGET_ADAPTER`, `DRAFT_ADAPTER`, `PORT` override defaults.

## 3. Reproduce the experiments
```bash
python data/prepare_data.py                       # downloads dataset, writes 37/8/8 split + data/split_manifest.json
python src/token_stats.py                          # token lengths / truncation check
python src/eval_model.py --split val --tag base_zeroshot            # "before" fine-tuning (validation)
python src/train_lora.py --run lora_r16 --epochs 3                  # LoRA run 1
python src/train_lora.py --run lora_bal --epochs 2 --balance        # LoRA run 2 (class-balanced)
python src/train_lora.py --model models/Qwen2.5-0.5B-Instruct --run lora_draft --epochs 2 --balance   # draft model (saved to results/lora_draft/adapter)
python src/eval_model.py --split val --adapter results/<run>/adapter --tag <run>
# the final adapter is the run with the best VALIDATION macro-F1 (lora_bal: 0.534 vs 0.467); copy it into place:
python -c "import shutil; shutil.copytree('results/lora_bal/adapter', 'results/lora_final/adapter')"
# final, ONCE, after choosing the adapter on validation:
python src/eval_model.py --split test --allow-test --tag base_zeroshot
python src/eval_model.py --split test --allow-test --adapter results/lora_final/adapter --tag lora_final
python src/error_analysis.py --tag lora_final --split test
python src/test_attention.py                       # Task 4: correctness + benchmark
python src/bench_kv.py --adapter results/lora_final/adapter          # Task 3
python src/bench_spec.py --target-adapter results/lora_final/adapter --draft-adapter results/lora_draft/adapter --tag final   # Task 5
python src/make_figures.py                         # figures for the paper
```
`eval_model.py` refuses to touch the test split unless `--allow-test` is given.

## 4. Repository map
| Path | Content |
|---|---|
| `data/prepare_data.py`, `data/split_manifest.json` | patient-grouped split (seed 42), leakage assertions, GPT-4 columns dropped |
| `src/prompts.py` | prompt, JSON output schema, robust parser |
| `src/metrics.py` | accuracy / macro-F1 / per-class / confusion matrices / evidence P-R-F1 (empty-set rules documented) |
| `src/train_lora.py`, `src/eval_model.py` | fine-tuning and evaluation |
| `src/kv_decode.py`, `src/bench_kv.py` | Task 3: no-cache reference vs explicit KV-cache decoder, tests, benchmarks |
| `src/attention.py`, `src/test_attention.py` | Task 4: reference vs tiled online-softmax attention |
| `src/speculative.py`, `src/bench_spec.py` | Task 5: greedy speculative decoding |
| `app/app.py` | Task 6: browser application |
| `src/test_units.py`, `src/test_app.py` | fast CPU unit tests (metrics, parser, split integrity) / application test (needs GPU, 18 checks) |
| `src/make_figures.py`, `src/make_diagrams.py`, `src/make_screenshots.py` | paper figures, architecture diagrams, app screenshots |
| `src/check_tex.py`, `src/check_tex_chars.py` | LaTeX sanity checks (no TeX install needed) |
| `DEMO_SCRIPT.md`, `DEFENSE_NOTES.md`, `WORKLOG.md` | video plan, defense notes, step-by-step log of everything done |
| `results/` | logs, metrics JSON, per-example predictions, benchmark CSVs |
| `paper/` | IEEE LaTeX report (`main.tex`; compile with `pdflatex`/`bibtex` or on Overleaf) |

## 5. Notes
* Only the TrialGPT Criterion Annotations dataset is used; its GPT-4 predictions/explanations are never used as input.
* The trial-level verdict in the app is a rule-based aggregation of criterion-level predictions, not a labelled outcome.
* AI-assistant use is acknowledged in the report appendix.
