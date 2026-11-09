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
29. **Speculative decoding tests and benchmark done** (`results/spec_*_final.csv`). Output is **identical to normal greedy
    decoding in every test** (12/12 task prompts for k=2,4,6,8; 4/4 long generations; and identical to Hugging Face's own
    assisted generation on 6/6). The fine-tuned draft is accepted 90-95% of the time at small k (62-84% on long free
    text), giving up to 7 tokens per big-model pass. But overall it is **slower** (0.71-0.82x of normal cached decoding):
    on this laptop GPU the 0.5B draft is almost as slow per token as the 1.5B model, so drafting costs too much.
30. **FINAL TEST RESULTS (run once, adapter chosen on validation only)**, 206 test rows, 0 invalid outputs:
    * Zero-shot base model: accuracy 0.058, macro-F1 0.084.
    * Fine-tuned (class-balanced LoRA): accuracy **0.718**, macro-F1 **0.604** (inclusion 0.787 / 0.579; exclusion 0.679 / 0.500).
    * Evidence sentences: micro-F1 0.531 on all rows, 0.636 on rows that have gold evidence (zero-shot: 0.122 / 0.398).
    * Honest caveats: exclusion accuracy (0.679) is slightly BELOW "always say not excluded" (0.695); the model finds only 3 of 28
      exclusion "not enough information" cases and 1 of 7 "not included"; the test set has no "excluded" example.
31. **Error analysis written** (`results/error_analysis_final.md`): 58 of 206 test predictions are wrong. 25 are "not enough
    information" wrongly called "not excluded" (the model treats 'not mentioned' as 'not a problem'); 42 of the 58 wrong
    answers cite no evidence; 12 distinct hand-explained cases (false certainty, missed implicit evidence, wrong
    evidence sentence, 'not applicable' vs 'not enough information' confusion, debatable gold labels). I double-checked every
    aggregate number against the prediction file.
32. **Browser app tested** (`src/test_app.py`, log `results/app_test.log`): 17 checks pass -- bad input is rejected with a
    message (empty note, no criteria, 31 criteria, overlong criterion); the three decoding modes give identical decisions;
    cited evidence is the exact note sentence; the HTML report has the disclaimer and provenance; the live benchmark runs
    (KV cache 12.5 tok/s vs 5.1 without cache vs 8.8 speculative on the demo input).
    **Bug found and fixed:** the web server would not start ("localhost is not accessible") because newer `pydantic` breaks
    Gradio 5.12; pinned `pydantic==2.10.6` (also in `requirements.txt`). After the fix the server returns HTTP 200 and a
    request through the real web API (the path the browser uses) returned correct results.
33. **App polish + screenshots.** Fixed three small things found while taking screenshots with headless Chrome: empty results
    table had no column names; the report printed my local file path (now repo-relative); and for "not excluded" with no cited
    sentence the explanation now says it is NOT proof the note rules the criterion out. Re-ran the 18 app checks (all pass)
    and saved screenshots `paper/figs/app_screening.png` and `paper/figs/app_report.png`.
34. **Wrote the rest of the paper:** `paper/sec_results.tex` (every number taken from the result files; 12 tables/figures, each
    explained in the text), `paper/sec_limits_conclusion.tex`, plus a patient-level bootstrap (`src/bootstrap_ci.py`:
    fine-tuned test accuracy 0.721, 95% interval [0.602, 0.803]; zero-shot 0.060 [0.031, 0.099]) and a trivial majority-label
    baseline (test acc 0.626, macro-F1 0.272) so the model is compared with something sensible. While re-reading I
    corrected several mistakes of my own (one wrong ratio, GB vs MiB, an unclear memory statement) before finalising.
35. **LaTeX checks without a compiler:** `src/check_tex.py` (citations, labels, figures, braces) and
    `src/check_tex_chars.py` (stray %, _, &, non-ASCII) both pass. Still needs a real compile on Overleaf.
36. **Release + docs.** Created GitHub release `v1.0` with `adapters.zip` (final 1.5B adapter + 0.5B draft adapter, 101 MB) and
    tested downloading it back (same size, zip OK). Re-ran the split script: the manifest is identical (reproducible). Added a
    design-decisions table to the paper (alternatives, evidence, difficulties) and wrote `DEMO_SCRIPT.md` (minute-by-minute
    plan for the 5-7 min video, with an unseen test note) and `DEFENSE_NOTES.md` (short answers + where each thing is in code).
    I cannot record the video or fill in your name/university -- see the TODO list at the end of this file.
37. **Untuned-draft comparison finished** (`results/spec_*_basedraft.csv`): output identical to greedy again; on the TASK the
    fine-tuned draft is clearly better (acceptance 0.90 vs 0.62 at k=4), but on free text (not the fine-tuning task) the
    untuned draft is as good or slightly better. The same plain-greedy baseline measured 13.8 vs 12.6 tokens/s in the two
    runs (~9% noise). I wrote this into the paper instead of claiming "fine-tuned draft is always better".
38. **Checked the demo example with the real model** (`results/demo_probe.py`): both inclusion criteria correct with right
    evidence; pregnancy "not applicable" (correct); GFR "not enough information" (correct); "active malignancy" (not in
    note) -> "not excluded" = the known weakness. `DEMO_SCRIPT.md` now shows the real outputs.
39. **More app screenshots** with Playwright driving the installed Chrome (`src/make_screenshots.py`): after a run, live
    benchmark tab, invalid input + recovery. Also removed my local file path from the app's status line. Added to the
    paper as three more figures. App test (18 checks) and both LaTeX checkers still pass.

---------------------------------------------------------------------------------------------------
# WHAT IS LEFT FOR YOU (I cannot do these) -- deadline Nov 10, 2026 (+2 hour grace)
1. **Make the GitHub repo public** (it is private now): GitHub -> repo -> Settings -> Danger zone -> Change visibility,
   or: `gh repo edit Abdullah-NasirQureshi/genai-trialgpt-criterion-matching --visibility public --accept-visibility-change-consequences`.
   Graders must be able to open it. (After that the release download works without login.)
2. **Compile the paper:** upload the whole `paper/` folder to Overleaf (main file `main.tex`, IEEEtran is built in) and
   download the PDF. I could not compile locally (no LaTeX; downloading one was blocked). If Overleaf reports an error,
   it is most likely a small typo -- the two checker scripts passed, but they are not a real compiler.
3. **Fill in your details** in `paper/main.tex`: search `TODO(author)` -> your name, university, student ID, email; the
   demo-video link (section "Reproducibility and Availability"); review the AI-use appendix (`paper/sec_appendix_ai.tex`)
   so it describes your own use of AI honestly.
4. **Record the 5-7 minute demo video** following `DEMO_SCRIPT.md` (it has the exact inputs and expected outputs) and upload it.
5. **Practise the defense** with `DEFENSE_NOTES.md`; run the app once yourself (`python app/app.py`), and re-run
   `python src/test_app.py` if you change anything. Be ready to change a component live (table in the notes).
6. **Submit on Google Classroom:** report PDF, GitHub URL, adapter link
   (https://github.com/Abdullah-NasirQureshi/genai-trialgpt-criterion-matching/releases/tag/v1.0), the video link, and mention the
   split manifest (`data/split_manifest.json`) and benchmark logs (`results/`).
7. Optional: `git pull`/`git push` -- the local repo and GitHub are in sync as of the last commit.

## Nov 10 -- full audit against the assignment text (found and fixed)
40. **Missing in the paper: architecture diagrams.** The assignment lists "architecture diagrams" under Materials and Methods and I
    had none. Added 4 (`src/make_diagrams.py`): system/application workflow, decoder + KV cache (prefill vs decode),
    tiled online-softmax attention, one round of speculative decoding with cache sync. Each is referenced in the text.
41. **Reproducibility bugs fixed:** `results/dl_curl.sh` failed on a fresh clone (folders did not exist); README did not say how
    `results/lora_final/adapter` is made (copy of `lora_bal`); README draft command said 3 epochs but I trained 2 epochs with `--balance`;
    `unzip` replaced by a Python command that works on Windows.
42. **Small factual fix:** cache for a 555-token prompt is 15.2 MiB (paper said 15.5).
43. **Claim made true:** the paper's appendix says metric/parse/split checks are automated in the repo -> added `src/test_units.py`
    (23 checks, all pass).
44. **References verified online:** TrialGPT = Nature Communications 15, 9074 (2024); dataset on Hugging Face has the 1015 expert pairs.
45. **Author details filled in** (paper/main.tex): Abdullah Nasir, Department of Computer Science, FAST-NUCES Islamabad, 23I-0726,
    i230726@isb.nu.edu.pk. AI-use appendix approved by the author (TODO comment removed). Fixed the author-block line breaks.
    ONLY placeholder left: the demo-video link (`[TODO(author): insert video link]` in paper/main.tex).
