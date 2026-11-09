# Demo video script (target 6 minutes, limit 5-7)

Record your screen with voice (OBS, Xbox Game Bar `Win+G`, or Zoom). Upload to YouTube (unlisted) or Drive and paste the
link into `paper/main.tex` (search `TODO(author): paste YouTube link here` (author block, top of the first page)) and into the Classroom submission.

**Before recording:** `cd` to the project, activate `.venv`, run `python app/app.py`, open http://127.0.0.1:7860.
Close other GPU programs. Do one dry run first so the models are already loaded (first click takes ~15 s).

| Time | What you show | What you say (simple words) |
|---|---|---|
| 0:00-0:30 | Title slide or the README top | "This is my Assignment 2: matching a patient note to each inclusion/exclusion criterion of a clinical trial with a fine-tuned Qwen2.5-1.5B. It is an educational prototype, not a clinical tool." |
| 0:30-1:15 | Paper figure 1/Table 1 or `data/split_manifest.json` | "I used only the TrialGPT annotations. Patient-grouped split with seed 42: 37 train, 8 validation, 8 test patients, no patient in two splits. GPT-4 columns are dropped. Test was evaluated once at the end." |
| 1:15-2:00 | Results table (test): accuracy 0.058 -> 0.718, macro-F1 0.084 -> 0.604 | "LoRA fine-tuning took accuracy from 5.8% to 71.8%. Honest weak point: for exclusion criteria the model often says 'not excluded' when the note simply doesn't say - 25 of 28 such cases." |
| 2:00-3:30 | **App, Screening tab.** Click *Run screening* with the built-in example. Point at: decision, evidence IDs, exact evidence text, explanation, overall outcome, disclaimer. Click the report download and open the HTML. | "Each criterion gets a label, the cited sentence IDs and the exact sentences. The overall outcome is a simple rule on the criterion results, not a ground truth." |
| 3:30-4:15 | **Unseen case.** Paste a *new* note + criteria (see below). Run it. Show a "not enough information" result. | "This note was never seen in training. Where the note is silent the model should say not enough information." |
| 4:15-4:45 | **Invalid input recovery.** Clear the note, click Run -> message. Then enter 40 criteria / empty criteria -> message. Then fix and run again. | "Bad input never crashes the app; it tells me what to fix." |
| 4:45-5:45 | **Inference benchmark tab.** Click *Run benchmark*. Show table: No cache vs KV cache vs Speculative, same tokens = True. Then show Saved results tab (attention benchmark). | "KV cache is 2-7x faster than no cache with the same output. Speculative decoding gives identical output too but is *slower* here (0.7-0.8x) because the draft model is almost as expensive per step as the big one on this laptop GPU." |
| 5:45-6:15 | Repo/README + release page | "Code, split manifest, logs, report and the adapter download are in the GitHub repo." |

## Unseen test input to paste (not from the dataset)
**Note** (numbered lines):
```
0. A 67-year-old woman presents with progressive shortness of breath over two weeks.
1. She has a 15-year history of type 2 diabetes treated with insulin.
2. Echocardiography shows an ejection fraction of 30%.
3. She takes lisinopril and furosemide.
4. She has never smoked.
```
**Inclusion criteria**
```
Adults with heart failure and reduced ejection fraction
Treated with an ACE inhibitor or ARB
```
**Exclusion criteria**
```
Pregnancy
Active malignancy
Estimated GFR below 30 mL/min
```
**What the final model actually returned (tested, KV-cache mode, greedy - the output is deterministic):**

| Criterion | Decision | Evidence | Comment |
|---|---|---|---|
| Heart failure with reduced EF (incl.) | included | 0, 2 | correct |
| ACE inhibitor or ARB (incl.) | included | 3 | correct |
| Pregnancy (excl.) | not applicable | - | correct for a 67-year-old |
| Active malignancy (excl.) | **not excluded** | - | note never mentions cancer, so "not enough information" would be the right answer: this is the known weakness |
| Estimated GFR < 30 (excl.) | not enough information | - | correct |

Overall outcome shown: **UNDETERMINED - more information needed (preliminary)** (because of the GFR row). In the video, point at the
"Active malignancy" row and say openly that this is exactly the failure described in the paper's error analysis (cases 1-3): the model
treats silence as "not excluded", which is why the app says "check whether the note is complete". Being honest here shows you understand the limitation.

## Checklist
- [ ] Disclaimer visible at the top of the app at least once
- [ ] Show all three inference modes in the app
- [ ] Show an invalid-input message and recovery
- [ ] Show the downloaded report
- [ ] Video length between 5 and 7 minutes
