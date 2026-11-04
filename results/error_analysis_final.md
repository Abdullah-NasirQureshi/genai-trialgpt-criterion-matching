# Error analysis of the final model on the held-out test split

Model: Qwen2.5-1.5B-Instruct + class-balanced LoRA (`results/lora_final`), greedy decoding, 206 test rows.
Source data: `results/eval_lora_final_test.preds.jsonl`; machine-selected candidates: `results/error_analysis_lora_final_test.md`
(selection script `src/error_analysis.py`). Explanations below were written by hand after reading each note.

## Overall error profile (58 wrong labels out of 206)

| gold -> predicted | count |
|---|---|
| not enough information -> not excluded | 25 |
| not excluded -> not enough information | 9 |
| not applicable -> not excluded | 7 |
| not included -> not enough information | 6 |
| included -> not enough information | 5 |
| not applicable -> not enough information | 2 |
| 4 single cases (not excluded -> not applicable; NEI -> not included; NEI -> included; NEI -> not applicable) | 4 |

* 42 of the 58 wrong predictions cite **no** evidence sentence.
* 27 of the 66 gold "not enough information" rows received a decisive label (false certainty) - 25 of them "not excluded".
* The model is bimodal: for inclusion it is conservative (says NEI too often: 13 of the 16 inclusion errors go *to* NEI), for exclusion it
  defaults to "not excluded" (all 42 exclusion errors are into or out of that label).

## Twelve distinct cases

| # | Annotation / criterion | Gold -> Predicted | Failure type |
|---|---|---|---|
| 1 | 174, excl.: "previous axillary surgery" | NEI -> not excluded | absence treated as proof |
| 2 | 193, excl.: "patients were moribund" | NEI -> not excluded | ignored suggestive signs |
| 3 | 389, excl.: "severe renal impairment (GFR<50%)" | NEI -> not excluded | ignored renal cues; missing lab value |
| 4 | 325, incl.: "genetic patients" | NEI -> not included | over-interpretation of a vague criterion |
| 5 | 185, incl.: "diagnosed with shock or ARDS" | included -> NEI | missed implicit evidence |
| 6 | 188, incl.: "vasopressors to keep SBP >= 90" | not included -> NEI | debatable gold label / hedging |
| 7 | 162, incl.: "undergoing open AAA repair" | not included -> NEI | cannot assert a negative |
| 8 | 745, incl.: "female patients have adequate birth control" | N/A -> NEI | missed the sex cue ("boy") |
| 9 | 743, incl.: "daily access to e-mail" | NEI -> N/A | NEI vs N/A ambiguity |
| 10 | 422, incl.: "if female, using birth control" | N/A -> NEI | right evidence, wrong decision |
| 11 | 186, incl.: "heart rate of at least 90/min" | included -> included | wrong evidence sentence |
| 12 | 179 / 197, excl.: "central venous catheterization before" / "decreased consciousness" | not excluded -> not excluded | no evidence cited |

### 1. Absence of mention treated as proof of eligibility (annotation 174, patient sigir-201410)
Criterion: *previous axillary surgery* (exclusion). The note describes a 67-year-old woman with a groin mass after a femoral
catheterization; it says nothing about axillary or any other prior surgery. The experts label this **not enough information**; the model
says **not excluded** with no evidence. It read "nothing contradicts eligibility" as "eligible", which is exactly the reasoning the task forbids
(absent evidence is neither proof of eligibility nor of ineligibility). This is the most frequent error type (25 of 58 errors).

### 2. Suggestive but inconclusive signs ignored (annotation 193, patient sigir-201411)
Criterion: *patients were moribund* (exclusion). The patient is pale, tachypneic, tachycardic with BP 80/60 (sentences 2-3), which are
possible signs of severe illness, so experts cite sentences 0-4 and still conclude **NEI** (severity cannot be determined from vitals alone). The model
answered **not excluded** and cited nothing: it neither used the relevant sentences nor recognised the uncertainty.

### 3. Relevant cue present, decisive value missing (annotation 389, patient sigir-201421)
Criterion: *severe renal impairment (GFR < 50 %)*. The note reports proteinuria and RBC casts (sentence 3), i.e. kidney involvement in a likely lupus
patient, but no GFR. Gold: **NEI**; model: **not excluded** without evidence. A missing-information case where the model ignored positive hints of the
very organ in question.

### 4. Over-interpretation of a vague criterion (annotation 325, patient sigir-201417)
Criterion: *genetic patients* (inclusion). The model answered **not included** citing sentence 0 (history of common variable immunodeficiency). CVID can
be genetic but the note does not say so; gold is **NEI**. The model turned "not stated" into a negative and used a loosely related sentence as support - an
**unsupported explanation**.

### 5. Implicit evidence missed (annotation 185, patient sigir-201411)
Criterion: *diagnosed with shock, ARDS or both* (inclusion). The word "shock" never appears; the findings (pallor, tachypnea, tachycardia, BP 80/60 in
sentences 2-3) imply it and the experts mark **included**. The model predicted **NEI** with no evidence: it needs clinical inference, not keyword matching, which a
1.5B model fine-tuned on 603 examples does only inconsistently.

### 6. Hedging on a debatable gold label (annotation 188, patient sigir-201411)
Criterion: *vasopressors needed to keep systolic BP >= 90 mmHg*. Experts say **not included** citing BP 80/60 (sentence 3), although no vasopressor
is mentioned either way; one could argue NEI. The model said **NEI**. This shows that some gold labels depend on a judgement call, so part of the
error is annotation ambiguity rather than model failure.

### 7. Asserting a negative is hard (annotation 162, patient sigir-201410)
Criterion: *patients undergoing open abdominal aortic aneurysm repair*. The patient has a pulsatile groin mass and bruit after femoral arterial access - a
different problem. Experts: **not included**. Model: **NEI**, citing sentence 0 (the catheterization). The 6 "not included -> NEI" errors show the model rarely commits to a
negative inclusion verdict (recall 1/7).

### 8. Missed demographic cue (annotation 745, patient sigir-201513)
Criterion: *female patients have an adequate birth-control method*. The patient is a 5-year-old **boy** (sentence 0). Gold: **not applicable**. Model: **NEI**.
Applicability depends on a single word; the two N/A -> NEI errors (cases 8 and 10) reveal weak handling of "if female/male" conditions.

### 9. NEI versus not-applicable ambiguity (annotation 743, patient sigir-201513)
Criterion: *patient has daily access to his/her e-mail* for the same 5-year-old. Experts: **NEI** (citing the compliance sentence 4); model: **not applicable**. The boundary between
"cannot be determined" and "does not apply" is subjective, and the two labels were confused in both directions (2 N/A -> NEI and 1 NEI -> N/A).

### 10. Right evidence, wrong decision (annotation 422, patient sigir-201422)
Criterion: *if female, using birth control*. Both experts and model cite sentence 3 ("no sexual partners"), but the experts conclude **not applicable** (birth control is
irrelevant) while the model concludes **NEI**. Evidence retrieval succeeded; the reasoning step from evidence to category failed.

### 11. Correct label, wrong evidence (annotation 186, patient sigir-201411)
Criterion: *heart rate of at least 90/min*. The model correctly says **included** but cites sentence 3 (blood pressure / temperature) instead of sentence 2
("tachycardic"). The explanation shown to a user would therefore not support the decision - a **traceability failure** even though the label is right (7 such cases).

### 12. Correct label, no evidence cited (annotations 179 and 197)
For *central venous puncture/catheterization before* the experts cite sentence 0 (arterial cardiac catheterization - related but not venous) and for *decreased level of
consciousness* sentences 0 and 5; the model predicts **not excluded** correctly but cites nothing. Across the test set the model cites no evidence for 143 of 206 rows, while experts leave 145 empty,
so this under-citation costs recall on the 61 rows with gold evidence (recall 0.60).

## Take-aways
1. The main weakness is **calibration on missing information in exclusion criteria**: the model almost never says NEI for them (3 of 28 correct).
2. Fine-tuning fixed format and eight-way label vocabulary (zero-shot accuracy 0.058 -> 0.718) but not clinical inference (cases 5, 6, 7).
3. Applicability ("if female") and NEI/N/A boundaries are fragile with 603 training examples and 12-23 examples in some classes.
4. Some gold labels are themselves debatable (case 6, 9), which bounds the achievable score.
