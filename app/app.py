"""Browser application: clinical-trial criterion screening with live decoding comparison.

Start:   python app/app.py            -> open http://127.0.0.1:7860
Needs:   models/Qwen2.5-1.5B-Instruct, models/Qwen2.5-0.5B-Instruct and the LoRA adapter
         (see README for download). Everything runs locally; no internet or paid API.

EDUCATIONAL SCREENING PROTOTYPE - NOT A CLINICAL ENROLLMENT DECISION.
"""
import datetime as dt
import html
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Never route local traffic through a (Windows/system) proxy: Gradio checks http://127.0.0.1 on start-up
# and fails with "localhost is not accessible" when a proxy intercepts it.
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "127.0.0.1,localhost,0.0.0.0," + os.environ.get(_k, "")

import gradio as gr
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kv_decode import greedy_kv, greedy_nocache  # noqa: E402
from prompts import encode_prompt, parse_output, split_sentences, labels_for  # noqa: E402
from speculative import speculative_greedy  # noqa: E402

BIG = os.environ.get("TARGET_MODEL", str(ROOT / "models" / "Qwen2.5-1.5B-Instruct"))
SMALL = os.environ.get("DRAFT_MODEL", str(ROOT / "models" / "Qwen2.5-0.5B-Instruct"))
ADAPTER = os.environ.get("TARGET_ADAPTER", str(ROOT / "results" / "lora_final" / "adapter"))
DRAFT_ADAPTER = os.environ.get("DRAFT_ADAPTER", str(ROOT / "results" / "lora_draft" / "adapter"))
MAX_NOTE_TOKENS = 1500
MAX_NEW = 64
DISCLAIMER = ("**Educational screening prototype - NOT a clinical enrollment decision.** Outputs come from a small "
              "language model fine-tuned on 37 patients and can be wrong. A qualified clinician must review every result.")

_STATE = {}


# ----------------------------------------------------------------------------- model loading
def load_models():
    if "target" in _STATE:
        return "Models already loaded."
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BIG)
    def mk(path, adapter):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
        used = "none (base weights)"
        if adapter and Path(adapter).exists():
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval(); used = adapter
        return m, used
    target, ta = mk(BIG, ADAPTER)
    draft, da = mk(SMALL, DRAFT_ADAPTER)
    _STATE.update(tok=tok, target=target, draft=draft, target_adapter=ta, draft_adapter=da,
                  eos=tok.convert_tokens_to_ids("<|im_end|>"))
    return (f"Loaded in {time.time() - t0:.0f}s. Target: {Path(BIG).name} + adapter: {short(ta)}. "
            f"Draft: {Path(SMALL).name} + adapter: {short(da)}.")


# ----------------------------------------------------------------------------- inference
def decode(prompt_ids, mode, k):
    tg, dr, eos = _STATE["target"], _STATE["draft"], _STATE["eos"]
    if mode == "No cache (reference)":
        r = greedy_nocache(tg, prompt_ids, MAX_NEW, eos); return r.tokens, r.prefill_s + r.decode_s, {}
    if mode == "KV cache":
        r = greedy_kv(tg, prompt_ids, MAX_NEW, eos); return r.tokens, r.prefill_s + r.decode_s, {}
    r = speculative_greedy(tg, dr, prompt_ids, MAX_NEW, k=k, eos_id=eos)
    return r.tokens, r.seconds, {"acceptance": r.acceptance_rate, "tokens_per_target_forward": r.tokens_per_round}


def explain(label, ev, sents):
    if label is None:
        return "The model did not return a valid answer; treat as unknown."
    if label == "not applicable":
        return "The model judged this criterion irrelevant to this patient."
    if not ev:
        if label == "not enough information":
            return ("No sentence in the note was cited. Absence of evidence is NOT proof either way; "
                    "this means the note itself does not settle the question.")
        if label in ("not excluded", "not included"):
            return (f"No sentence was cited. '{label}' here only means the model found nothing in the note that decides the "
                    "criterion; this is NOT confirmation that the note rules it out - check whether the note is complete.")
        return "The model gave this label without citing a sentence (unsupported); verify manually."
    quotes = " ".join(f'[{i}] "{sents[i]}"' for i in ev)
    return f"Label '{label}' is based on note sentence(s): {quotes}"


def aggregate(results):
    """Explicit rule-based roll-up (NOT a ground-truth label)."""
    bad = [r for r in results if r["label"] in ("not included", "excluded")]
    unknown = [r for r in results if r["label"] in (None, "not enough information")]
    if bad:
        return ("LIKELY NOT ELIGIBLE (preliminary)",
                "Rule: at least one inclusion criterion is 'not included' or one exclusion criterion is 'excluded'.")
    if unknown:
        return ("UNDETERMINED - more information needed (preliminary)",
                "Rule: no criterion was violated but some criteria lack information in the note.")
    return ("LIKELY ELIGIBLE (preliminary)",
            "Rule: every applicable inclusion criterion is 'included' and no exclusion criterion applies.")


def parse_criteria(text, ctype):
    return [(ctype, ln.strip()) for ln in (text or "").splitlines() if ln.strip()]


def validate(note, inc, exc):
    errs = []
    if not note or len(note.strip()) < 20:
        errs.append("The patient note is empty or too short (need at least ~20 characters).")
    crit = parse_criteria(inc, "inclusion") + parse_criteria(exc, "exclusion")
    if not crit:
        errs.append("Enter at least one inclusion or exclusion criterion (one per line).")
    if len(crit) > 30:
        errs.append("Too many criteria (max 30 per run).")
    if any(len(c[1]) > 800 for c in crit):
        errs.append("A criterion is longer than 800 characters.")
    if not errs and "tok" in _STATE:
        n = len(_STATE["tok"](note).input_ids)
        if n > MAX_NOTE_TOKENS:
            errs.append(f"The patient note has {n} tokens (limit {MAX_NOTE_TOKENS}); please shorten it.")
    return errs, crit


def screen(note, trial_title, inc, exc, mode, k, progress=gr.Progress()):
    errs, crit = validate(note, inc, exc)
    if errs:
        return None, "### Input problem\n" + "\n".join(f"- {e}" for e in errs), "", None
    try:
        status = load_models()
    except Exception as e:  # missing files / no GPU etc.
        return None, f"### Could not load the models\n`{type(e).__name__}: {e}`\nSee README for setup.", "", None
    tok = _STATE["tok"]
    sents = split_sentences(note.strip())
    rows, results = [], []
    t_all = time.time()
    for n, (ctype, ctext) in enumerate(crit):
        progress(n / len(crit), desc=f"criterion {n + 1}/{len(crit)}")
        prompt = encode_prompt(tok, sents, ctype, ctext)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        try:
            toks, secs, extra = decode(ids, mode, int(k))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, "### GPU out of memory\nShorten the note or choose a lighter decoding mode.", "", None
        text = tok.decode(toks, skip_special_tokens=True)
        label, ev, ok = parse_output(text, ctype, len(sents))
        results.append(dict(label=label))
        rows.append({"#": n + 1, "Type": ctype, "Criterion": ctext, "Decision": label if ok else "INVALID OUTPUT",
                     "Evidence IDs": ", ".join(map(str, ev)) or "-",
                     "Evidence text": " | ".join(f"[{i}] {sents[i]}" for i in ev) or "-",
                     "Explanation": explain(label, ev, sents), "Seconds": round(secs, 2),
                     "Raw output": text.strip()})
    df = pd.DataFrame(rows)
    verdict, rule = aggregate(results)
    missing = [f"- ({r['Type']}) {r['Criterion']}" for r in rows if r["Decision"] in ("not enough information", "INVALID OUTPUT")]
    summary = (f"## Overall preliminary outcome: {verdict}\n\n{rule}\n\n"
               f"**Criteria with missing/unclear information ({len(missing)}):**\n" + ("\n".join(missing) or "- none") +
               f"\n\n_Mode: {mode}; total time {time.time() - t_all:.1f}s for {len(rows)} criteria._\n\n{DISCLAIMER}")
    report = write_report(note, trial_title, df, verdict, rule, mode, k)
    return df, summary, status, report


def short(path):
    """repo-relative path for provenance (never print the user's absolute path)"""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return Path(str(path)).name


def write_report(note, title, df, verdict, rule, mode, k):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    esc = html.escape
    rows = "".join(
        f"<tr><td>{r['#']}</td><td>{r['Type']}</td><td>{esc(r['Criterion'])}</td><td><b>{esc(str(r['Decision']))}</b></td>"
        f"<td>{esc(r['Evidence IDs'])}</td><td>{esc(r['Evidence text'])}</td><td>{esc(r['Explanation'])}</td></tr>"
        for _, r in df.iterrows())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:Arial;margin:2em;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #999;padding:6px;vertical-align:top;font-size:13px}}th{{background:#eee}}
.warn{{background:#fff3cd;border:1px solid #e0b000;padding:10px}}</style></head><body>
<h1>Eligibility screening report</h1>
<div class="warn"><b>Educational screening prototype - not a clinical enrollment decision.</b> Review by a qualified clinician is required.</div>
<p><b>Trial:</b> {esc(title or '(not given)')}<br><b>Generated:</b> {now}</p>
<h2>Overall preliminary outcome: {esc(verdict)}</h2><p>{esc(rule)}</p>
<h2>Criterion-level results</h2>
<table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Decision</th><th>Evidence IDs</th><th>Evidence text</th><th>Explanation</th></tr>{rows}</table>
<h2>Patient note (as screened)</h2><pre style="white-space:pre-wrap">{esc(note)}</pre>
<h2>Provenance</h2><ul><li>Target model: {esc(Path(BIG).name)} + LoRA adapter {esc(short(_STATE.get('target_adapter')))}</li>
<li>Decoding mode: {esc(mode)}{' (k=%d, draft %s + %s)' % (k, Path(SMALL).name, short(_STATE.get('draft_adapter'))) if mode.startswith('Spec') else ''}</li>
<li>Greedy decoding, max {MAX_NEW} new tokens, fp16, local execution</li>
<li>Overall outcome is a rule-based aggregation of criterion-level predictions, not a labelled ground truth.</li></ul>
</body></html>"""
    f = tempfile.NamedTemporaryFile("w", suffix=".html", prefix="screening_report_", delete=False, encoding="utf-8")
    f.write(doc); f.close()
    return f.name


# ----------------------------------------------------------------------------- benchmark tab
def live_benchmark(note, inc, exc, new_tokens, k, reps):
    errs, crit = validate(note, inc, exc)
    if errs:
        return None, "### Input problem\n" + "\n".join(f"- {e}" for e in errs)
    try:
        load_models()
    except Exception as e:
        return None, f"### Could not load the models\n`{type(e).__name__}: {e}`"
    tok, tg, dr, eos = _STATE["tok"], _STATE["target"], _STATE["draft"], _STATE["eos"]
    ctype, ctext = crit[0]
    ids = tok(encode_prompt(tok, split_sentences(note.strip()), ctype, ctext), return_tensors="pt",
              add_special_tokens=False).input_ids.cuda()
    N = int(new_tokens)
    ref = greedy_kv(tg, ids, N, None).tokens          # reference tokens (also warm-up)
    out = []
    def run(name, fn):
        rs = [fn() for _ in range(int(reps))]
        t = sorted(r[0] for r in rs)[len(rs) // 2]; toks = rs[0][1]
        out.append({"Mode": name, "Seconds (median)": round(t, 3), "Tokens/s": round(len(toks) / t, 2),
                    "Peak GPU MiB": round(torch.cuda.max_memory_allocated() / 2**20),
                    "Same tokens as KV-cache": toks == ref, "Extra": rs[0][2]})
    def f_nc():
        r = greedy_nocache(tg, ids, N, None); return r.prefill_s + r.decode_s, r.tokens, ""
    def f_kv():
        r = greedy_kv(tg, ids, N, None); return r.prefill_s + r.decode_s, r.tokens, ""
    def f_sp():
        r = speculative_greedy(tg, dr, ids, N, k=int(k), eos_id=None)
        return r.seconds, r.tokens, f"accept {r.acceptance_rate:.2f}, {r.tokens_per_round:.2f} tok/target-fwd"
    for name, fn in (("No cache", f_nc), ("KV cache", f_kv), (f"Speculative (k={int(k)})", f_sp)):
        torch.cuda.reset_peak_memory_stats(); run(name, fn)
    df = pd.DataFrame(out)
    return df, (f"Prompt = {ids.shape[1]} tokens (first criterion of your input), generating {N} tokens, "
                f"{int(reps)} timed repeats after warm-up. Speculative decoding may be slower than plain KV decoding - "
                f"see the report for why.")


def saved_results():
    out = []
    for f, title in (("kv_benchmark.csv", "KV cache vs no cache (saved benchmark)"),
                     ("spec_benchmark_final.csv", "Speculative decoding (saved benchmark)"),
                     ("attention_benchmark.csv", "Attention implementations (saved benchmark)")):
        p = ROOT / "results" / f
        out.append((title, pd.read_csv(p) if p.exists() else pd.DataFrame({"note": [f"{f} not found - run the benchmarks"]})))
    return out


EXAMPLE_NOTE = ("0. A 58-year-old man presents with two days of crushing chest pain radiating to the left arm.\n"
                "1. He has a history of hypertension and type 2 diabetes treated with metformin.\n"
                "2. He smokes one pack of cigarettes per day.\n"
                "3. ECG shows ST elevation in leads II, III and aVF.\n"
                "4. He denies any history of bleeding disorders.")
EXAMPLE_INC = "Adults aged 18 years or older with acute myocardial infarction\nPatients with ST-segment elevation on ECG"
EXAMPLE_EXC = "History of bleeding disorder or active bleeding\nPregnancy or lactation\nKnown allergy to aspirin"


def build_ui():
    with gr.Blocks(title="Trial criterion screening (educational prototype)") as demo:
        gr.Markdown("# Clinical-trial criterion screening\n" + DISCLAIMER)
        with gr.Tab("Screening"):
            with gr.Row():
                with gr.Column():
                    note = gr.Textbox(label="Patient note (free text; numbered lines '0. ...' are used as sentence IDs, "
                                            "otherwise sentences are numbered automatically)", lines=10, value=EXAMPLE_NOTE)
                    title = gr.Textbox(label="Trial title (optional)", value="Example trial")
                    inc = gr.Textbox(label="Inclusion criteria (one per line)", lines=4, value=EXAMPLE_INC)
                    exc = gr.Textbox(label="Exclusion criteria (one per line)", lines=4, value=EXAMPLE_EXC)
                    mode = gr.Radio(["No cache (reference)", "KV cache", "Speculative decoding"], value="KV cache",
                                    label="Inference mode (all produce identical greedy output)")
                    k = gr.Slider(1, 8, value=4, step=1, label="Speculative draft length k")
                    run = gr.Button("Run screening", variant="primary")
                with gr.Column():
                    summary = gr.Markdown()
                    status = gr.Textbox(label="Model status", interactive=False)
                    report = gr.File(label="Download screening report (HTML)")
            table = gr.Dataframe(label="Criterion-level results (decision, exact evidence, explanation)", wrap=True,
                                 headers=["#", "Type", "Criterion", "Decision", "Evidence IDs", "Evidence text", "Explanation", "Seconds", "Raw output"])
            run.click(screen, [note, title, inc, exc, mode, k], [table, summary, status, report])
        with gr.Tab("Inference benchmark (live)"):
            gr.Markdown("Runs the three decoders on **your** input right now and compares speed, memory and token equality.")
            nt = gr.Slider(16, 128, value=64, step=16, label="New tokens to generate")
            kk = gr.Slider(1, 8, value=4, step=1, label="Speculative k")
            rp = gr.Slider(1, 5, value=2, step=1, label="Timed repeats")
            b = gr.Button("Run benchmark")
            bt = gr.Dataframe(); bm = gr.Markdown()
            b.click(live_benchmark, [note, inc, exc, nt, kk, rp], [bt, bm])
        with gr.Tab("Saved experiment results"):
            gr.Markdown("Benchmarks recorded during the project (CSV files in `results/`).")
            for title_, df in saved_results():
                gr.Markdown(f"### {title_}"); gr.Dataframe(df, interactive=False)
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)))
