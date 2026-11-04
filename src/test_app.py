"""Functional test of the application logic (no browser needed).  python src/test_app.py
Writes results/app_test.log-style output to stdout. Checks: input validation, all 3 decoding modes
give identical decisions, evidence text is exact note text, report export contains provenance + disclaimer."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import app  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""), flush=True)


# ---- 1. invalid inputs are rejected with a message, not a crash (no model needed)
for label, note, inc, exc in [("empty note", "", "criterion A", ""),
                              ("tiny note", "hi", "criterion A", ""),
                              ("no criteria", "A 40-year-old woman presents with chest pain and dyspnea.", "", "  \n "),
                              ("31 criteria", "A 40-year-old woman presents with chest pain and dyspnea.", "\n".join(f"c{i}" for i in range(31)), ""),
                              ("overlong criterion", "A 40-year-old woman presents with chest pain and dyspnea.", "x" * 900, "")]:
    df, msg, status, rep = app.screen(note, "", inc, exc, "KV cache", 4)
    check(f"invalid input rejected: {label}", df is None and "Input problem" in msg and rep is None, msg.splitlines()[-1][:70])

# ---- 2. valid run in all three modes
res = {}
for mode in ("KV cache", "No cache (reference)", "Speculative decoding"):
    df, msg, status, rep = app.screen(app.EXAMPLE_NOTE, "Example trial", app.EXAMPLE_INC, app.EXAMPLE_EXC, mode, 4)
    check(f"screening runs in mode '{mode}'", df is not None and len(df) == 5, status[:90])
    res[mode] = df
    if mode == "KV cache":
        print(df[["Type", "Criterion", "Decision", "Evidence IDs"]].to_string(), "\n", msg[:420], flush=True)
        html = Path(rep).read_text(encoding="utf-8")
        (ROOT / "results" / "sample_screening_report.html").write_text(html, encoding="utf-8")
        check("report does not leak absolute local paths", "Uni Stuff" not in html and "Users" not in html and ":" + chr(92) not in html)
        check("report contains disclaimer", "not a clinical enrollment decision" in html)
        check("report contains provenance", "Provenance" in html and "LoRA adapter" in html and "Decoding mode" in html)
        sents = app.split_sentences(app.EXAMPLE_NOTE)
        good = True
        for _, r in df.iterrows():
            for i in [int(x) for x in str(r["Evidence IDs"]).replace("-", "").split(",") if x.strip()]:
                good &= sents[i] in r["Evidence text"]
        check("evidence text is the exact note sentence for every cited ID", good)
        check("overall outcome present", "Overall preliminary outcome" in msg)
a, b, c = (res[m]["Decision"].tolist() for m in res)
check("all three decoding modes give the same decisions", a == b == c, f"{a}")

# ---- 3. free-text (unnumbered) note gets sentence IDs automatically
df, msg, status, rep = app.screen("The patient is a 70 year old man with diabetes. He takes metformin. He has no known allergies.",
                                  "", "Adults with type 2 diabetes", "Known allergy to metformin", "KV cache", 4)
check("unnumbered free-text note accepted", df is not None and len(df) == 2)

# ---- 4. live benchmark function
bdf, bmsg = app.live_benchmark(app.EXAMPLE_NOTE, app.EXAMPLE_INC, app.EXAMPLE_EXC, 32, 4, 1)
check("live benchmark returns 3 modes", bdf is not None and len(bdf) == 3)
if bdf is not None:
    print(bdf.to_string(), flush=True)
    check("benchmark modes produce identical tokens", bool(bdf["Same tokens as KV-cache"].all()))
bdf, bmsg = app.live_benchmark("", "", "", 32, 4, 1)
check("benchmark with invalid input is rejected", bdf is None and "Input problem" in bmsg)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
