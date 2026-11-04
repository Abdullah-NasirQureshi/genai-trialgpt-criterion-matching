"""Poor-man's LaTeX checker (no TeX install needed):
braces, begin/end pairs, \\cite keys vs refs.bib, \\ref labels, \\input files, figure files,
and labels that are never referenced in the text (every figure/table must be discussed)."""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "paper"
BS = "\\"
main = (P / "main.tex").read_text(encoding="utf-8")
files = [P / "main.tex"] + [P / (m + ".tex") for m in re.findall(re.escape(BS) + r"input\{([^}]+)\}", main)]
bib = (P / "refs.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


txt, ok = "", True
for f in files:
    if not f.exists():
        print("MISSING input file:", f.name); ok = False; continue
    t = f.read_text(encoding="utf-8"); txt += "\n" + t
    body = strip_comments(t)
    opens = body.count("{") - body.count(BS + "{")
    closes = body.count("}") - body.count(BS + "}")
    if opens != closes:
        print(f"UNBALANCED braces in {f.name}: {opens} open vs {closes} close"); ok = False
    for env in set(re.findall(re.escape(BS) + r"begin\{([^}]+)\}", body)):
        nb = len(re.findall(re.escape(BS) + r"begin\{%s\}" % re.escape(env), body))
        ne = len(re.findall(re.escape(BS) + r"end\{%s\}" % re.escape(env), body))
        if nb != ne:
            print(f"begin/end mismatch for {env} in {f.name}: {nb} vs {ne}"); ok = False

body = strip_comments(txt)
cited = {k.strip() for g in re.findall(re.escape(BS) + r"cite\w*\{([^}]+)\}", body) for k in g.split(",")}
for k in sorted(cited - keys):
    print("UNDEFINED citation:", k); ok = False
print(f"{len(cited)} cited keys; {len(keys)} entries in refs.bib; uncited: {sorted(keys - cited)}")
labels = set(re.findall(re.escape(BS) + r"label\{([^}]+)\}", body))
refs = {r for g in re.findall(re.escape(BS) + r"(?:ref|eqref|cref)\{([^}]+)\}", body) for r in g.split(",")}
for r in sorted(refs - labels):
    print("UNDEFINED ref:", r); ok = False
for l in sorted(labels - refs):
    print("label never referenced in text:", l); ok = False
for g in re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    if not any((P / "figs" / (g + e)).exists() or (P / (g + e)).exists() for e in ("", ".pdf", ".png")):
        print("MISSING figure file:", g); ok = False
n_words = len(re.findall(r"\w+", re.sub(re.escape(BS) + r"\w+", " ", body)))
print(f"approx. {n_words} words of text; {'OK' if ok else 'PROBLEMS FOUND'}")
sys.exit(0 if ok else 1)
