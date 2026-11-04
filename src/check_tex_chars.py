"""Second LaTeX sanity pass: characters that silently break compilation.
Flags (per line): non-ASCII characters, unescaped % that is not a leading comment, unescaped _ or ^ or # in running text,
& outside tabular environments, and tab/control characters.   python src/check_tex_chars.py"""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "paper"
BS = chr(92)
bad = 0
for f in [P / "main.tex"] + sorted(P.glob("sec_*.tex")):
    in_tab = 0
    in_math = False
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        raw = line
        if re.search(r"[^\x00-\x7f]", raw):
            print(f"{f.name}:{n}: non-ASCII character"); bad += 1
        if "\t" in raw or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw):
            print(f"{f.name}:{n}: tab/control character"); bad += 1
        if re.match(r"\s*%", raw):
            continue                                   # whole-line comment
        if re.search(re.escape(BS) + r"begin\{(?:align|equation|eqnarray)\*?\}", raw):
            in_math = True
        if in_math:                                    # _ ^ & are legal inside math environments
            if re.search(re.escape(BS) + r"end\{(?:align|equation|eqnarray)\*?\}", raw):
                in_math = False
            continue
        # unescaped % that starts a comment in the middle of a line (usually a mistake in prose)
        for m in re.finditer(r"(?<!" + re.escape(BS) + r")%", raw):
            print(f"{f.name}:{n}: unescaped % -> rest of line is a comment: ...{raw[max(0, m.start() - 25):m.start() + 15]}"); bad += 1
        if BS + "begin{tabular" in raw:
            in_tab += 1
        EB = re.escape(BS)
        body = re.sub(EB + r"texttt\{[^}]*\}|" + EB + r"(?:label|ref|cite|includegraphics|url|graphicspath|input|bibliography\w*|usepackage|documentclass)(?:\[[^\]]*\])?\{[^}]*\}", "", raw)
        body = re.sub(r"\$[^$]*\$", "", body)           # inline math may contain _ ^ &
        if re.search(r"(?<!" + re.escape(BS) + r")[_#^]", body):
            print(f"{f.name}:{n}: unescaped _ ^ or # in text: {body.strip()[:90]}"); bad += 1
        if not in_tab and re.search(r"(?<!" + re.escape(BS) + r")&", body):
            print(f"{f.name}:{n}: & outside tabular: {body.strip()[:90]}"); bad += 1
        if BS + "end{tabular" in raw:
            in_tab -= 1
print("chars OK" if bad == 0 else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
