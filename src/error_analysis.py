"""Select and print distinct prediction errors for the error analysis.
python src/error_analysis.py --tag lora_final --split test [--per-cat 3]  -> results/error_analysis_<tag>_<split>.md
Categories (each shown with gold vs predicted label, cited vs gold evidence and the quoted sentences):
  A false_certainty  : gold = not enough information, model gave a decisive label
  B missed_info      : gold decisive, model said not enough information / not applicable
  C wrong_polarity   : decisive label flipped (e.g. included <-> not included, excluded <-> not excluded)
  D unsupported      : decisive prediction with NO cited evidence
  E wrong_evidence   : label right but cited sentences disjoint from the expert's
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECISIVE = {"included", "not included", "excluded", "not excluded"}
POL = {("included", "not included"), ("not included", "included"), ("excluded", "not excluded"), ("not excluded", "excluded")}


def category(p):
    g, q = p["gold_label"], p["pred_label"]
    if g == q:
        if g in DECISIVE and p["gold_evidence"] and not set(p["gold_evidence"]) & set(p["pred_evidence"]):
            return "E wrong_evidence"
        return None
    if g == "not enough information" and q in DECISIVE:
        return "A false_certainty"
    if g in DECISIVE and q in ("not enough information", "not applicable", None):
        return "B missed_info"
    if (g, q) in POL:
        return "C wrong_polarity"
    if q in DECISIVE and not p["pred_evidence"]:
        return "D unsupported"
    return "F other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-cat", type=int, default=3)
    a = ap.parse_args()
    preds = [json.loads(l) for l in open(ROOT / "results" / f"eval_{a.tag}_{a.split}.preds.jsonl", encoding="utf-8")]
    data = {r["annotation_id"]: r for r in (json.loads(l) for l in open(ROOT / "data" / f"{a.split}.jsonl", encoding="utf-8"))}
    # extra rows with wrong decisive prediction w/o evidence (category D) are checked separately
    by = {}
    for p in preds:
        c = category(p)
        if c is None and p["pred_label"] in DECISIVE and not p["pred_evidence"] and p["gold_label"] == p["pred_label"]:
            c = None
        if p["pred_label"] in DECISIVE and not p["pred_evidence"] and p["gold_label"] != p["pred_label"] and c in ("A false_certainty", "C wrong_polarity"):
            by.setdefault("D unsupported", []).append(p)
        if c:
            by.setdefault(c, []).append(p)
    lines = [f"# Error analysis ({a.tag}, {a.split})", ""]
    n_err = sum(p["gold_label"] != p["pred_label"] for p in preds)
    lines.append(f"{n_err} of {len(preds)} criterion predictions have a wrong label. Counts per category: " +
                 ", ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())) + "\n")
    shown, seen_pairs = 0, set()
    for cat in sorted(by):
        lines.append(f"## {cat}  ({len(by[cat])} cases)\n")
        picked, pats = 0, set()
        for p in by[cat]:
            key = (p["gold_label"], p["pred_label"], p["patient_id"])
            if key in seen_pairs or (p["patient_id"] in pats and picked < a.per_cat - 1 and len(by[cat]) > 8):
                continue
            seen_pairs.add(key); pats.add(p["patient_id"])
            r = data[p["annotation_id"]]; s = split_sentences(r["note"])
            q = lambda ids: "; ".join(f"[{i}] {s[i]}" for i in ids if i < len(s)) or "(none)"
            lines += [f"### Case {shown + 1}: annotation {p['annotation_id']} (patient {p['patient_id']}, {r['trial_id']}, {p['criterion_type']})",
                      f"- **Criterion:** {r['criterion_text']}",
                      f"- **Gold:** {p['gold_label']} | evidence {p['gold_evidence']}: {q(p['gold_evidence'])}",
                      f"- **Predicted:** {p['pred_label']} | evidence {p['pred_evidence']}: {q(p['pred_evidence'])}",
                      "- **Note:** " + " ".join(f"[{i}] {x}" for i, x in enumerate(s)),
                      "- **Why it failed:** _(to be filled in by hand in the report)_", ""]
            shown += 1; picked += 1
            if picked >= a.per_cat:
                break
    out = ROOT / "results" / f"error_analysis_{a.tag}_{a.split}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out, "cases shown:", shown)


if __name__ == "__main__":
    main()
