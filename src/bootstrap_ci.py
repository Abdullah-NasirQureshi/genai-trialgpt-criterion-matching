"""Patient-level bootstrap (resample the 8 test patients with replacement, 2000 times) for accuracy / macro-F1.
Annotations of one patient stay together, so the interval reflects patient-to-patient variability.
python src/bootstrap_ci.py --tags base_zeroshot lora_final --split test -> results/bootstrap_ci_<split>.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import full_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=2000); a = ap.parse_args()
rng = np.random.default_rng(42)
out = {}
for tag in a.tags:
    preds = [json.loads(l) for l in open(ROOT / "results" / f"eval_{tag}_{a.split}.preds.jsonl", encoding="utf-8")]
    by = {}
    for p in preds: by.setdefault(p["patient_id"], []).append(p)
    pats = sorted(by)
    stats = {"overall_acc": [], "overall_macro_f1": [], "incl_macro_f1": [], "excl_macro_f1": [], "evid_micro_f1_all": []}
    for _ in range(a.n):
        samp = [q for pid in rng.choice(pats, len(pats), replace=True) for q in by[pid]]
        r = full_report(samp)
        stats["overall_acc"].append(r["overall"]["classification"]["accuracy"])
        stats["overall_macro_f1"].append(r["overall"]["classification"]["macro_f1"])
        stats["incl_macro_f1"].append(r["inclusion"]["classification"]["macro_f1"])
        stats["excl_macro_f1"].append(r["exclusion"]["classification"]["macro_f1"])
        stats["evid_micro_f1_all"].append(r["overall"]["evidence"]["all_rows"].get("micro_f1", float("nan")))
    out[tag] = {k: {"mean": float(np.nanmean(v)), "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]} for k, v in stats.items()}
    print(tag, {k: f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for k, v in out[tag].items()})
json.dump(out, open(ROOT / "results" / f"bootstrap_ci_{a.split}.json", "w"), indent=1)
