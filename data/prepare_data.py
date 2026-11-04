"""Prepare the TrialGPT Criterion Annotations dataset.

* Patient-grouped split: sorted unique patient IDs -> numpy default_rng(42)
  shuffle -> first 37 train / next 8 val / remaining 8 test.
* Writes data/split_manifest.json and data/{train,val,test}.jsonl.
* The jsonl files contain ONLY fields that may be used as model input or gold
  labels. All gpt4_* fields and explanation_correctness are dropped so that
  previous model predictions can never leak into training or evaluation.
* The dataset's own `training` flag is ignored (we use the assignment split).
"""
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8
OUT = Path(__file__).resolve().parent

KEEP = [
    "annotation_id", "patient_id", "note", "trial_id", "trial_title",
    "criterion_type", "criterion_text", "expert_eligibility", "expert_sentences",
]

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]


def split_sentences(note: str) -> list[str]:
    """Return sentence texts indexed by their original ID (0-based, as in the data)."""
    parts = re.split(r"(?m)^(\d+)\. ", note)
    # parts = ['', '0', 'text', '1', 'text', ...]
    sents = {}
    for i in range(1, len(parts), 2):
        sents[int(parts[i])] = parts[i + 1].strip()
    n = max(sents) + 1
    assert sorted(sents) == list(range(n)), "non-contiguous sentence IDs"
    return [sents[i] for i in range(n)]


def main():
    df = load_dataset("ncbi/TrialGPT-Criterion-Annotations")["train"].to_pandas()

    patients = sorted(df.patient_id.unique())
    assert len(patients) == N_TRAIN + N_VAL + N_TEST, len(patients)
    rng = np.random.default_rng(SEED)
    shuffled = list(rng.permutation(patients))
    split_ids = {
        "train": sorted(shuffled[:N_TRAIN]),
        "val": sorted(shuffled[N_TRAIN:N_TRAIN + N_VAL]),
        "test": sorted(shuffled[N_TRAIN + N_VAL:]),
    }

    # ---- leakage checks -------------------------------------------------
    sets = {k: set(v) for k, v in split_ids.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "patient leakage between splits"
    assert set().union(*sets.values()) == set(patients)

    manifest = {"seed": SEED, "rng": "numpy.random.default_rng(42).permutation(sorted unique patient ids)",
                "counts": {}, "patients": split_ids}
    for name, ids in split_ids.items():
        part = df[df.patient_id.isin(ids)]
        assert part.patient_id.nunique() == len(ids)
        manifest["counts"][name] = {"patients": len(ids), "annotations": int(len(part))}
        rows = []
        for r in part.sort_values("annotation_id").to_dict("records"):
            rec = {k: r[k] for k in KEEP}
            # stored in the dataset as a JSON string such as "[0, 1, 2]"
            rec["expert_sentences"] = sorted(int(x) for x in json.loads(rec["expert_sentences"]))
            rec["annotation_id"] = int(rec["annotation_id"])
            sents = split_sentences(rec["note"])
            assert all(0 <= s < len(sents) for s in rec["expert_sentences"]), \
                f"evidence id out of range in annotation {rec['annotation_id']}"
            lbls = INCLUSION_LABELS if rec["criterion_type"] == "inclusion" else EXCLUSION_LABELS
            assert rec["expert_eligibility"] in lbls, rec["expert_eligibility"]
            rows.append(rec)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(name, manifest["counts"][name])

    assert sum(c["annotations"] for c in manifest["counts"].values()) == len(df)
    with open(OUT / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("OK: no patient leakage; manifest written")


if __name__ == "__main__":
    main()
