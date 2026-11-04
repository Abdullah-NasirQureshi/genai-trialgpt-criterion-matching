"""Metrics for criterion classification and evidence-sentence retrieval.

Classification : accuracy, macro-F1, per-class P/R/F1, confusion matrix; overall and
                 per criterion type. Invalid model outputs (label=None) count as wrong
                 and are placed in an extra "<invalid>" column of the confusion matrix.
Macro-F1       : averaged over classes that occur in gold OR predictions for that slice
                 (classes with no gold and no predictions are ignored, never counted as 0).
Evidence       : sentence-level P/R/F1. Empty-set rule (documented in the paper):
                   gold empty & pred empty -> P=R=F1=1
                   gold empty & pred non-empty -> P=0, R=1 (nothing to miss), F1=0
                   gold non-empty & pred empty -> P=1 (nothing wrong claimed), R=0, F1=0
                 Reported (a) on all rows and (b) only on rows with gold evidence.
                 micro = pooled TP/FP/FN; macro = mean of per-row scores.
"""
from collections import Counter

import numpy as np

from prompts import INCLUSION_LABELS, EXCLUSION_LABELS, ALL_LABELS

INVALID = "<invalid>"


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification_report(gold, pred, labels):
    """gold/pred: lists of str (pred may contain None). Returns dict."""
    pred = [p if p is not None else INVALID for p in pred]
    n = len(gold)
    acc = float(np.mean([g == p for g, p in zip(gold, pred)])) if n else float("nan")
    per = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        p, r, f = _prf(tp, fp, fn)
        per[lab] = {"precision": p, "recall": r, "f1": f, "support": sum(g == lab for g in gold),
                    "predicted": sum(x == lab for x in pred)}
    present = [l for l in labels if per[l]["support"] > 0 or per[l]["predicted"] > 0]
    macro = float(np.mean([per[l]["f1"] for l in present])) if present else float("nan")
    cols = labels + [INVALID]
    cm = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in cols] for a in labels]
    return {"n": n, "accuracy": acc, "macro_f1": macro, "macro_f1_classes": present,
            "per_class": per, "confusion_matrix": {"rows_gold": labels, "cols_pred": cols, "matrix": cm},
            "invalid_outputs": sum(p == INVALID for p in pred)}


def evidence_row_scores(gold_ev, pred_ev):
    g, p = set(gold_ev), set(pred_ev)
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g:
        return 0.0, 1.0, 0.0
    if not p:
        return 1.0, 0.0, 0.0
    tp = len(g & p)
    pr, rc, f = _prf(tp, len(p - g), len(g - p))
    return pr, rc, f


def evidence_report(gold_evs, pred_evs):
    def agg(idx):
        if not idx:
            return {"n": 0}
        rows = [evidence_row_scores(gold_evs[i], pred_evs[i]) for i in idx]
        tp = sum(len(set(gold_evs[i]) & set(pred_evs[i])) for i in idx)
        fp = sum(len(set(pred_evs[i]) - set(gold_evs[i])) for i in idx)
        fn = sum(len(set(gold_evs[i]) - set(pred_evs[i])) for i in idx)
        mp, mr, mf = _prf(tp, fp, fn)
        return {"n": len(idx), "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
                "macro_precision": float(np.mean([r[0] for r in rows])),
                "macro_recall": float(np.mean([r[1] for r in rows])),
                "macro_f1": float(np.mean([r[2] for r in rows])),
                "empty_gold_empty_pred": sum(not gold_evs[i] and not pred_evs[i] for i in idx),
                "empty_gold": sum(not gold_evs[i] for i in idx),
                "empty_pred": sum(not pred_evs[i] for i in idx)}
    all_idx = list(range(len(gold_evs)))
    with_gold = [i for i in all_idx if gold_evs[i]]
    return {"all_rows": agg(all_idx), "rows_with_gold_evidence": agg(with_gold)}


def full_report(records):
    """records: list of dicts with criterion_type, gold_label, pred_label, gold_evidence, pred_evidence."""
    out = {}
    slices = {"overall": (ALL_LABELS, lambda r: True),
              "inclusion": (INCLUSION_LABELS, lambda r: r["criterion_type"] == "inclusion"),
              "exclusion": (EXCLUSION_LABELS, lambda r: r["criterion_type"] == "exclusion")}
    for name, (labels, sel) in slices.items():
        rs = [r for r in records if sel(r)]
        out[name] = {
            "classification": classification_report([r["gold_label"] for r in rs],
                                                    [r["pred_label"] for r in rs], labels),
            "evidence": evidence_report([r["gold_evidence"] for r in rs],
                                        [r["pred_evidence"] for r in rs]),
        }
    return out


def majority_baseline(records):
    """Accuracy of always predicting the most common training-style label per type (sanity)."""
    res = {}
    for t in ("inclusion", "exclusion"):
        c = Counter(r["gold_label"] for r in records if r["criterion_type"] == t)
        res[t] = c.most_common(1)[0] if c else None
    return res
