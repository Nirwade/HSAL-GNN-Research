"""
Scores prequel/sequel LLM extractions against the full-corpus structural
ground truth (ground_truth_full_corpus.csv, built by build_ground_truth.py
from the real r_sequel column).

Deliberately mirrors score_sample400.py / score_sequel400.py's exact scoring
methodology so full-corpus numbers are directly comparable to the 400-book
report numbers:
  - A parse failure (parse_status starts with "fail") is NOT excluded from
    scoring. The extraction scripts' own call_model() already forces
    has_prequel/has_sequel=False on failure, so a failed call is read and
    counted as a negative prediction, exactly like the 400-book scoring did.
    This matches the established methodology on purpose.
  - Iterates ground truth as the outer loop (same as the original scripts),
    so a "missing" count means a ground-truth item with no prediction yet
    (useful signal for an incomplete extraction run), not the reverse.
  - No cosine-similarity/coherence recomputation happens here — that's the
    separate matching stage (match_prequel.py / match_sequel.py), unchanged
    from the 400-book pipeline, run against the full-corpus extraction
    output via --test-file/--test-label.

One real evaluation-design difference from the 400-book report, worth
noting explicitly rather than glossing over: the 400-book ground truth
(prequel_sample_400.csv / sequel_sample_400.csv) has a "stratum" column
that separates out an "ambiguous" set with no reliable ground truth,
reported informationally rather than folded into P/R/F1. The full-corpus
ground truth (ground_truth_full_corpus.csv) has no such concept — every
one of the 8,108 items has a structurally-derived boolean label, so there
is no ambiguous carve-out here. The counting logic (TP/FP/TN/FN -> P/R/F1/
FNR/Specificity) is identical; the population being scored is different
(a full census vs. a stratified sample), which is exactly the intended
change (400-book sample -> full 8,108-book corpus).

This is a SEPARATE script from score_sample400.py / score_sequel400.py —
it does not read, write, or touch anything those use.

Usage:
    python3 score_full_corpus.py
"""

import csv
import os

GROUND_TRUTH_CSV = "ground_truth_full_corpus.csv"
OUTPUT_SUMMARY_CSV = "full_corpus_scoring_results.csv"

MODELS = ["llama3.1_8b", "mistral"]
VERSIONS = ["A", "B"]

RELATION_CONFIGS = {
    "prequel": {
        "dir": "prequel_extraction",
        "file_pattern": "prequels_v{version}_{model}_verified.csv",
        "gt_column": "ground_truth_has_prequel",
        "pred_column": "has_prequel",
    },
    "sequel": {
        "dir": "sequel_extraction",
        "file_pattern": "sequels_v{version}_{model}_verified.csv",
        "gt_column": "ground_truth_has_sequel",
        "pred_column": "has_sequel",
    },
}


def load_ground_truth(gt_column):
    gt = {}
    with open(GROUND_TRUTH_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["item_id"]] = row[gt_column].strip().lower() == "true"
    return gt


def load_predictions(path, pred_column):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # A failed call already has this column forced to "False" by
            # call_model() in the extraction script — no special-casing of
            # parse_status here, matching the original 400-book scripts.
            preds[row["item_id"]] = row[pred_column].strip().lower() == "true"
    return preds


def score(gt, preds, label):
    tp = fp = tn = fn = 0
    missing = 0
    for item_id, truth in gt.items():
        if item_id not in preds:
            missing += 1
            continue
        pred = preds[item_id]
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"\n=== {label} ===")
    print(f"  (missing from predictions file: {missing})")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"  FNR={fnr:.3f}  Specificity={specificity:.3f}")

    def r(x):
        return round(x, 4) if x == x else None  # NaN check

    return {
        "label": label, "status": "scored",
        "n_ground_truth": len(gt), "n_missing_predictions": missing,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": r(precision), "recall": r(recall), "f1": r(f1),
        "fnr": r(fnr), "specificity": r(specificity),
    }


def main():
    if not os.path.exists(GROUND_TRUTH_CSV):
        raise SystemExit(f"ERROR: ground truth file not found: {GROUND_TRUTH_CSV}")

    rows = []
    for relation, cfg in RELATION_CONFIGS.items():
        gt = load_ground_truth(cfg["gt_column"])
        for version in VERSIONS:
            for model in MODELS:
                path = os.path.join(cfg["dir"], cfg["file_pattern"].format(version=version, model=model))
                label = f"{relation} | v{version} | {model}"
                if not os.path.exists(path):
                    print(f"\n=== {label} ===\n  NOT FOUND yet: {path}")
                    rows.append({"label": label, "status": "not_found", "path": path})
                    continue
                preds = load_predictions(path, cfg["pred_column"])
                rows.append(score(gt, preds, label))

    fieldnames = ["label", "status", "path", "n_ground_truth", "n_missing_predictions",
                  "TP", "FP", "TN", "FN", "precision", "recall", "f1", "fnr", "specificity"]
    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
