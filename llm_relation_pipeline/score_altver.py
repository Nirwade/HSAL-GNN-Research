import csv

GROUND_TRUTH_FILE = "alternative_version_eval_set_29.csv"
MODEL_FILES_BY_VERSION = {
    "A": {
        "llama3.1_8b": "alternative_version_extraction/altver_eval_llama3.1_8b_verified.csv",
        "mistral": "alternative_version_extraction/altver_eval_mistral_verified.csv",
    },
    "B": {
        "llama3.1_8b": "alternative_version_extraction/altver_eval_vB_llama3.1_8b_verified.csv",
        "mistral": "alternative_version_extraction/altver_eval_vB_mistral_verified.csv",
    },
}


def load_ground_truth():
    gt = {}
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["item_id"]] = row["ground_truth_has_alternative_version"].strip().lower() == "true"
    return gt


def load_predictions(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[row["item_id"]] = row["has_alternative_version"].strip().lower() == "true"
    return preds


def score(gt, preds, model_name):
    tp = fp = tn = fn = 0
    for item_id, truth in gt.items():
        if item_id not in preds:
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
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    print(f"\n=== {model_name} ===")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"  FNR={fnr:.3f}  Specificity={specificity:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["A", "B"], default="A")
    args = parser.parse_args()

    gt = load_ground_truth()
    print(f"Ground truth loaded: {len(gt)} books   (Version {args.version})")
    for name, path in MODEL_FILES_BY_VERSION[args.version].items():
        preds = load_predictions(path)
        score(gt, preds, name)


if __name__ == "__main__":
    main()