import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RULE_RE = re.compile(r"Rule_\d+")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def round_metric(value):
    return round(value, 6)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def rule_ids_from_prediction(value):
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return set(RULE_RE.findall(text))


def ground_truth_involved_rules(item):
    truth = item.get("Ground_Truth", {})
    if truth.get("has_anomaly") is not True:
        return set()
    return rule_ids_from_prediction(truth.get("involved_rules", []))


def localization_scores(predicted_rules, truth_rules):
    if not truth_rules:
        return {"exact_match": False, "overlap_precision": 0.0, "overlap_recall": 0.0, "overlap_f1": 0.0}
    overlap = predicted_rules & truth_rules
    precision = safe_divide(len(overlap), len(predicted_rules))
    recall = safe_divide(len(overlap), len(truth_rules))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "exact_match": predicted_rules == truth_rules,
        "overlap_precision": precision,
        "overlap_recall": recall,
        "overlap_f1": f1,
    }


def empty_counts():
    return {"total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0}


def add_confusion(counts, truth, prediction):
    counts["total"] += 1
    if prediction is None:
        counts["unknown"] += 1
    elif truth and prediction:
        counts["tp"] += 1
    elif truth and not prediction:
        counts["fn"] += 1
    elif not truth and prediction:
        counts["fp"] += 1
    else:
        counts["tn"] += 1


def finalize_counts(counts):
    predicted_known = counts["total"] - counts["unknown"]
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    return {
        **counts,
        "accuracy": round_metric(safe_divide(counts["tp"] + counts["tn"], predicted_known)),
        "precision": round_metric(safe_divide(counts["tp"], counts["tp"] + counts["fp"])),
        "recall": round_metric(safe_divide(counts["tp"], positives)),
        "specificity": round_metric(safe_divide(counts["tn"], negatives)),
        "false_positive_rate": round_metric(safe_divide(counts["fp"], negatives)),
        "false_negative_rate": round_metric(safe_divide(counts["fn"], positives)),
    }


def group_key(item, name):
    if name == "anomaly_type":
        return item.get("Ground_Truth", {}).get("anomaly_type") or "__benign__"
    if name == "sub_category":
        return item.get("Ground_Truth", {}).get("sub_category") or "__benign__"
    if name == "scenario_group":
        return item.get("Scenario_Group") or "__missing__"
    if name == "chain_length":
        return str(item.get("Chain_Length") or "__missing__")
    return "__unknown__"


def evaluate(items):
    overall = empty_counts()
    groups = {
        "anomaly_type": defaultdict(empty_counts),
        "sub_category": defaultdict(empty_counts),
        "scenario_group": defaultdict(empty_counts),
        "chain_length": defaultdict(empty_counts),
    }
    benign_total = 0
    benign_overthinking = 0
    localization_all = []
    localization_detected = []
    type_mentioned = 0
    abnormal_total = 0
    enriched = []

    for item in items:
        truth = item.get("Ground_Truth", {}).get("has_anomaly") is True
        prediction_obj = item.get("LLM_Prediction", {})
        prediction = normalize_bool(prediction_obj.get("has_anomaly"))
        reason = str(prediction_obj.get("reason", ""))
        predicted_rules = rule_ids_from_prediction(prediction_obj.get("involved_rules", []))
        truth_rules = ground_truth_involved_rules(item)

        add_confusion(overall, truth, prediction)
        for group_name, group_counts in groups.items():
            add_confusion(group_counts[group_key(item, group_name)], truth, prediction)

        item_eval = {
            "truth_has_anomaly": truth,
            "predicted_has_anomaly": prediction,
            "truth_involved_rules": sorted(truth_rules, key=lambda key: int(key.split("_")[1])),
            "predicted_involved_rules": sorted(predicted_rules, key=lambda key: int(key.split("_")[1])),
        }

        if truth:
            abnormal_total += 1
            loc = localization_scores(predicted_rules, truth_rules)
            localization_all.append(loc)
            if prediction is True:
                localization_detected.append(loc)
            anomaly_type = str(item.get("Ground_Truth", {}).get("anomaly_type") or "").lower()
            if anomaly_type and anomaly_type in reason.lower():
                type_mentioned += 1
            item_eval["localization"] = {key: round_metric(value) if isinstance(value, float) else value for key, value in loc.items()}
        else:
            benign_total += 1
            if prediction is True or (prediction is not None and reason.strip() != "No anomalies identified."):
                benign_overthinking += 1

        enriched.append({**item, "Evaluation": item_eval})

    def average_localization(records):
        if not records:
            return {"exact_match_rate": 0.0, "overlap_precision": 0.0, "overlap_recall": 0.0, "overlap_f1": 0.0}
        return {
            "exact_match_rate": round_metric(safe_divide(sum(1 for item in records if item["exact_match"]), len(records))),
            "overlap_precision": round_metric(sum(item["overlap_precision"] for item in records) / len(records)),
            "overlap_recall": round_metric(sum(item["overlap_recall"] for item in records) / len(records)),
            "overlap_f1": round_metric(sum(item["overlap_f1"] for item in records) / len(records)),
        }

    overall_metrics = finalize_counts(overall)
    metrics = {
        "overall": overall_metrics,
        "benign": {
            "total": benign_total,
            "false_positive_rate": overall_metrics["false_positive_rate"],
            "overthinking_rate": round_metric(safe_divide(benign_overthinking, benign_total)),
        },
        "abnormal": {
            "total": abnormal_total,
            "recall": overall_metrics["recall"],
            "type_mentioned_rate": round_metric(safe_divide(type_mentioned, abnormal_total)),
            "localization_all_abnormal": average_localization(localization_all),
            "localization_detected_abnormal": average_localization(localization_detected),
        },
        "groups": {
            group_name: {str(key): finalize_counts(counts) for key, counts in sorted(group_counts.items(), key=lambda item: str(item[0]))}
            for group_name, group_counts in groups.items()
        },
    }
    return metrics, enriched


def main():
    parser = argparse.ArgumentParser(description="Evaluate token-compression benchmark predictions.")
    parser.add_argument("prediction_file", type=Path)
    parser.add_argument("--write-enriched", action="store_true")
    args = parser.parse_args()

    items = load_json(args.prediction_file)
    metrics, enriched = evaluate(items)
    metrics_path = args.prediction_file.with_name(args.prediction_file.stem + "_metrics.json")
    write_json(metrics_path, metrics)
    print(f"Wrote {metrics_path}")
    if args.write_enriched:
        enriched_path = args.prediction_file.with_name(args.prediction_file.stem + "_evaluated.json")
        write_json(enriched_path, enriched)
        print(f"Wrote {enriched_path}")


if __name__ == "__main__":
    main()
