import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Compare 1/3/5-shot ICL metrics.")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    summary = {"model": args.model, "shots": {}}
    for shot in (1, 3, 5):
        metrics_file = args.metrics_dir / f"icl_{args.model}_{shot}shot_metrics.json"
        metrics = load_json(metrics_file)
        summary["shots"][str(shot)] = {
            "accuracy": metrics["overall"]["accuracy"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "false_positive_rate": metrics["overall"]["false_positive_rate"],
            "overthinking_rate": metrics["benign"]["overthinking_rate"],
            "localization_overlap_f1": metrics["abnormal"]["localization_all_abnormal"]["overlap_f1"],
            "detected_localization_overlap_f1": metrics["abnormal"]["localization_detected_abnormal"]["overlap_f1"],
        }

    output_path = args.metrics_dir / f"icl_{args.model}_shot_comparison.json"
    write_json(output_path, summary)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
