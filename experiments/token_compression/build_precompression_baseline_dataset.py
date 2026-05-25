import argparse
import copy
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def expand_rule_text(rule_text: str) -> list[str]:
    if " THEN " not in rule_text:
        return [rule_text]

    prefix, actions_text = rule_text.split(" THEN ", 1)
    actions = [action.strip() for action in actions_text.split(",") if action.strip()]
    if len(actions) <= 1:
        return [rule_text]
    return [f"{prefix} THEN {action}" for action in actions]


def abnormal_rule_expansions(sample: dict) -> dict[str, list[str]]:
    segment = sample.get("independent_abnormal_segment") or {}
    source_rules = sample.get("source_abnormal_fragment_rules") or []
    rule_id = segment.get("rule_id")
    if not rule_id or not source_rules:
        return {}
    return {rule_id: source_rules}


def rebuild_rules(sample: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    special_expansions = abnormal_rule_expansions(sample)
    rebuilt_rules: dict[str, str] = {}
    rule_mapping: dict[str, list[str]] = {}
    next_index = 1

    for old_rule_id, rule_text in sample.get("rules", {}).items():
        expanded_rules = special_expansions.get(old_rule_id) or expand_rule_text(rule_text)
        new_rule_ids = []
        for expanded_rule in expanded_rules:
            new_rule_id = f"Rule_{next_index}"
            rebuilt_rules[new_rule_id] = expanded_rule
            new_rule_ids.append(new_rule_id)
            next_index += 1
        rule_mapping[old_rule_id] = new_rule_ids

    return rebuilt_rules, rule_mapping


def rebuild_ground_truth(sample: dict, rule_mapping: dict[str, list[str]]) -> dict:
    ground_truth = copy.deepcopy(sample.get("ground_truth", {}))
    original_involved = ground_truth.get("involved_rules", [])
    if not original_involved:
        ground_truth["involved_rules"] = []
        return ground_truth

    expanded_involved = []
    for old_rule_id in original_involved:
        expanded_involved.extend(rule_mapping.get(old_rule_id, [old_rule_id]))
    ground_truth["involved_rules"] = expanded_involved
    return ground_truth


def convert_sample(sample: dict) -> dict:
    converted = copy.deepcopy(sample)
    rebuilt_rules, rule_mapping = rebuild_rules(sample)

    converted["rules"] = rebuilt_rules
    converted["chain_length"] = len(rebuilt_rules)
    converted["ground_truth"] = rebuild_ground_truth(sample, rule_mapping)
    converted["precompression_source"] = {
        "source_sample_id": sample.get("sample_id"),
        "source_chain_id": sample.get("chain_id"),
        "source_visible_chain_length": sample.get("chain_length"),
        "visible_to_expanded_rule_ids": rule_mapping,
    }

    label = converted.get("label")
    if label == "abnormal":
        converted["description"] = (
            "A pre-compression abnormal baseline item reconstructed from the compressed evaluation sample."
        )
    else:
        converted["description"] = (
            "A pre-compression benign baseline item reconstructed from the compressed evaluation sample."
        )
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Build a pre-compression baseline dataset paired with the compressed token-compression eval set."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=BASE_DIR / "datasets" / "token_compression_eval_500_250.json",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=BASE_DIR / "datasets" / "token_compression_eval_500_250_before_compression.json",
    )
    args = parser.parse_args()

    source = load_json(args.input_file)
    converted_samples = [convert_sample(sample) for sample in source.get("samples", [])]

    output = {
        "metadata": {
            "dataset": "token_compression_eval_before_compression",
            "sample_count": len(converted_samples),
            "source_compressed_dataset": str(args.input_file.resolve()),
            "note": (
                "Expanded from the compressed evaluation set. Benign compressed rules are split into "
                "single-action rules; abnormal inserted segments are restored from source_abnormal_fragment_rules."
            ),
        },
        "samples": converted_samples,
    }
    write_json(args.output_file, output)
    print(f"Wrote {len(converted_samples)} pre-compression samples to {args.output_file}")


if __name__ == "__main__":
    main()
