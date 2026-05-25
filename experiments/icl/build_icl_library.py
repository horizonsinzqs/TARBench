import argparse
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "library"

BENIGN_SOURCE = ROOT_DIR / "data" / "benign" / "benign_long_chains_gpt54.json"
ABNORMAL_SOURCE = ROOT_DIR / "data" / "abnormal" / "abnormal_chains_gpt54.json"
TYPE_DEFINITION_SOURCE = ROOT_DIR / "docs" / "abnormal_type_definitions.md"
ICL_DATASET = ROOT_DIR / "experiments" / "datasets" / "icl_dataset.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_rules(chain):
    rule_keys = [key for key in chain if re.fullmatch(r"Rule_\d+", key)]
    rule_keys.sort(key=lambda key: int(key.split("_", 1)[1]))
    return {key: chain[key] for key in rule_keys}


def rules_to_lines(rules):
    return [f"{rule_id}: {text}" for rule_id, text in rules.items()]


def rule_range(start_index, length):
    if start_index is None or length is None:
        return []
    start = int(start_index)
    return [f"Rule_{index}" for index in range(start, start + int(length))]


def short_reason_for_abnormal(chain, involved_rules):
    anomaly_type = chain.get("Type") or "Anomaly"
    description = str(chain.get("Description") or "").strip()
    if description:
        description = description.replace("Chain formed by ", "").replace("result in ", "").strip()
    if not description:
        description = f"{anomaly_type} detected in the rule chain."
    joined_rules = " & ".join(involved_rules) if involved_rules else "relevant rules"
    return f"1.[{anomaly_type}]: {joined_rules} cause {description}"


def parse_type_definitions(path):
    entries = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\.\s*([^:]+):\s*(.*)$", line)
        if match:
            if current:
                entries.append(current)
            current = {
                "type_id": f"TD_{int(match.group(1)):03d}",
                "type_name": match.group(2).strip(),
                "definition": match.group(3).strip(),
                "examples": [],
            }
            continue
        if current and line.startswith("- Example:"):
            current["examples"].append(line.removeprefix("- Example:").strip())
    if current:
        entries.append(current)
    return entries


def build_example_item(chain, label):
    rules = extract_rules(chain)
    if label == "benign":
        return {
            "example_id": f"BENIGN_{chain['chain_id']}",
            "example_type": "benign_example",
            "chain_id": chain["chain_id"],
            "scenario": chain.get("scenario"),
            "scenario_group": chain.get("scenario_group"),
            "chain_length": chain.get("chain_length", len(rules)),
            "rules": rules,
            "interpretation": (
                "This is a user-intended benign automation chain. Convenience, comfort, safety, and security actions "
                "should not be flagged without direct logical conflict or unsafe behavior."
            ),
            "target_output": {
                "sample_index": 1,
                "has_anomaly": False,
                "reason": "No anomalies identified.",
                "involved_rules": [],
            },
        }

    involved_rules = rule_range(
        chain.get("abnormal_fragment_start_index"),
        chain.get("abnormal_fragment_length"),
    )
    return {
        "example_id": f"ABNORMAL_{chain['chain_id']}",
        "example_type": "abnormal_example",
        "chain_id": chain["chain_id"],
        "scenario": chain.get("scenario"),
        "scenario_group": chain.get("scenario_group"),
        "chain_length": chain.get("chain_length", len(rules)),
        "anomaly_type": chain.get("Type"),
        "sub_category": chain.get("Sub_Category"),
        "rules": rules,
        "interpretation": (
            "This is an abnormal chain. The listed involved rules are the core abnormal fragment and should be used "
            "as the explanation anchor."
        ),
        "target_output": {
            "sample_index": 1,
            "has_anomaly": True,
            "reason": short_reason_for_abnormal(chain, involved_rules),
            "involved_rules": involved_rules,
        },
    }


def build_library(output_dir):
    dataset = load_json(ICL_DATASET)
    excluded_ids = {sample["chain_id"] for sample in dataset["samples"]}

    benign_source = load_json(BENIGN_SOURCE)["chains"]
    abnormal_source = load_json(ABNORMAL_SOURCE)["chains"]

    type_definitions = parse_type_definitions(TYPE_DEFINITION_SOURCE)
    benign_examples = [
        build_example_item(chain, "benign")
        for chain in benign_source
        if chain["chain_id"] not in excluded_ids
    ]
    abnormal_examples = [
        build_example_item(chain, "abnormal")
        for chain in abnormal_source
        if chain["chain_id"] not in excluded_ids
    ]

    library = {
        "metadata": {
            "icl_dataset": str(ICL_DATASET),
            "excluded_chain_ids": len(excluded_ids),
            "sources": {
                "benign": str(BENIGN_SOURCE),
                "abnormal": str(ABNORMAL_SOURCE),
                "type_definitions": str(TYPE_DEFINITION_SOURCE),
            },
            "counts": {
                "type_definitions": len(type_definitions),
                "benign_examples": len(benign_examples),
                "abnormal_examples": len(abnormal_examples),
            },
        },
        "type_definitions": type_definitions,
        "benign_examples": benign_examples,
        "abnormal_examples": abnormal_examples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "icl_library.json", library)
    write_json(
        output_dir / "icl_library_summary.json",
        {
            "excluded_chain_ids": len(excluded_ids),
            "counts": library["metadata"]["counts"],
            "abnormal_type_counts": dict(
                sorted(
                    Counter(example.get("anomaly_type") for example in abnormal_examples).items(),
                    key=lambda item: str(item[0]),
                )
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Build the ICL example library without leaking ICL evaluation chains.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_library(args.output_dir)
    print(f"Wrote ICL library to {args.output_dir}")


if __name__ == "__main__":
    main()
