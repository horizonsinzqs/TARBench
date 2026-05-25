import argparse
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "knowledge_base"

BENIGN_SOURCE = ROOT_DIR / "data" / "benign" / "benign_long_chains_gpt54.json"
ABNORMAL_SOURCE = ROOT_DIR / "data" / "abnormal" / "abnormal_chains_gpt54.json"
TYPE_DEFINITION_SOURCE = ROOT_DIR / "docs" / "abnormal_type_definitions.md"
RAG_DATASET = ROOT_DIR / "experiments" / "datasets" / "rag_dataset.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def extract_rules(chain):
    rule_keys = [key for key in chain if re.fullmatch(r"Rule_\d+", key)]
    rule_keys.sort(key=lambda key: int(key.split("_", 1)[1]))
    return {key: chain[key] for key in rule_keys}


def rule_range(start_index, length):
    if start_index is None or length is None:
        return []
    start = int(start_index)
    return [f"Rule_{index}" for index in range(start, start + int(length))]


def rules_to_text(rules):
    return "\n".join(f"{rule_id}: {rule}" for rule_id, rule in rules.items())


def split_action_target(rule):
    match = re.search(r"\bTHEN\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\s*=", rule)
    return match.group(1) if match else None


def collect_action_targets(rules):
    return sorted({target for target in (split_action_target(rule) for rule in rules.values()) if target})


def make_doc(doc_id, doc_type, title, text, metadata):
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": title,
        "text": text.strip(),
        "metadata": metadata,
    }


def parse_type_definitions(path):
    docs = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\.\s*([^:]+):\s*(.*)$", line)
        if match:
            if current:
                docs.append(current)
            current = {
                "index": int(match.group(1)),
                "name": match.group(2).strip(),
                "definition": match.group(3).strip(),
                "examples": [],
            }
            continue
        if current and line.startswith("- Example:"):
            current["examples"].append(line.removeprefix("- Example:").strip())
    if current:
        docs.append(current)

    type_docs = []
    for item in docs:
        aliases = []
        if item["name"] == "Implicit Physical Attack (Chaining)":
            aliases.append("Physical Cross-rule Interaction")
        text = [
            f"Anomaly Type: {item['name']}",
            f"Definition: {item['definition']}",
        ]
        if aliases:
            text.append(f"Dataset Alias: {', '.join(aliases)}")
        if item["examples"]:
            text.append("Examples: " + " ".join(item["examples"]))
        text.append("Diagnosis cue: only mark this type when the current rules contain direct logical evidence.")
        type_docs.append(
            make_doc(
                doc_id=f"TD_{item['index']:03d}",
                doc_type="abnormal_type_definition",
                title=item["name"],
                text="\n".join(text),
                metadata={
                    "type_name": item["name"],
                    "aliases": aliases,
                    "source": str(path),
                },
            )
        )
    return type_docs


def build_abnormal_case_docs(chains, excluded_chain_ids):
    docs = []
    for chain in chains:
        chain_id = chain.get("chain_id")
        if chain_id in excluded_chain_ids:
            continue
        rules = extract_rules(chain)
        involved_rule_ids = rule_range(
            chain.get("abnormal_fragment_start_index"),
            chain.get("abnormal_fragment_length"),
        )
        involved_rules = {rule_id: rules[rule_id] for rule_id in involved_rule_ids if rule_id in rules}
        text = "\n".join(
            [
                f"Abnormal Case: {chain_id}",
                f"Scenario: {chain.get('scenario')}",
                f"Scenario Group: {chain.get('scenario_group')}",
                f"Anomaly Type: {chain.get('Type')}",
                f"Sub Category: {chain.get('Sub_Category')}",
                f"Ground Truth Description: {chain.get('Description')}",
                f"Involved Rules: {', '.join(involved_rules)}",
                "Diagnostic Reading: the listed involved rules form the abnormal fragment; compare current rules for the same logical pattern before deciding.",
                "Involved Rule Text:",
                rules_to_text(involved_rules),
                "Full Rule Chain:",
                rules_to_text(rules),
            ]
        )
        docs.append(
            make_doc(
                doc_id=f"ACASE_{chain_id}",
                doc_type="abnormal_case",
                title=f"{chain.get('Type')} / {chain.get('Sub_Category')} / {chain_id}",
                text=text,
                metadata={
                    "chain_id": chain_id,
                    "scenario": chain.get("scenario"),
                    "scenario_group": chain.get("scenario_group"),
                    "method": chain.get("method"),
                    "chain_length": chain.get("chain_length", len(rules)),
                    "anomaly_type": chain.get("Type"),
                    "type_id": chain.get("Type_ID"),
                    "sub_category": chain.get("Sub_Category"),
                    "involved_rules": involved_rule_ids,
                    "action_targets": collect_action_targets(rules),
                    "source": str(ABNORMAL_SOURCE),
                },
            )
        )
    return docs


def build_benign_pattern_docs(chains, excluded_chain_ids):
    docs = []
    for chain in chains:
        chain_id = chain.get("chain_id")
        if chain_id in excluded_chain_ids:
            continue
        rules = extract_rules(chain)
        text = "\n".join(
            [
                f"Benign Pattern: {chain_id}",
                f"Scenario: {chain.get('scenario')}",
                f"Scenario Group: {chain.get('scenario_group')}",
                f"Description: {chain.get('description')}",
                "Benign Reading: this chain represents user-intended automation. Do not flag it unless the current sample adds clear conflict, impossible conditions, unsafe thresholds, or high-risk privilege changes.",
                "Rules:",
                rules_to_text(rules),
            ]
        )
        docs.append(
            make_doc(
                doc_id=f"BCASE_{chain_id}",
                doc_type="benign_pattern",
                title=f"{chain.get('scenario_group')} / {chain.get('scenario')} / {chain_id}",
                text=text,
                metadata={
                    "chain_id": chain_id,
                    "scenario": chain.get("scenario"),
                    "scenario_group": chain.get("scenario_group"),
                    "method": chain.get("method"),
                    "chain_length": chain.get("chain_length", len(rules)),
                    "action_targets": collect_action_targets(rules),
                    "source": str(BENIGN_SOURCE),
                },
            )
        )
    return docs


def build_contrastive_guidelines():
    guidelines = [
        (
            "CG_001",
            "Convenience automation is not automatically anomalous",
            "Motion, door, clock, light, temperature, humidity, and air-quality sensors commonly trigger lights, speakers, curtains, fans, purifiers, humidifiers, or cameras as normal smart-home convenience automation.",
        ),
        (
            "CG_002",
            "Same actuator requires simultaneous conflict evidence",
            "Multiple rules controlling the same device attribute are benign when their trigger conditions are mutually exclusive, time-separated, or scenario-specific. Flag only if the rules can fire together and require incompatible states.",
        ),
        (
            "CG_003",
            "Avoid privacy overthinking",
            "Turning on a camera or secure mode from a door, window, motion, smoke, or away-mode signal is usually a security feature, not an anomaly, unless it escalates from an unrelated low-trust trigger to a high-risk actuator.",
        ),
        (
            "CG_004",
            "Environmental comfort is usually benign",
            "Heating, cooling, fan, humidifier, dehumidifier, purifier, sprinkler, and window actions are normal if thresholds are plausible and actions do not physically block later required conditions.",
        ),
        (
            "CG_005",
            "Action duplication needs redundancy evidence",
            "Do not mark duplicated-looking actions unless the same actuator receives redundant identical commands under equivalent triggers or one action is strictly subsumed by a broader command.",
        ),
        (
            "CG_006",
            "Condition bypass needs missing-constraint evidence",
            "A simpler trigger is not automatically a bypass. Mark Condition Bypass only when one rule omits a necessary constraint that another rule establishes as safety- or logic-critical.",
        ),
        (
            "CG_007",
            "Rule chains need causal continuity",
            "For chained anomalies, involved rules must connect through condition-action dependencies or a shared physical channel. Do not connect unrelated rules solely because they appear in the same sample.",
        ),
        (
            "CG_008",
            "No evidence means no anomaly",
            "If retrieved cases are only loosely similar and current rules lack direct logical evidence, output has_anomaly=false and reason No anomalies identified.",
        ),
    ]
    return [
        make_doc(
            doc_id=doc_id,
            doc_type="contrastive_guideline",
            title=title,
            text=f"Guideline: {title}\n{text}",
            metadata={"source": "manual_contrastive_guideline"},
        )
        for doc_id, title, text in guidelines
    ]


def build_summary(docs, excluded_ids):
    doc_type_counts = Counter(doc["doc_type"] for doc in docs)
    abnormal_type_counts = Counter(
        doc["metadata"].get("anomaly_type")
        for doc in docs
        if doc["doc_type"] == "abnormal_case"
    )
    scenario_counts = Counter(
        doc["metadata"].get("scenario_group")
        for doc in docs
        if doc["doc_type"] in {"abnormal_case", "benign_pattern"}
    )
    return {
        "total_documents": len(docs),
        "doc_type_counts": dict(sorted(doc_type_counts.items())),
        "excluded_rag_chain_ids": len(excluded_ids),
        "abnormal_type_counts": dict(sorted((str(k), v) for k, v in abnormal_type_counts.items())),
        "scenario_group_counts": dict(sorted((str(k), v) for k, v in scenario_counts.items())),
        "source_files": {
            "benign": str(BENIGN_SOURCE),
            "abnormal": str(ABNORMAL_SOURCE),
            "type_definitions": str(TYPE_DEFINITION_SOURCE),
            "rag_dataset": str(RAG_DATASET),
        },
    }


def build(output_dir):
    rag_dataset = load_json(RAG_DATASET)
    excluded_ids = {sample["chain_id"] for sample in rag_dataset["samples"]}

    benign_chains = load_json(BENIGN_SOURCE)["chains"]
    abnormal_chains = load_json(ABNORMAL_SOURCE)["chains"]

    docs = []
    docs.extend(parse_type_definitions(TYPE_DEFINITION_SOURCE))
    docs.extend(build_contrastive_guidelines())
    docs.extend(build_abnormal_case_docs(abnormal_chains, excluded_ids))
    docs.extend(build_benign_pattern_docs(benign_chains, excluded_ids))
    docs.sort(key=lambda doc: (doc["doc_type"], doc["doc_id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "knowledge_base.jsonl", docs)
    write_json(output_dir / "knowledge_base_summary.json", build_summary(docs, excluded_ids))

    return docs


def main():
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base without leaking RAG test cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    docs = build(args.output_dir)
    print(f"Wrote {len(docs)} documents to {args.output_dir}")


if __name__ == "__main__":
    main()
