import argparse
import copy
import json
import random
import re
from pathlib import Path


ENTITY_PATTERN = re.compile(r"([A-Za-z_]+)\.[A-Za-z_]+")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_rule_text(text: str) -> str:
    return text.replace("°C", "").strip()


def extract_entities(rule_text: str) -> set[str]:
    return set(ENTITY_PATTERN.findall(rule_text))


def compress_fragment_rules(rule_texts: list[str]) -> str:
    if not rule_texts:
        raise ValueError("Cannot compress an empty fragment.")

    first = sanitize_rule_text(rule_texts[0])
    if " THEN " not in first:
        raise ValueError(f"Invalid rule format: {first}")

    trigger = first.split("IF ", 1)[1].split(" THEN ", 1)[0].strip()
    ordered_actions: list[str] = []
    seen: set[str] = set()

    for rule_text in rule_texts:
        cleaned = sanitize_rule_text(rule_text)
        if " THEN " not in cleaned:
            continue
        action = cleaned.split(" THEN ", 1)[1].strip()
        if action and action not in seen:
            seen.add(action)
            ordered_actions.append(action)

    return f"IF {trigger} THEN {', '.join(ordered_actions)}"


def build_abnormal_candidates(abnormal_dataset: dict) -> list[dict]:
    candidates = []
    for chain in abnormal_dataset.get("chains", []):
        start = int(chain.get("abnormal_fragment_start_index", 1))
        length = int(chain.get("abnormal_fragment_length", 0))
        if length <= 0:
            continue

        fragment_rule_ids = [f"Rule_{idx}" for idx in range(start, start + length)]
        fragment_rule_texts = [chain.get(rule_id) for rule_id in fragment_rule_ids if chain.get(rule_id)]
        if not fragment_rule_texts:
            continue

        compressed = compress_fragment_rules(fragment_rule_texts)
        entity_set = set()
        for text in fragment_rule_texts:
            entity_set.update(extract_entities(sanitize_rule_text(text)))

        candidates.append(
            {
                "source_chain_id": chain.get("chain_id"),
                "type": chain.get("Type"),
                "sub_category": chain.get("Sub_Category"),
                "description": chain.get("Description"),
                "fragment_start_index": start,
                "fragment_length": length,
                "fragment_rule_ids": fragment_rule_ids,
                "fragment_rule_texts": [sanitize_rule_text(t) for t in fragment_rule_texts],
                "compressed_rule": compressed,
                "entities": entity_set,
            }
        )
    return candidates


def choose_independent_candidate(
    rng: random.Random, candidates: list[dict], main_entities: set[str], used_chain_ids: set[str]
) -> dict:
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    for candidate in shuffled:
        if candidate["source_chain_id"] in used_chain_ids:
            continue
        if candidate["entities"].isdisjoint(main_entities):
            used_chain_ids.add(candidate["source_chain_id"])
            return candidate

    for candidate in shuffled:
        if candidate["source_chain_id"] in used_chain_ids:
            continue
        used_chain_ids.add(candidate["source_chain_id"])
        return candidate

    raise RuntimeError("No abnormal candidates available.")


def sample_entities_from_rules(rules: dict) -> set[str]:
    entities = set()
    for text in rules.values():
        entities.update(extract_entities(text))
    return entities


def build_abnormal_sample(sample_index: int, benign_sample: dict, candidate: dict, rng: random.Random) -> dict:
    sample = copy.deepcopy(benign_sample)
    original_rules = list(sample["rules"].values())
    insert_at = rng.randint(0, len(original_rules))
    original_rules.insert(insert_at, candidate["compressed_rule"])

    remapped_rules = {f"Rule_{idx + 1}": text for idx, text in enumerate(original_rules)}
    abnormal_rule_id = f"Rule_{insert_at + 1}"

    sample["sample_index"] = sample_index
    sample["sample_id"] = f"TC_ABNORMAL_{sample_index:06d}"
    sample["chain_id"] = f"TCA_{sample_index:06d}"
    sample["chain_type"] = "abnormal"
    sample["label"] = "abnormal"
    sample["chain_length"] = len(remapped_rules)
    sample["rules"] = remapped_rules
    sample["description"] = (
        f"A token-compression abnormal item built from a benign compressed chain with one inserted "
        f"independent abnormal compressed segment ({candidate['type']})."
    )
    sample["ground_truth"] = {
        "has_anomaly": True,
        "involved_rules": [abnormal_rule_id],
        "anomaly_type": candidate["type"],
        "sub_category": candidate["sub_category"],
    }
    sample["source_abnormal_chain_id"] = candidate["source_chain_id"]
    sample["source_abnormal_fragment_length"] = candidate["fragment_length"]
    sample["source_abnormal_fragment_rule_ids"] = candidate["fragment_rule_ids"]
    sample["source_abnormal_fragment_rules"] = candidate["fragment_rule_texts"]
    sample["independent_abnormal_segment"] = {
        "rule_id": abnormal_rule_id,
        "compressed_rule": candidate["compressed_rule"],
        "attachment_type": "independent",
        "source_chain_id": candidate["source_chain_id"],
        "type": candidate["type"],
        "sub_category": candidate["sub_category"],
        "description": candidate["description"],
    }
    sample["mixed_index"] = sample_index
    return sample


def main():
    parser = argparse.ArgumentParser(description="Build token-compression abnormal dataset with independent segments.")
    parser.add_argument("--benign-dataset", type=Path, required=True)
    parser.add_argument("--abnormal-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    benign_dataset = load_json(args.benign_dataset)
    abnormal_source = load_json(args.abnormal_source)
    benign_samples = benign_dataset.get("samples", [])
    if len(benign_samples) < args.count:
        raise ValueError("Not enough benign compressed samples to build abnormal dataset.")

    rng = random.Random(args.seed)
    selected_benign = rng.sample(benign_samples, args.count)
    candidates = build_abnormal_candidates(abnormal_source)
    used_chain_ids: set[str] = set()

    samples = []
    for sample_index, benign_sample in enumerate(selected_benign, start=1):
        main_entities = sample_entities_from_rules(benign_sample["rules"])
        candidate = choose_independent_candidate(rng, candidates, main_entities, used_chain_ids)
        samples.append(build_abnormal_sample(sample_index, benign_sample, candidate, rng))

    output = {
        "metadata": {
            "dataset": "token_compression_abnormal",
            "label": "abnormal",
            "sample_count": len(samples),
            "target_sample_count": args.count,
            "sampling_seed": args.seed,
            "method": "token_compression_manual_with_independent_abnormal_segment",
            "benign_dataset": str(args.benign_dataset),
            "abnormal_source": str(args.abnormal_source),
            "abnormal_attachment_type": "independent",
        },
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(samples)} abnormal samples to {args.output}")


if __name__ == "__main__":
    main()
