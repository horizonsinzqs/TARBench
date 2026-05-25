#!/usr/bin/env python3
"""Build 5000 formatted abnormal samples from 1000 abnormal fragments."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENIGN = ROOT / "data" / "benign" / "benign_long_chains_gpt54.json"
DEFAULT_ABNORMAL_INPUTS = [
    ROOT / "data" / "abnormal_fragments" / "IoT_Abnormal_Dataset.json",
    ROOT / "data" / "abnormal_fragments" / "IoT_Multi-type_Anomalies.json",
]
DEFAULT_OUTPUT = ROOT / "data" / "abnormal" / "abnormal_chains_gpt54.json"
DEFAULT_EXPECTED_FRAGMENTS = 1000
DEFAULT_INSERTIONS_PER_FRAGMENT = 5

RULE_KEY_RE = re.compile(r"^Rule_(\d+)$")
RID_KEY_RE = re.compile(r"^R(\d+)_ID$")
RID_VALUE_RE = re.compile(r"\bR_\d+\b")
RULE_DUP_RE = re.compile(r"\b[Rr]ule\s+(Rule_\d+)\b")
REF_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

FIELDS_TO_REMOVE = {
    "source_fragment_file",
    "abnormal_devices",
    "abnormal_env_params",
    "candidate_benign_chain_count",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ordered_rule_values(record: dict[str, Any]) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for key, value in record.items():
        match = RULE_KEY_RE.match(key)
        if match:
            pairs.append((int(match.group(1)), str(value)))
    pairs.sort(key=lambda item: item[0])
    return [value for _, value in pairs]


def ordered_rule_numbers(record: dict[str, Any]) -> list[int]:
    rule_numbers: list[int] = []
    for key in record:
        match = RULE_KEY_RE.match(key)
        if match:
            rule_numbers.append(int(match.group(1)))
    return sorted(rule_numbers)


def is_environment_reference(obj_name: str) -> bool:
    return obj_name.startswith("environment_") or "_sensor" in obj_name


def extract_resources(rule_text: str) -> tuple[set[str], set[str]]:
    device_names: set[str] = set()
    env_params: set[str] = set()

    for obj_name, attr_name in REF_RE.findall(rule_text):
        if is_environment_reference(obj_name):
            env_params.add(f"{obj_name}.{attr_name}")
        else:
            device_names.add(obj_name)

    return device_names, env_params


def profile_rules(rule_texts: list[str]) -> tuple[set[str], set[str]]:
    devices: set[str] = set()
    env_params: set[str] = set()

    for rule_text in rule_texts:
        rule_devices, rule_env_params = extract_resources(rule_text)
        devices.update(rule_devices)
        env_params.update(rule_env_params)

    return devices, env_params


def load_benign_chains(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    benign_chains: list[dict[str, Any]] = []

    for chain in data.get("chains", []):
        rules = ordered_rule_values(chain)
        devices, env_params = profile_rules(rules)
        benign_chains.append(
            {
                "scenario": chain.get("scenario"),
                "description": chain.get("description"),
                "method": chain.get("method"),
                "scenario_group": chain.get("scenario_group"),
                "source_benign_chain_id": chain.get("chain_id"),
                "source_benign_chain_length": int(chain.get("chain_length", len(rules))),
                "rules": rules,
                "devices": devices,
                "env_params": env_params,
            }
        )

    return benign_chains


def load_abnormal_fragments(paths: list[Path]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []

    for path in paths:
        data = load_json(path)
        relative_path = str(path.relative_to(ROOT)).replace("\\", "/")
        for top_key, items in data.items():
            if top_key == "Metadata_Statistics" or not isinstance(items, list):
                continue

            for item in items:
                rules = ordered_rule_values(item)
                devices, env_params = profile_rules(rules)
                fragments.append(
                    {
                        "source_fragment_file": relative_path,
                        "source_fragment_group": top_key,
                        "Type": item.get("Type"),
                        "Type_ID": item.get("Type_ID"),
                        "Sub_Category": item.get("Sub_Category"),
                        "Description": item.get("Description"),
                        "rules": rules,
                        "devices": devices,
                        "env_params": env_params,
                    }
                )

    return fragments


def is_candidate_chain(fragment: dict[str, Any], benign_chain: dict[str, Any]) -> bool:
    if fragment["devices"] & benign_chain["devices"]:
        return False
    if fragment["env_params"] & benign_chain["env_params"]:
        return False
    return True


def build_output_chain(
    chain_idx: int,
    fragment: dict[str, Any],
    benign_chain: dict[str, Any],
    insert_pos: int,
    candidate_count: int,
) -> dict[str, Any]:
    benign_rules = benign_chain["rules"]
    fragment_rules = fragment["rules"]
    combined_rules = benign_rules[:insert_pos] + fragment_rules + benign_rules[insert_pos:]

    output_chain: dict[str, Any] = {
        "chain_id": f"AB_{chain_idx:06d}",
        "chain_type": "abnormal",
        "scenario": benign_chain.get("scenario"),
        "scenario_group": benign_chain.get("scenario_group"),
        "method": benign_chain.get("method"),
        "benign_description": benign_chain.get("description"),
        "source_benign_chain_id": benign_chain.get("source_benign_chain_id"),
        "source_benign_chain_length": benign_chain.get("source_benign_chain_length"),
        "source_fragment_file": fragment.get("source_fragment_file"),
        "source_fragment_group": fragment.get("source_fragment_group"),
        "Type": fragment.get("Type"),
        "Type_ID": fragment.get("Type_ID"),
        "Sub_Category": fragment.get("Sub_Category"),
        "Description": fragment.get("Description"),
        "abnormal_devices": sorted(fragment["devices"]),
        "abnormal_env_params": sorted(fragment["env_params"]),
        "candidate_benign_chain_count": candidate_count,
        "abnormal_fragment_start_index": insert_pos + 1,
        "abnormal_fragment_length": len(fragment_rules),
        "chain_length": len(combined_rules),
    }

    for idx, rule_text in enumerate(combined_rules, start=1):
        output_chain[f"Rule_{idx}"] = rule_text

    return output_chain


def build_abnormal_collections(
    fragments: list[dict[str, Any]],
    benign_chains: list[dict[str, Any]],
    insertions_per_fragment: int,
    seed: int,
    require_full_insertions: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    built_chains: list[dict[str, Any]] = []
    fragments_without_candidates: list[str] = []
    fragments_below_requirement: list[str] = []
    insertion_counts: list[int] = []
    next_chain_id = 1

    for fragment in fragments:
        candidates = [
            benign_chain
            for benign_chain in benign_chains
            if is_candidate_chain(fragment, benign_chain)
        ]
        candidate_count = len(candidates)
        type_id = str(fragment.get("Type_ID"))

        if candidate_count == 0:
            fragments_without_candidates.append(type_id)
            insertion_counts.append(0)
            continue

        if candidate_count < insertions_per_fragment:
            fragments_below_requirement.append(type_id)
            if require_full_insertions:
                continue

        selected_count = min(insertions_per_fragment, candidate_count)
        selected_chains = rng.sample(candidates, selected_count)
        insertion_counts.append(selected_count)

        for benign_chain in selected_chains:
            insert_pos = rng.randint(0, len(benign_chain["rules"]))
            built_chains.append(
                build_output_chain(
                    chain_idx=next_chain_id,
                    fragment=fragment,
                    benign_chain=benign_chain,
                    insert_pos=insert_pos,
                    candidate_count=candidate_count,
                )
            )
            next_chain_id += 1

    if require_full_insertions and fragments_below_requirement:
        preview = ", ".join(fragments_below_requirement[:10])
        raise ValueError(
            "Some abnormal fragments do not have enough benign candidates for "
            f"{insertions_per_fragment} insertions. Count={len(fragments_below_requirement)}. "
            f"Examples: {preview}"
        )

    stats = {
        "total_fragments": len(fragments),
        "total_built_chains": len(built_chains),
        "fragments_without_candidates": len(fragments_without_candidates),
        "fragment_ids_without_candidates": fragments_without_candidates,
        "fragments_below_required_insertions": len(fragments_below_requirement),
        "fragment_ids_below_required_insertions": fragments_below_requirement,
        "max_insertions_per_fragment": insertions_per_fragment,
        "actual_max_insertions_used": max(insertion_counts) if insertion_counts else 0,
    }
    return built_chains, stats


def build_source_index(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}

    for path in paths:
        data = load_json(path)
        relative_path = str(path.relative_to(ROOT)).replace("\\", "/")
        for group_name, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                type_id = item.get("Type_ID")
                if not type_id:
                    continue
                index[(relative_path, group_name, str(type_id))] = item

    return index


def build_rid_to_output_rule_map(
    source_item: dict[str, Any],
    abnormal_fragment_start_index: int,
) -> dict[str, str]:
    source_rule_numbers = ordered_rule_numbers(source_item)
    if not source_rule_numbers:
        return {}

    first_source_rule_number = min(source_rule_numbers)
    mapping: dict[str, str] = {}

    for key, value in source_item.items():
        match = RID_KEY_RE.match(key)
        if not match or not value:
            continue
        source_rule_number = int(match.group(1))
        output_rule_number = abnormal_fragment_start_index + (
            source_rule_number - first_source_rule_number
        )
        mapping[str(value)] = f"Rule_{output_rule_number}"

    return mapping


def build_sequential_description_map(
    description: str,
    abnormal_fragment_start_index: int,
) -> dict[str, str]:
    ordered_rids: list[str] = []
    seen: set[str] = set()

    for match in RID_VALUE_RE.finditer(description):
        rid = match.group(0)
        if rid in seen:
            continue
        seen.add(rid)
        ordered_rids.append(rid)

    return {
        rid: f"Rule_{abnormal_fragment_start_index + idx}"
        for idx, rid in enumerate(ordered_rids)
    }


def rewrite_description(
    description: str,
    source_item: dict[str, Any],
    abnormal_fragment_start_index: int,
) -> str:
    description_rids = [match.group(0) for match in RID_VALUE_RE.finditer(description)]
    if not description_rids:
        return description

    rid_to_rule = build_rid_to_output_rule_map(
        source_item=source_item,
        abnormal_fragment_start_index=abnormal_fragment_start_index,
    )
    if not rid_to_rule or any(rid not in rid_to_rule for rid in description_rids):
        rid_to_rule = build_sequential_description_map(
            description=description,
            abnormal_fragment_start_index=abnormal_fragment_start_index,
        )

    def replace_rid(match: re.Match[str]) -> str:
        rid = match.group(0)
        return rid_to_rule.get(rid, rid)

    updated = RID_VALUE_RE.sub(replace_rid, description)
    return RULE_DUP_RE.sub(r"\1", updated)


def build_abnormal_fragment_id_list(
    abnormal_fragment_start_index: int,
    abnormal_fragment_length: int,
) -> list[str]:
    return [
        f"R_{rule_number:04d}"
        for rule_number in range(
            abnormal_fragment_start_index,
            abnormal_fragment_start_index + abnormal_fragment_length,
        )
    ]


def insert_field_before(
    record: dict[str, Any],
    before_key: str,
    new_key: str,
    new_value: Any,
) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    inserted = False

    for key, value in record.items():
        if key == before_key:
            updated[new_key] = new_value
            inserted = True
        updated[key] = value

    if not inserted:
        updated[new_key] = new_value

    return updated


def format_chain(
    chain: dict[str, Any],
    source_index: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    formatted = dict(chain)
    source_key = (
        str(formatted.get("source_fragment_file", "")),
        str(formatted.get("source_fragment_group", "")),
        str(formatted.get("Type_ID", "")),
    )
    source_item = source_index.get(source_key)
    if source_item is None:
        raise KeyError(
            "Cannot find source abnormal fragment for "
            f"chain_id={formatted.get('chain_id')} with key={source_key!r}"
        )

    start_index = int(formatted.get("abnormal_fragment_start_index", 1))
    fragment_length = int(formatted.get("abnormal_fragment_length", 0))
    description = formatted.get("Description")
    if isinstance(description, str):
        formatted["Description"] = rewrite_description(
            description=description,
            source_item=source_item,
            abnormal_fragment_start_index=start_index,
        )

    formatted = insert_field_before(
        record=formatted,
        before_key="abnormal_fragment_start_index",
        new_key="abnormal_fragment_id_list",
        new_value=build_abnormal_fragment_id_list(start_index, fragment_length),
    )

    for field in FIELDS_TO_REMOVE:
        formatted.pop(field, None)

    return formatted


def format_dataset(
    data: dict[str, Any],
    source_index: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    formatted = dict(data)
    chains = formatted.get("chains", [])
    if not isinstance(chains, list):
        raise TypeError("Expected 'chains' to be a list.")
    formatted["chains"] = [format_chain(chain, source_index) for chain in chains]
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build formatted abnormal chains from the 800 single-abnormal and "
            "200 multi-abnormal fragments against the GPT-5.4 benign dataset."
        )
    )
    parser.add_argument(
        "--benign-input",
        type=Path,
        default=DEFAULT_BENIGN,
        help=f"Benign chain dataset path. Default: {DEFAULT_BENIGN}",
    )
    parser.add_argument(
        "--abnormal-inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_ABNORMAL_INPUTS,
        help="Abnormal fragment dataset paths.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--expected-fragments",
        type=int,
        default=DEFAULT_EXPECTED_FRAGMENTS,
        help=f"Expected merged abnormal fragment count. Default: {DEFAULT_EXPECTED_FRAGMENTS}",
    )
    parser.add_argument(
        "--insertions-per-fragment",
        type=int,
        default=DEFAULT_INSERTIONS_PER_FRAGMENT,
        help=f"Target insertions per abnormal fragment. Default: {DEFAULT_INSERTIONS_PER_FRAGMENT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible candidate sampling and insertion positions.",
    )
    parser.add_argument(
        "--allow-fewer-insertions",
        action="store_true",
        help="Allow fragments with fewer than the target candidate count to produce fewer outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benign_input = args.benign_input.resolve()
    abnormal_inputs = [path.resolve() for path in args.abnormal_inputs]
    output_path = args.output.resolve()

    benign_chains = load_benign_chains(benign_input)
    fragments = load_abnormal_fragments(abnormal_inputs)
    if len(fragments) != args.expected_fragments:
        raise ValueError(
            f"Expected {args.expected_fragments} abnormal fragments, found {len(fragments)}."
        )

    built_chains, stats = build_abnormal_collections(
        fragments=fragments,
        benign_chains=benign_chains,
        insertions_per_fragment=args.insertions_per_fragment,
        seed=args.seed,
        require_full_insertions=not args.allow_fewer_insertions,
    )

    raw_output = {
        "metadata": {
            "benign_input": str(benign_input),
            "abnormal_inputs": [str(path) for path in abnormal_inputs],
            "output_file": str(output_path),
            "selection_unit": "whole_benign_chain",
            "candidate_exclusion": {
                "device_match": "full_device_name",
                "environment_match": "sensor_and_environment_parameters",
            },
            "insertion_policy": "insert abnormal fragment contiguously at a random position in the benign chain",
            "abnormal_fragment_start_index_base": 1,
            "random_seed": args.seed,
            **stats,
        },
        "chains": built_chains,
    }

    source_index = build_source_index(abnormal_inputs)
    formatted_output = format_dataset(raw_output, source_index)
    dump_json(output_path, formatted_output)

    print(f"Loaded benign chains: {len(benign_chains)}")
    print(f"Loaded abnormal fragments: {len(fragments)}")
    print(f"Built abnormal chains: {formatted_output['metadata']['total_built_chains']}")
    print(
        "Fragments below required insertions: "
        f"{formatted_output['metadata']['fragments_below_required_insertions']}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
