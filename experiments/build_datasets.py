import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"

BENIGN_SOURCE = ROOT_DIR / "data" / "benign" / "benign_long_chains_gpt54.json"
ABNORMAL_SOURCE = ROOT_DIR / "data" / "abnormal" / "abnormal_chains_gpt54.json"

SEED = 20260513

DATASET_SPECS = {
    "rag": {"benign": 2000, "abnormal": 1000},
    "icl": {"benign": 500, "abnormal": 250},
}


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


def value_for_key(chain, key):
    value = chain.get(key)
    if value is None or value == "":
        return "__missing__"
    return value


def make_bucket_key(chain, key_fields):
    return tuple(value_for_key(chain, key) for key in key_fields)


def diverse_sample(chains, sample_count, key_fields, seed, excluded_ids=None):
    excluded_ids = excluded_ids or set()
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for chain in chains:
        chain_id = chain.get("chain_id")
        if chain_id in excluded_ids:
            continue
        buckets[make_bucket_key(chain, key_fields)].append(chain)

    if sum(len(bucket) for bucket in buckets.values()) < sample_count:
        raise ValueError(f"Need {sample_count} samples, but the remaining pool is too small.")

    bucket_keys = sorted(buckets, key=lambda key: tuple(str(part) for part in key))
    for bucket in buckets.values():
        rng.shuffle(bucket)
    rng.shuffle(bucket_keys)

    selected = []
    selected_ids = set()
    while len(selected) < sample_count:
        added = False
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            while bucket and bucket[-1].get("chain_id") in selected_ids:
                bucket.pop()
            if not bucket:
                continue

            chain = bucket.pop()
            selected.append(chain)
            selected_ids.add(chain.get("chain_id"))
            added = True
            if len(selected) == sample_count:
                break

        if not added:
            raise ValueError(f"Only selected {len(selected)} samples out of {sample_count}.")

    return selected


def build_sample(chain, dataset_name, label, sample_index):
    rules = extract_rules(chain)
    sample = {
        "sample_index": sample_index,
        "sample_id": f"{dataset_name.upper()}_{label.upper()}_{sample_index:06d}",
        "chain_id": chain.get("chain_id"),
        "chain_type": chain.get("chain_type"),
        "label": label,
        "scenario": chain.get("scenario"),
        "scenario_group": chain.get("scenario_group"),
        "method": chain.get("method"),
        "chain_length": chain.get("chain_length", len(rules)),
        "rules": rules,
    }

    if label == "benign":
        sample["description"] = chain.get("description")
        sample["ground_truth"] = {
            "has_anomaly": False,
            "involved_rules": [],
            "anomaly_type": None,
        }
    else:
        sample["benign_description"] = chain.get("benign_description")
        sample["ground_truth"] = {
            "has_anomaly": True,
            "anomaly_type": chain.get("Type"),
            "type_id": chain.get("Type_ID"),
            "sub_category": chain.get("Sub_Category"),
            "description": chain.get("Description"),
            "abnormal_fragment_start_index": chain.get("abnormal_fragment_start_index"),
            "abnormal_fragment_length": chain.get("abnormal_fragment_length"),
            "abnormal_fragment_id_list": chain.get("abnormal_fragment_id_list", []),
        }

    return sample


def count_by(samples, key_path):
    counter = Counter()
    for sample in samples:
        current = sample
        for key in key_path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        counter[str(current if current is not None else "__missing__")] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def distribution_summary(samples):
    return {
        "label": count_by(samples, ["label"]),
        "chain_length": count_by(samples, ["chain_length"]),
        "scenario_group": count_by(samples, ["scenario_group"]),
        "method": count_by(samples, ["method"]),
        "abnormal_type": count_by(
            [sample for sample in samples if sample["label"] == "abnormal"],
            ["ground_truth", "anomaly_type"],
        ),
        "abnormal_sub_category": count_by(
            [sample for sample in samples if sample["label"] == "abnormal"],
            ["ground_truth", "sub_category"],
        ),
    }


def build_dataset(name, benign_chains, abnormal_chains):
    benign_samples = [
        build_sample(chain, name, "benign", index)
        for index, chain in enumerate(benign_chains, start=1)
    ]
    abnormal_samples = [
        build_sample(chain, name, "abnormal", index)
        for index, chain in enumerate(abnormal_chains, start=1)
    ]

    combined = benign_samples + abnormal_samples
    random.Random(SEED + len(name)).shuffle(combined)
    for index, sample in enumerate(combined, start=1):
        sample["mixed_index"] = index

    metadata = {
        "dataset": name,
        "sampling_seed": SEED,
        "source_files": {
            "benign": str(BENIGN_SOURCE),
            "abnormal": str(ABNORMAL_SOURCE),
        },
        "sample_counts": {
            "benign": len(benign_samples),
            "abnormal": len(abnormal_samples),
            "total": len(combined),
        },
        "sampling_policy": {
            "benign": "round-robin stratified by scenario_group, method, and chain_length",
            "abnormal": "round-robin stratified by Type, Sub_Category, chain_length, and scenario_group",
            "cross_dataset_overlap": "ICL samples are drawn after excluding RAG chain_ids.",
        },
        "distributions": distribution_summary(combined),
    }

    return {
        "metadata": metadata,
        "samples": combined,
    }, benign_samples, abnormal_samples


def write_split_file(name, label, samples):
    write_json(
        OUTPUT_DIR / f"{name}_{label}_{len(samples)}.json",
        {
            "metadata": {
                "dataset": name,
                "label": label,
                "sample_count": len(samples),
                "sampling_seed": SEED,
            },
            "samples": samples,
        },
    )


def main():
    benign_source = load_json(BENIGN_SOURCE)
    abnormal_source = load_json(ABNORMAL_SOURCE)
    benign_pool = benign_source["chains"]
    abnormal_pool = abnormal_source["chains"]

    benign_key_fields = ("scenario_group", "method", "chain_length")
    abnormal_key_fields = ("Type", "Sub_Category", "chain_length", "scenario_group")

    rag_benign = diverse_sample(
        benign_pool,
        DATASET_SPECS["rag"]["benign"],
        benign_key_fields,
        SEED + 101,
    )
    rag_abnormal = diverse_sample(
        abnormal_pool,
        DATASET_SPECS["rag"]["abnormal"],
        abnormal_key_fields,
        SEED + 201,
    )

    used_benign_ids = {chain["chain_id"] for chain in rag_benign}
    used_abnormal_ids = {chain["chain_id"] for chain in rag_abnormal}

    icl_benign = diverse_sample(
        benign_pool,
        DATASET_SPECS["icl"]["benign"],
        benign_key_fields,
        SEED + 301,
        excluded_ids=used_benign_ids,
    )
    icl_abnormal = diverse_sample(
        abnormal_pool,
        DATASET_SPECS["icl"]["abnormal"],
        abnormal_key_fields,
        SEED + 401,
        excluded_ids=used_abnormal_ids,
    )

    summary = {}
    for name, benign_chains, abnormal_chains in (
        ("rag", rag_benign, rag_abnormal),
        ("icl", icl_benign, icl_abnormal),
    ):
        dataset, benign_samples, abnormal_samples = build_dataset(name, benign_chains, abnormal_chains)
        write_json(OUTPUT_DIR / f"{name}_dataset.json", dataset)
        write_split_file(name, "benign", benign_samples)
        write_split_file(name, "abnormal", abnormal_samples)
        summary[name] = dataset["metadata"]

    write_json(
        OUTPUT_DIR / "dataset_summary.json",
        {
            "metadata": {
                "sampling_seed": SEED,
                "benign_source_total": len(benign_pool),
                "abnormal_source_total": len(abnormal_pool),
                "output_dir": str(OUTPUT_DIR),
            },
            "datasets": summary,
        },
    )

    print(f"Wrote datasets to {OUTPUT_DIR}")
    for name, metadata in summary.items():
        counts = metadata["sample_counts"]
        print(f"{name}: benign={counts['benign']}, abnormal={counts['abnormal']}, total={counts['total']}")


if __name__ == "__main__":
    main()
