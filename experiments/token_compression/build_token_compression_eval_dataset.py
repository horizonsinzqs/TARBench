import argparse
import copy
import json
import random
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


def relabel_samples(samples, prefix: str):
    relabeled = []
    for sample in samples:
        item = copy.deepcopy(sample)
        item["source_sample_id"] = sample.get("sample_id")
        item["source_chain_id"] = sample.get("chain_id")
        relabeled.append(item)
    return relabeled


def main():
    parser = argparse.ArgumentParser(description="Build mixed token-compression eval dataset from benign and abnormal samples.")
    parser.add_argument("--benign-file", type=Path, default=BASE_DIR / "benign_dataset_500.json")
    parser.add_argument("--abnormal-file", type=Path, default=BASE_DIR / "abnormal_dataset_250.json")
    parser.add_argument("--output-file", type=Path, default=BASE_DIR / "datasets" / "token_compression_eval_500_250.json")
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    benign = load_json(args.benign_file).get("samples", [])
    abnormal = load_json(args.abnormal_file).get("samples", [])

    if len(benign) != 500:
        raise ValueError(f"Expected 500 benign samples, got {len(benign)}")
    if len(abnormal) != 250:
        raise ValueError(f"Expected 250 abnormal samples, got {len(abnormal)}")

    combined = relabel_samples(benign, "BENIGN") + relabel_samples(abnormal, "ABNORMAL")
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    for idx, sample in enumerate(combined, start=1):
        sample["sample_index"] = idx
        sample["mixed_index"] = idx

    output = {
        "metadata": {
            "dataset": "token_compression_eval",
            "benign_sample_count": 500,
            "abnormal_sample_count": 250,
            "sample_count": len(combined),
            "sampling_seed": args.seed,
            "benign_file": str(args.benign_file.resolve()),
            "abnormal_file": str(args.abnormal_file.resolve()),
        },
        "samples": combined,
    }
    write_json(args.output_file, output)
    print(f"Wrote mixed eval dataset to {args.output_file}")


if __name__ == "__main__":
    main()
