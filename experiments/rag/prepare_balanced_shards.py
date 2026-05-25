import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = (BASE_DIR / "../datasets/rag_dataset.json").resolve()
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def select_balanced_samples(dataset, benign_count, abnormal_count):
    benign = []
    abnormal = []
    for sample in dataset["samples"]:
        label = sample.get("label")
        if label == "benign" and len(benign) < benign_count:
            benign.append(sample)
        elif label == "abnormal" and len(abnormal) < abnormal_count:
            abnormal.append(sample)
        if len(benign) >= benign_count and len(abnormal) >= abnormal_count:
            break

    if len(benign) < benign_count or len(abnormal) < abnormal_count:
        raise ValueError(
            f"Unable to collect balanced subset: benign={len(benign)}/{benign_count}, abnormal={len(abnormal)}/{abnormal_count}"
        )
    return benign, abnormal


def chunk_round_robin(items, shard_count):
    shards = [[] for _ in range(shard_count)]
    for index, item in enumerate(items):
        shards[index % shard_count].append(item)
    return shards


def rebuild_samples(samples, shard_name):
    rebuilt = []
    for index, sample in enumerate(samples, start=1):
        item = dict(sample)
        item["sample_index"] = index
        item["mixed_index"] = index
        item["source_sample_id"] = sample.get("sample_id")
        item["source_mixed_index"] = sample.get("mixed_index")
        item["shard_name"] = shard_name
        rebuilt.append(item)
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced RAG subset shards for parallel execution.")
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benign-count", type=int, required=True)
    parser.add_argument("--abnormal-count", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--prefix", default="balanced_parallel")
    args = parser.parse_args()

    dataset = load_json(args.dataset_file)
    benign, abnormal = select_balanced_samples(dataset, args.benign_count, args.abnormal_count)
    benign_shards = chunk_round_robin(benign, args.shard_count)
    abnormal_shards = chunk_round_robin(abnormal, args.shard_count)

    summary = {
        "dataset_file": str(args.dataset_file.resolve()),
        "prefix": args.prefix,
        "benign_count": args.benign_count,
        "abnormal_count": args.abnormal_count,
        "shard_count": args.shard_count,
        "shards": [],
    }

    for shard_index in range(args.shard_count):
        shard_name = f"{args.prefix}_shard{shard_index + 1}"
        shard_samples = benign_shards[shard_index] + abnormal_shards[shard_index]
        shard_samples = rebuild_samples(shard_samples, shard_name)
        shard_file = args.output_dir / f"{shard_name}.json"
        write_json(
            shard_file,
            {
                "metadata": {
                    "dataset": shard_name,
                    "source_dataset": str(args.dataset_file.resolve()),
                    "sample_counts": {
                        "benign": len(benign_shards[shard_index]),
                        "abnormal": len(abnormal_shards[shard_index]),
                        "total": len(shard_samples),
                    },
                    "note": "Prepared for parallel RAG execution.",
                },
                "samples": shard_samples,
            },
        )
        summary["shards"].append(
            {
                "shard_name": shard_name,
                "file": str(shard_file.resolve()),
                "benign": len(benign_shards[shard_index]),
                "abnormal": len(abnormal_shards[shard_index]),
                "total": len(shard_samples),
            }
        )

    summary_file = args.output_dir / f"{args.prefix}_summary.json"
    write_json(summary_file, summary)
    print(f"Wrote shard summary to {summary_file}")


if __name__ == "__main__":
    main()
