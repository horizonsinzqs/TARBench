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


def sorted_shard_files(shard_dir, pattern):
    files = sorted(Path(shard_dir).glob(pattern))
    if not files:
        raise ValueError(f"No shard files matched {pattern} in {shard_dir}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Merge parallel RAG shard outputs into one result file.")
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="shard*/rag_*.json")
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    merged = []
    for shard_file in sorted_shard_files(args.shard_dir, args.pattern):
        items = load_json(shard_file)
        if not isinstance(items, list):
            raise ValueError(f"Shard output is not a list: {shard_file}")
        merged.extend(items)

    for index, item in enumerate(merged, start=1):
        item["Sample_Index"] = index

    write_json(args.output_file, merged)
    print(f"Wrote merged output to {args.output_file}")


if __name__ == "__main__":
    main()
