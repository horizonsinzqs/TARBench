import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from build_knowledge_base import build as build_knowledge_base
from retriever import KnowledgeBaseRetriever, compact_retrieval


SUPPORTED_MODELS = {
    "gpt-5.4",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen3-32b",
}

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_path(config, key):
    value = Path(config[key])
    if value.is_absolute():
        return value
    return (BASE_DIR / value).resolve()


def load_runtime_config():
    config = load_json(CONFIG_PATH)
    api_config_path = config.get("api_config_file")
    if api_config_path:
        resolved_api_config = (BASE_DIR / api_config_path).resolve()
        api_config = load_json(resolved_api_config)
        merged = {**api_config, **config}
        merged["api_config_file"] = str(resolved_api_config)
        config = merged
    config.setdefault("api_key", os.getenv("TARBENCH_API_KEY", ""))
    config.setdefault("base_url", os.getenv("TARBENCH_BASE_URL", ""))
    if not config.get("api_key"):
        config["api_key"] = os.getenv("TARBENCH_API_KEY", "")
    if not config.get("base_url"):
        config["base_url"] = os.getenv("TARBENCH_BASE_URL", "")
    return config


def extract_section(text, heading):
    lines = text.splitlines()
    marker = f"## {heading}"
    start = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Missing section: {marker}")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def load_prompt_template(path):
    text = Path(path).read_text(encoding="utf-8")
    return {
        "system": extract_section(text, "System Prompt"),
        "user": extract_section(text, "User Prompt"),
    }


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start // batch_size + 1, items[start : start + batch_size]


def safe_model_name(model):
    return model.replace("/", "_").replace(":", "_").replace(" ", "_")


def parse_model_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def call_poixe(config, model, system_prompt, user_prompt):
    url = config["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not model.startswith("claude-"):
        payload["temperature"] = config.get("temperature", 0)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenAI/Python 1.0.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, {"error": body}
    except http.client.RemoteDisconnected as exc:
        return 0, {"error": str(exc)}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}
    except OSError as exc:
        return 0, {"error": str(exc)}


def call_poixe_with_retries(config, model, system_prompt, user_prompt):
    max_retries = int(config.get("max_retries", 5))
    retryable_statuses = {0, 408, 409, 429, 500, 502, 503, 504}
    last_status = 0
    last_response = {"error": "Unknown error"}

    for attempt in range(max_retries):
        status_code, response = call_poixe(config, model, system_prompt, user_prompt)
        last_status = status_code
        last_response = response
        if status_code == 200:
            return status_code, response
        if status_code not in retryable_statuses:
            return status_code, response
        if attempt < max_retries - 1:
            delay_seconds = min(2 ** attempt, 20)
            print(
                f"Retrying model call after batch error (status={status_code}, attempt={attempt + 1}/{max_retries}, sleep={delay_seconds}s)..."
            )
            time.sleep(delay_seconds)

    return last_status, last_response


def format_rules_for_output(sample):
    return [f"{rule_id}: {rule_text}" for rule_id, rule_text in sample["rules"].items()]


def format_prediction(item):
    if not isinstance(item, dict):
        return {
            "has_anomaly": None,
            "reason": "Model output could not be parsed.",
            "involved_rules": [],
        }
    return {
        "has_anomaly": item.get("has_anomaly"),
        "reason": item.get("reason", ""),
        "involved_rules": item.get("involved_rules", []),
    }


def retrieved_doc_refs(compact_context):
    refs = []
    for doc_type, docs in compact_context.items():
        for doc in docs:
            refs.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_type": doc_type,
                    "title": doc["title"],
                    "score": doc["score"],
                }
            )
    return refs


def build_sample_prompt_item(sample, compact_context=None, include_retrieved_context=True):
    prompt_sample = {
        "sample_index": sample["global_sample_index"],
        "case_id": sample["chain_id"],
        "scenario": sample.get("scenario"),
        "scenario_group": sample.get("scenario_group"),
        "chain_length": sample.get("chain_length"),
        "rules": sample["rules"],
    }
    if include_retrieved_context:
        prompt_sample["retrieved_context"] = compact_context or {}
    return json.dumps(prompt_sample, ensure_ascii=False, indent=2)


def build_samples_text(batch, retriever, top_k, max_chars, include_retrieved_context=True):
    if not include_retrieved_context:
        retrievals = {sample["global_sample_index"]: {} for sample in batch}
        rendered = [build_sample_prompt_item(sample, include_retrieved_context=False) for sample in batch]
        return "[\n" + ",\n".join(rendered) + "\n]", retrievals

    rendered = []
    retrievals = {}
    for sample in batch:
        retrieval = retriever.retrieve(sample, top_k)
        compact_context = compact_retrieval(retrieval, max_chars=max_chars)
        retrievals[sample["global_sample_index"]] = compact_context
        rendered.append(build_sample_prompt_item(sample, compact_context))
    return "[\n" + ",\n".join(rendered) + "\n]", retrievals


def match_prediction(parsed_results, sample, position):
    if not isinstance(parsed_results, list):
        return None
    expected_index = sample["global_sample_index"]
    for item in parsed_results:
        if isinstance(item, dict) and item.get("sample_index") == expected_index:
            return item
    return parsed_results[position] if position < len(parsed_results) else None


def prepare_samples(dataset, sample_count):
    samples = dataset["samples"][:sample_count] if sample_count else dataset["samples"]
    prepared = []
    for index, sample in enumerate(samples, start=1):
        item = dict(sample)
        item["global_sample_index"] = index
        prepared.append(item)
    return prepared


def prepare_balanced_samples(dataset, benign_count, abnormal_count):
    selected = []
    benign_seen = 0
    abnormal_seen = 0
    for sample in dataset["samples"]:
        label = sample.get("label")
        if label == "benign" and benign_seen < benign_count:
            selected.append(sample)
            benign_seen += 1
        elif label == "abnormal" and abnormal_seen < abnormal_count:
            selected.append(sample)
            abnormal_seen += 1

        if benign_seen >= benign_count and abnormal_seen >= abnormal_count:
            break

    if benign_seen < benign_count or abnormal_seen < abnormal_count:
        raise ValueError(
            f"Unable to collect requested balanced subset: benign={benign_seen}/{benign_count}, abnormal={abnormal_seen}/{abnormal_count}"
        )

    prepared = []
    for index, sample in enumerate(selected, start=1):
        item = dict(sample)
        item["global_sample_index"] = index
        prepared.append(item)
    return prepared


def ensure_knowledge_base(path):
    if path.exists():
        return
    build_knowledge_base(path.parent)


def load_existing_results(path):
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Existing output file is not a list: {path}")
    return data


def run(args):
    config = load_runtime_config()
    model = args.model or config["model"]
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}. Supported: {', '.join(sorted(SUPPORTED_MODELS))}")

    dataset_path = Path(args.dataset_file).resolve() if args.dataset_file else resolve_path(config, "dataset_file")
    knowledge_base_path = resolve_path(config, "knowledge_base_file")
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).resolve()
    elif args.without_rag:
        prompt_path = BASE_DIR / "rag_without_rag_prompt.md"
    else:
        prompt_path = resolve_path(config, "prompt_file")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolve_path(config, "output_dir")
    sample_count = args.sample_count if args.sample_count is not None else config.get("sample_count")
    batch_size = args.batch_size or config.get("batch_size", 5)
    top_k = config["top_k"]
    max_chars = int(config.get("retrieved_doc_max_chars", 900))

    if args.rebuild_kb and args.without_rag:
        raise ValueError("--rebuild-kb cannot be used together with --without-rag.")
    if args.rebuild_kb and knowledge_base_path.exists():
        knowledge_base_path.unlink()
    if not args.without_rag:
        ensure_knowledge_base(knowledge_base_path)

    dataset = load_json(dataset_path)
    if args.dataset_file and args.sample_count is None and args.benign_count is None and args.abnormal_count is None:
        sample_count = len(dataset.get("samples", []))
    if args.benign_count is not None or args.abnormal_count is not None:
        if args.benign_count is None or args.abnormal_count is None:
            raise ValueError("Both --benign-count and --abnormal-count must be provided together.")
        samples = prepare_balanced_samples(dataset, args.benign_count, args.abnormal_count)
    else:
        samples = prepare_samples(dataset, sample_count)
    prompt_template = load_prompt_template(prompt_path)
    retriever = None if args.without_rag else KnowledgeBaseRetriever.from_jsonl(knowledge_base_path)

    if retriever is not None:
        print(f"Loaded {len(retriever.docs)} knowledge documents.")
    else:
        print("Running without retrieval context.")
    print(f"Prepared {len(samples)} RAG samples from {dataset_path}.")
    print(f"Model: {model}; batch_size: {batch_size}; dry_run: {args.dry_run}; prompt: {prompt_path}")

    output_suffix = "_dry_run" if args.dry_run else ""
    inferred_tag = "without_rag" if args.without_rag else ""
    output_tag_value = args.output_tag or inferred_tag
    output_tag = f"_{output_tag_value}" if output_tag_value else ""
    output_path = output_dir / f"rag_{safe_model_name(model)}{output_tag}{output_suffix}.json"
    evaluation_results = load_existing_results(output_path) if args.resume else []
    completed_count = len(evaluation_results)
    if completed_count:
        print(f"Resuming from existing output: {completed_count} samples already completed.")
    if completed_count > len(samples):
        raise ValueError(
            f"Existing output has {completed_count} samples, which exceeds current target size {len(samples)}."
        )
    remaining_samples = samples[completed_count:]

    for batch_offset, batch in iter_batches(remaining_samples, batch_size):
        batch_index = batch_offset + (completed_count // batch_size)
        samples_text, retrievals = build_samples_text(
            batch,
            retriever,
            top_k,
            max_chars,
            include_retrieved_context=not args.without_rag,
        )
        user_prompt = prompt_template["user"].replace("{samples_text}", samples_text)
        raw_content = ""

        if args.dry_run:
            parsed_results = []
        else:
            status_code, response = call_poixe_with_retries(config, model, prompt_template["system"], user_prompt)
            if status_code == 200:
                choices = response.get("choices", [])
                if choices:
                    raw_content = choices[0].get("message", {}).get("content", "")
            parsed_json = parse_model_json(raw_content) if raw_content else None
            if status_code != 200:
                raise RuntimeError(f"Batch {batch_index} failed with status {status_code}: {response.get('error')}")
            if not isinstance(parsed_json, dict) or not isinstance(parsed_json.get("results"), list):
                raise RuntimeError(f"Batch {batch_index} did not return parseable results.")
            parsed_results = parsed_json["results"]

        for position, sample in enumerate(batch):
            compact_context = retrievals[sample["global_sample_index"]]
            prediction = match_prediction(parsed_results, sample, position)
            evaluation_results.append(
                {
                    "Model": model,
                    "Sample_Index": sample["global_sample_index"],
                    "Case_ID": sample["chain_id"],
                    "Label": sample["label"],
                    "Scenario_Group": sample.get("scenario_group"),
                    "Chain_Length": sample.get("chain_length"),
                    "Rules": format_rules_for_output(sample),
                    "Ground_Truth": sample.get("ground_truth", {}),
                    "Retrieved_Documents": retrieved_doc_refs(compact_context),
                    "Retrieved_Context": compact_context,
                    "LLM_Prediction": format_prediction(prediction),
                    "Raw_Response": raw_content if not args.dry_run else "",
                }
            )
        write_json(output_path, evaluation_results)
        print(f"Completed batch {batch_index}")

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the TAPBench RAG benchmark through Poixe.")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), help="Poixe model name.")
    parser.add_argument("--dataset-file", help="Override the dataset JSON path.")
    parser.add_argument("--prompt-file", help="Override the prompt markdown path.")
    parser.add_argument("--output-dir", help="Override the output directory path.")
    parser.add_argument("--batch-size", type=int, help="Samples per API call.")
    parser.add_argument("--sample-count", type=int, help="Number of RAG samples to test.")
    parser.add_argument("--benign-count", type=int, help="Number of benign samples to select from the dataset.")
    parser.add_argument("--abnormal-count", type=int, help="Number of abnormal samples to select from the dataset.")
    parser.add_argument("--output-tag", help="Optional tag appended to the output file name.")
    parser.add_argument("--without-rag", action="store_true", help="Run the no-retrieval baseline on the same dataset/output schema.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare retrieval outputs without calling the API.")
    parser.add_argument("--rebuild-kb", action="store_true", help="Rebuild the knowledge base before running.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file if present.")
    args = parser.parse_args()

    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
