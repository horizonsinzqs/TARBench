import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


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
            time.sleep(min(2 ** attempt, 20))
    return last_status, last_response


def prepare_samples(dataset, sample_count):
    samples = dataset["samples"][:sample_count] if sample_count else dataset["samples"]
    prepared = []
    for index, sample in enumerate(samples, start=1):
        item = dict(sample)
        item["global_sample_index"] = index
        prepared.append(item)
    return prepared


def load_existing_results(path):
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Existing output file is not a list: {path}")
    return data


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


def format_rules_for_output(sample):
    return [f"{rule_id}: {rule_text}" for rule_id, rule_text in sample["rules"].items()]


def build_sample_prompt_item(sample):
    return json.dumps(
        {
            "sample_index": sample["global_sample_index"],
            "case_id": sample["chain_id"],
            "scenario": sample.get("scenario"),
            "scenario_group": sample.get("scenario_group"),
            "chain_length": sample.get("chain_length"),
            "rules": sample["rules"],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_samples_text(batch):
    rendered = [build_sample_prompt_item(sample) for sample in batch]
    return "[\n" + ",\n".join(rendered) + "\n]"


def match_prediction(parsed_results, sample, position):
    if not isinstance(parsed_results, list):
        return None
    expected_index = sample["global_sample_index"]
    for item in parsed_results:
        if isinstance(item, dict) and item.get("sample_index") == expected_index:
            return item
    return parsed_results[position] if position < len(parsed_results) else None


def run(args):
    config = load_runtime_config()
    model = args.model or config["model"]
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    dataset_path = Path(args.dataset_file).resolve() if args.dataset_file else resolve_path(config, "dataset_file")
    prompt_path = Path(args.prompt_file).resolve() if args.prompt_file else resolve_path(config, "prompt_file")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolve_path(config, "output_dir")
    batch_size = args.batch_size or int(config.get("batch_size", 5))
    sample_count = args.sample_count if args.sample_count is not None else config.get("sample_count")

    dataset = load_json(dataset_path)
    if args.dataset_file and args.sample_count is None:
        sample_count = len(dataset.get("samples", []))
    samples = prepare_samples(dataset, sample_count)
    prompt_template = load_prompt_template(prompt_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    output_path = output_dir / f"token_compression_{safe_model_name(model)}{output_tag}.json"
    evaluation_results = load_existing_results(output_path) if args.resume else []
    completed_count = len(evaluation_results)
    remaining_samples = samples[completed_count:]

    if completed_count:
        print(f"Resuming from sample {completed_count + 1}.")
    print(f"Prepared {len(samples)} samples from {dataset_path}; prompt: {prompt_path}; dry_run: {args.dry_run}")

    for batch_number, batch in iter_batches(remaining_samples, batch_size):
        samples_text = build_samples_text(batch)
        user_prompt = prompt_template["user"].replace("{{SAMPLES}}", samples_text)
        if args.dry_run:
            content = ""
            parsed_results = []
        else:
            status_code, response = call_poixe_with_retries(config, model, prompt_template["system"], user_prompt)
            if status_code != 200:
                raise RuntimeError(f"Batch {batch_number} failed with status {status_code}: {response.get('error')}")

            content = response["choices"][0]["message"]["content"]
            parsed = parse_model_json(content)
            parsed_results = parsed.get("results") if isinstance(parsed, dict) else None

        for position, sample in enumerate(batch):
            matched = match_prediction(parsed_results, sample, position)
            evaluation_results.append(
                {
                    "Model": model,
                    "Sample_Index": sample["global_sample_index"],
                    "Case_ID": sample["chain_id"],
                    "Label": sample.get("label"),
                    "Scenario_Group": sample.get("scenario_group"),
                    "Chain_Length": sample.get("chain_length"),
                    "Rules": format_rules_for_output(sample),
                    "Ground_Truth": sample.get("ground_truth", {}),
                    "LLM_Prediction": format_prediction(matched),
                    "Raw_Response": content,
                }
            )

        write_json(output_path, evaluation_results)
        processed = completed_count + len(evaluation_results) - completed_count
        print(f"Completed batch {batch_number}; saved {len(evaluation_results)} samples to {output_path}")

    print(f"Finished {len(evaluation_results)} samples. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run token-compression anomaly benchmark.")
    parser.add_argument("--model", help="Override model from config.")
    parser.add_argument("--dataset-file", help="Override dataset file path.")
    parser.add_argument("--prompt-file", help="Override prompt markdown path.")
    parser.add_argument("--sample-count", type=int, help="Override number of evaluation samples.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--output-dir", help="Override the output directory path.")
    parser.add_argument("--output-tag", help="Optional tag appended to the output file name.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts without calling the API.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing partial output file if available.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
