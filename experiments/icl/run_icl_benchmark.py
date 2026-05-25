import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from build_icl_library import build_library
from selector import ICLExampleSelector, render_examples, render_type_definitions


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


def ensure_library(path):
    if path.exists():
        return
    build_library(path.parent)


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


def build_sample_prompt_item(sample, selected_package, type_definition_limit):
    return json.dumps(
        {
            "sample_index": sample["global_sample_index"],
            "case_id": sample["chain_id"],
            "shot_count": selected_package["shot_count"],
            "anomaly_type_definitions": render_type_definitions(selected_package["type_definitions"][:type_definition_limit]),
            "benign_reference": render_examples(selected_package["benign_reference"]),
            "few_shot_examples": render_examples(selected_package["few_shot_examples"]),
            "target_sample": {
                "scenario": sample.get("scenario"),
                "scenario_group": sample.get("scenario_group"),
                "chain_length": sample.get("chain_length"),
                "rules": sample["rules"],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def build_zero_shot_prompt_item(sample):
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


def build_samples_text(batch, selector, shot_count, shot_mix, type_definition_limit):
    if shot_count == 0:
        packages = {
            sample["global_sample_index"]: {
                "type_definitions": [],
                "benign_reference": [],
                "few_shot_examples": [],
                "shot_count": 0,
                "shot_mix": {"benign": 0, "abnormal": 0},
            }
            for sample in batch
        }
        rendered = [build_zero_shot_prompt_item(sample) for sample in batch]
        return "[\n" + ",\n".join(rendered) + "\n]", packages

    rendered = []
    packages = {}
    for sample in batch:
        package = selector.select_package(sample, shot_count, shot_mix=shot_mix)
        packages[sample["global_sample_index"]] = package
        rendered.append(build_sample_prompt_item(sample, package, type_definition_limit))
    return "[\n" + ",\n".join(rendered) + "\n]", packages


def match_prediction(parsed_results, sample, position):
    if not isinstance(parsed_results, list):
        return None
    expected_index = sample["global_sample_index"]
    for item in parsed_results:
        if isinstance(item, dict) and item.get("sample_index") == expected_index:
            return item
    return parsed_results[position] if position < len(parsed_results) else None


def normalize_shot_mix(raw_shot_mix):
    return {
        int(key): {"benign": int(value["benign"]), "abnormal": int(value["abnormal"])}
        for key, value in raw_shot_mix.items()
    }


def run(args):
    config = load_runtime_config()
    model = args.model or config["model"]
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}. Supported: {', '.join(sorted(SUPPORTED_MODELS))}")

    dataset_path = Path(args.dataset_file).resolve() if args.dataset_file else resolve_path(config, "dataset_file")
    library_path = resolve_path(config, "library_file")
    shot_count = args.shot_count
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).resolve()
    elif shot_count == 0:
        prompt_path = BASE_DIR / "zero_shot_prompt.md"
    else:
        prompt_path = resolve_path(config, "prompt_file")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else resolve_path(config, "output_dir")
    sample_count = args.sample_count if args.sample_count is not None else config.get("sample_count")
    batch_size = args.batch_size or config.get("batch_size", 5)
    type_definition_limit = int(config.get("type_definition_limit", 14))
    shot_mix = normalize_shot_mix(config["shot_mix"])
    shot_mix.setdefault(0, {"benign": 0, "abnormal": 0})

    if shot_count != 0:
        ensure_library(library_path)
    dataset = load_json(dataset_path)
    if args.dataset_file and args.sample_count is None:
        sample_count = len(dataset.get("samples", []))
    samples = prepare_samples(dataset, sample_count)
    selector = ICLExampleSelector.from_json(library_path) if shot_count != 0 else None
    prompt_template = load_prompt_template(prompt_path)

    output_path = output_dir / f"icl_{safe_model_name(model)}_{shot_count}shot.json"
    evaluation_results = load_existing_results(output_path) if args.resume else []
    completed_count = len(evaluation_results)
    remaining_samples = samples[completed_count:]

    print(f"Prepared {len(samples)} ICL samples from {dataset_path}")
    print(f"Model: {model}; shot_count: {shot_count}; batch_size: {batch_size}; prompt: {prompt_path}")
    if completed_count:
        print(f"Resuming from existing output: {completed_count} samples already completed.")

    for batch_offset, batch in iter_batches(remaining_samples, batch_size):
        batch_index = batch_offset + (completed_count // batch_size)
        samples_text, packages = build_samples_text(batch, selector, shot_count, shot_mix, type_definition_limit)
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
            package = packages[sample["global_sample_index"]]
            prediction = match_prediction(parsed_results, sample, position)
            evaluation_results.append(
                {
                    "Model": model,
                    "Shot_Count": shot_count,
                    "Sample_Index": sample["global_sample_index"],
                    "Case_ID": sample["chain_id"],
                    "Label": sample["label"],
                    "Scenario_Group": sample.get("scenario_group"),
                    "Chain_Length": sample.get("chain_length"),
                    "Rules": format_rules_for_output(sample),
                    "Ground_Truth": sample.get("ground_truth", {}),
                    "ICL_Context": {
                        "type_definitions": render_type_definitions(package["type_definitions"][:type_definition_limit]),
                        "benign_reference": render_examples(package["benign_reference"]),
                        "few_shot_examples": render_examples(package["few_shot_examples"]),
                    },
                    "LLM_Prediction": format_prediction(prediction),
                    "Raw_Response": raw_content if not args.dry_run else "",
                }
            )
        write_json(output_path, evaluation_results)
        print(f"Completed batch {batch_index}")

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the TAPBench ICL benchmark through Poixe.")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), help="Poixe model name.")
    parser.add_argument("--dataset-file", help="Override the dataset JSON path.")
    parser.add_argument("--prompt-file", help="Override the prompt markdown path.")
    parser.add_argument("--output-dir", help="Override the output directory path.")
    parser.add_argument("--batch-size", type=int, help="Samples per API call.")
    parser.add_argument("--sample-count", type=int, help="Number of ICL samples to test.")
    parser.add_argument("--shot-count", type=int, choices=[0, 1, 3, 5], required=True, help="Few-shot count. Use 0 for the true zero-shot baseline.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts without calling the API.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file if present.")
    args = parser.parse_args()

    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
