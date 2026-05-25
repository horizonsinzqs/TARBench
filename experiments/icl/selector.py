import json
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
RULE_ID_RE = re.compile(r"Rule_\d+")
DEFAULT_SHOT_MIX = {
    0: {"benign": 0, "abnormal": 0},
    1: {"benign": 0, "abnormal": 1},
    3: {"benign": 1, "abnormal": 2},
    5: {"benign": 2, "abnormal": 3},
}


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def extract_rules(sample):
    if "rules" in sample and isinstance(sample["rules"], dict):
        return sample["rules"]
    rule_keys = [key for key in sample if RULE_ID_RE.fullmatch(key)]
    rule_keys.sort(key=lambda key: int(key.split("_", 1)[1]))
    return {key: sample[key] for key in rule_keys}


def split_rule(rule):
    condition = ""
    action = ""
    if " THEN " in rule:
        condition, action = rule.split(" THEN ", 1)
    else:
        condition = rule
    condition_attrs = re.findall(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", condition)
    action_match = re.search(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\s*=", action)
    return condition_attrs, action_match.group(1) if action_match else None


def device_name(attribute_name):
    return attribute_name.split(".", 1)[0] if "." in attribute_name else attribute_name


def extract_features(sample):
    rules = extract_rules(sample)
    tokens = []
    action_targets = set()
    action_target_counts = Counter()
    devices = set()
    condition_attrs = set()
    for rule_id, rule in rules.items():
        tokens.extend(tokenize(f"{rule_id} {rule}"))
        attrs, target = split_rule(rule)
        condition_attrs.update(attrs)
        if target:
            action_targets.add(target)
            action_target_counts[target] += 1
        for attr in attrs:
            devices.add(device_name(attr))
        if target:
            devices.add(device_name(target))

    return {
        "rules": rules,
        "tokens": Counter(tokens + tokenize(sample.get("scenario", "")) + tokenize(sample.get("scenario_group", ""))),
        "scenario_group": sample.get("scenario_group"),
        "chain_length": int(sample.get("chain_length", len(rules))),
        "action_targets": action_targets,
        "repeated_action_targets": {target for target, count in action_target_counts.items() if count > 1},
        "devices": devices,
        "condition_attrs": condition_attrs,
    }


def example_features(example):
    return extract_features(
        {
            "rules": example["rules"],
            "scenario": example.get("scenario"),
            "scenario_group": example.get("scenario_group"),
            "chain_length": example.get("chain_length"),
        }
    )


def similarity_score(sample_features, example, cached_features):
    score = 0.0
    example_tokens = cached_features["tokens"]
    overlap_tokens = sum(min(sample_features["tokens"][token], example_tokens[token]) for token in sample_features["tokens"] if token in example_tokens)
    score += overlap_tokens * 0.01

    if example.get("scenario_group") == sample_features["scenario_group"]:
        score += 0.6

    example_targets = cached_features["action_targets"]
    example_devices = cached_features["devices"]
    score += min(len(sample_features["action_targets"] & example_targets) * 0.25, 1.0)
    score += min(len(sample_features["devices"] & example_devices) * 0.10, 0.5)

    try:
        distance = abs(int(example.get("chain_length", sample_features["chain_length"])) - sample_features["chain_length"])
        if distance <= 1:
            score += 0.2
        elif distance <= 3:
            score += 0.1
    except (TypeError, ValueError):
        pass

    if example.get("example_type") == "abnormal_example":
        anomaly_type = str(example.get("anomaly_type") or "").lower()
        if any(word in anomaly_type for word in ("conflict", "duplication", "mutex")) and sample_features["repeated_action_targets"]:
            score += 0.2
        if any(word in anomaly_type for word in ("loop", "block", "revert", "physical")) and sample_features["condition_attrs"] & sample_features["action_targets"]:
            score += 0.15

    return score


class ICLExampleSelector:
    def __init__(self, library):
        self.library = library
        self.type_definitions = library["type_definitions"]
        self.benign_examples = library["benign_examples"]
        self.abnormal_examples = library["abnormal_examples"]
        self._benign_features = {item["example_id"]: example_features(item) for item in self.benign_examples}
        self._abnormal_features = {item["example_id"]: example_features(item) for item in self.abnormal_examples}

    @classmethod
    def from_json(cls, path):
        return cls(load_json(path))

    def _top_examples(self, sample, pool, cached_map, count):
        features = extract_features(sample)
        scored = []
        for example in pool:
            score = similarity_score(features, example, cached_map[example["example_id"]])
            scored.append((score, example["example_id"], example))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:count]]

    def select_package(self, sample, shot_count, shot_mix=None):
        shot_mix = shot_mix or DEFAULT_SHOT_MIX
        if shot_count not in shot_mix:
            raise ValueError(f"Unsupported shot_count: {shot_count}. Supported: {sorted(shot_mix)}")

        mix = shot_mix[shot_count]
        benign_reference = self._top_examples(sample, self.benign_examples, self._benign_features, 1)
        selected_benign = self._top_examples(sample, self.benign_examples, self._benign_features, mix["benign"])
        selected_abnormal = self._top_examples(sample, self.abnormal_examples, self._abnormal_features, mix["abnormal"])

        return {
            "type_definitions": self.type_definitions,
            "benign_reference": benign_reference,
            "few_shot_examples": selected_benign + selected_abnormal,
            "shot_count": shot_count,
            "shot_mix": mix,
        }


def render_type_definitions(type_definitions):
    lines = []
    for item in type_definitions:
        lines.append(
            {
                "type_name": item["type_name"],
                "definition": item["definition"],
                "examples": item.get("examples", []),
            }
        )
    return lines


def render_examples(examples):
    rendered = []
    for index, example in enumerate(examples, start=1):
        rendered.append(
            {
                "example_index": index,
                "example_type": example["example_type"],
                "scenario": example.get("scenario"),
                "scenario_group": example.get("scenario_group"),
                "anomaly_type": example.get("anomaly_type"),
                "interpretation": example.get("interpretation"),
                "rules": example["rules"],
                "target_output": example["target_output"],
            }
        )
    return rendered
