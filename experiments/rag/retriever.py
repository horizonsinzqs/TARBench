import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
RULE_ID_RE = re.compile(r"Rule_\d+")


def load_jsonl(path):
    docs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


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
    target_match = re.search(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)\s*=", action)
    condition_attrs = re.findall(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", condition)
    action_target = target_match.group(1) if target_match else None
    return condition_attrs, action_target


def device_from_attr(attr):
    return attr.split(".", 1)[0] if "." in attr else attr


def extract_sample_features(sample):
    rules = extract_rules(sample)
    condition_attrs = set()
    action_targets = set()
    devices = set()
    text_parts = [
        sample.get("scenario", ""),
        sample.get("scenario_group", ""),
        sample.get("chain_type", ""),
        str(sample.get("chain_length", "")),
    ]

    for rule_id, rule in rules.items():
        attrs, target = split_rule(rule)
        condition_attrs.update(attrs)
        if target:
            action_targets.add(target)
        for attr in attrs:
            devices.add(device_from_attr(attr))
        if target:
            devices.add(device_from_attr(target))
        text_parts.append(f"{rule_id}: {rule}")

    repeated_targets = {
        target
        for target, count in Counter(
            split_rule(rule)[1]
            for rule in rules.values()
            if split_rule(rule)[1]
        ).items()
        if count > 1
    }

    return {
        "query_text": "\n".join(text_parts),
        "tokens": set(tokenize("\n".join(text_parts))),
        "condition_attrs": condition_attrs,
        "action_targets": action_targets,
        "repeated_action_targets": repeated_targets,
        "devices": devices,
        "scenario_group": sample.get("scenario_group"),
        "chain_length": sample.get("chain_length", len(rules)),
    }


class KnowledgeBaseRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.doc_tokens = []
        self.doc_term_counts = []
        self.doc_norms = []
        self.inverted = defaultdict(set)
        self.by_type = defaultdict(list)
        self.idf = {}
        self._build_index()

    @classmethod
    def from_jsonl(cls, path):
        return cls(load_jsonl(path))

    def _build_index(self):
        document_frequency = Counter()
        for index, doc in enumerate(self.docs):
            text = f"{doc.get('title', '')}\n{doc.get('text', '')}\n{json.dumps(doc.get('metadata', {}), ensure_ascii=False)}"
            counts = Counter(tokenize(text))
            self.doc_term_counts.append(counts)
            self.doc_tokens.append(set(counts))
            self.by_type[doc["doc_type"]].append(index)
            for token in counts:
                document_frequency[token] += 1
                self.inverted[token].add(index)

        doc_count = max(len(self.docs), 1)
        self.idf = {
            token: math.log((doc_count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        for counts in self.doc_term_counts:
            norm = math.sqrt(sum((count * self.idf.get(token, 1.0)) ** 2 for token, count in counts.items()))
            self.doc_norms.append(norm or 1.0)

    def _candidate_indexes(self, query_tokens, doc_type):
        indexes = set()
        for token in query_tokens:
            indexes.update(self.inverted.get(token, ()))
        typed = set(self.by_type.get(doc_type, ()))
        indexes &= typed
        if not indexes:
            indexes = typed
        return indexes

    def _text_score(self, query_tokens, doc_index):
        if not query_tokens:
            return 0.0
        counts = self.doc_term_counts[doc_index]
        score = 0.0
        query_norm = 0.0
        for token in query_tokens:
            weight = self.idf.get(token, 1.0)
            query_norm += weight * weight
            if token in counts:
                score += weight * counts[token] * weight
        return score / ((math.sqrt(query_norm) or 1.0) * self.doc_norms[doc_index])

    def _metadata_score(self, features, doc):
        metadata = doc.get("metadata", {})
        score = 0.0

        if metadata.get("scenario_group") and metadata.get("scenario_group") == features["scenario_group"]:
            score += 0.35

        try:
            sample_length = int(features["chain_length"])
            doc_length = int(metadata.get("chain_length", sample_length))
            if abs(sample_length - doc_length) <= 1:
                score += 0.15
            elif abs(sample_length - doc_length) <= 3:
                score += 0.05
        except (TypeError, ValueError):
            pass

        doc_targets = set(metadata.get("action_targets") or [])
        if doc_targets:
            shared_targets = doc_targets & features["action_targets"]
            shared_devices = {device_from_attr(target) for target in doc_targets} & features["devices"]
            score += min(len(shared_targets) * 0.20, 0.80)
            score += min(len(shared_devices) * 0.05, 0.30)

        anomaly_type = str(metadata.get("anomaly_type") or metadata.get("type_name") or "").lower()
        if anomaly_type:
            repeated = features["repeated_action_targets"]
            if repeated and any(token in anomaly_type for token in ("conflict", "duplication", "mutex")):
                score += 0.20
            if features["condition_attrs"] & features["action_targets"] and any(
                token in anomaly_type for token in ("loop", "revert", "block", "physical")
            ):
                score += 0.10

        return score

    def retrieve_type(self, sample, doc_type, top_k):
        features = extract_sample_features(sample)
        query_tokens = features["tokens"]
        scored = []
        for doc_index in self._candidate_indexes(query_tokens, doc_type):
            doc = self.docs[doc_index]
            score = self._text_score(query_tokens, doc_index) + self._metadata_score(features, doc)
            scored.append((score, doc_index))
        scored.sort(key=lambda item: (-item[0], self.docs[item[1]]["doc_id"]))
        return [
            {
                "score": round(score, 6),
                "doc": self.docs[doc_index],
            }
            for score, doc_index in scored[:top_k]
        ]

    def retrieve(self, sample, top_k_by_type):
        return {
            doc_type: self.retrieve_type(sample, doc_type, top_k)
            for doc_type, top_k in top_k_by_type.items()
            if top_k > 0
        }


def compact_doc(hit, max_chars=900):
    doc = hit["doc"]
    text = doc["text"]
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return {
        "doc_id": doc["doc_id"],
        "doc_type": doc["doc_type"],
        "title": doc["title"],
        "score": hit["score"],
        "text": text,
        "metadata": doc.get("metadata", {}),
    }


def compact_retrieval(retrieval, max_chars=900):
    return {
        doc_type: [compact_doc(hit, max_chars=max_chars) for hit in hits]
        for doc_type, hits in retrieval.items()
    }
