#!/usr/bin/env python3
"""
validate_chains.py — LLM-Based Long Chain Quality Validator

Reads a long-chain JSON file (from Strategy A or Strategy B) and uses an LLM
to score each chain on a 1–5 scale.  Chains scoring below the threshold are
removed.  The validated output is written to a new file.

Can be used as:
  1. A standalone CLI tool
  2. An importable module (call validate_all / validate_file)

Usage
-----
    # Validate Strategy A output
    python validate_chains.py ^
        --input long_chains_graph.json ^
        --output long_chains_graph_validated.json ^
        --api-key YOUR_API_KEY

    # Validate Strategy B output with custom threshold
    python validate_chains.py ^
        --input long_chains_llm.json ^
        --output long_chains_llm_validated.json ^
        --api-key YOUR_API_KEY ^
        --score-threshold 4

    # Validate any long-chain file
    python validate_chains.py --input FILE --api-key KEY --score-threshold 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import requests


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LLM:
    """OpenAI-compatible LLM API client."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM output text."""
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            pass
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Validation logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_VALIDATE_SYS = """\
You are a quality auditor for an IoT interaction chain benchmark.

For each candidate multi-rule chain, score it 1–5:

  5 = Highly realistic, clear causal connections between every pair of rules
  4 = Realistic, plausible smart-home scenario
  3 = Somewhat plausible but a bit contrived
  2 = Weak causal logic or unlikely scenario
  1 = Incoherent or unrealistic

Output ONLY a JSON array. Each element:
{"index": <int>, "score": <int>, "comment": "brief reason"}
"""


def validate_batch(
    llm: LLM, batch: List[dict], threshold: int
) -> List[dict]:
    """Score a batch of chains via LLM; keep those >= threshold."""
    simplified = [
        {"index": i, "rules": c["rules"]}
        for i, c in enumerate(batch)
    ]
    raw = llm.chat(
        _VALIDATE_SYS,
        json.dumps(simplified, ensure_ascii=False),
        temperature=0.2,
    )
    results = _extract_json_array(raw)
    accepted = []
    for r in results:
        idx = r.get("index")
        score = r.get("score", 0)
        if idx is not None and 0 <= idx < len(batch) and score >= threshold:
            ch = batch[idx]
            ch["validation_score"] = score
            accepted.append(ch)
    return accepted


def validate_all(
    candidates: List[dict],
    llm: LLM,
    batch_size: int = 10,
    threshold: int = 3,
) -> List[dict]:
    """Validate all candidate chains in batches. Returns accepted chains."""
    total = len(candidates)
    print(
        f"  [Validate] {total} candidates, "
        f"batch_size={batch_size}, threshold>={threshold}"
    )
    accepted: List[dict] = []
    for i in range(0, total, batch_size):
        batch = candidates[i: i + batch_size]
        try:
            ok = validate_batch(llm, batch, threshold)
            accepted.extend(ok)
            print(
                f"    batch {i // batch_size + 1}: "
                f"{len(ok)}/{len(batch)} accepted"
            )
        except Exception as e:
            print(f"    batch {i // batch_size + 1}: ERROR {e}")
        time.sleep(1)
    print(f"  [Validate] Total accepted: {len(accepted)}/{total}")
    return accepted


def validate_file(
    input_path: str,
    output_path: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-chat",
    batch_size: int = 10,
    threshold: int = 3,
) -> dict:
    """Validate a long-chain JSON file end-to-end. Returns output dict."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    chains = data.get("chains", [])
    metadata = data.get("metadata", {})

    print(f"Loaded {len(chains)} chains from {input_path}")

    llm = LLM(api_key, base_url, model)
    accepted = validate_all(chains, llm, batch_size, threshold)

    # Score distribution
    score_dist: Dict[int, int] = defaultdict(int)
    for ch in accepted:
        score_dist[ch.get("validation_score", 0)] += 1

    metadata["validated"] = True
    metadata["validation_model"] = model
    metadata["validation_threshold"] = threshold
    metadata["pre_validation_count"] = len(chains)
    metadata["post_validation_count"] = len(accepted)
    metadata["validation_date"] = time.strftime("%Y-%m-%d")

    output = {"metadata": metadata, "chains": accepted}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Output: {len(accepted)} validated chains -> {output_path}")
    print(f"  Score distribution:")
    for s in sorted(score_dist):
        print(f"    score {s}: {score_dist[s]} chains")

    return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    ap = argparse.ArgumentParser(
        description="Validate long interaction chains via LLM scoring (1-5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Input long-chain JSON file to validate")
    ap.add_argument("--output", default=None,
                    help="Output file path (default: <input>_validated.json)")
    ap.add_argument("--api-key", default=None,
                    help="LLM API key (or set LLM_API_KEY env var)")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1",
                    help="OpenAI-compatible API base URL")
    ap.add_argument("--model", default="deepseek-chat",
                    help="Model name (default: deepseek-chat)")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Number of chains per LLM call (default: 10)")
    ap.add_argument("--score-threshold", type=int, default=3,
                    help="Min score to keep (1-5, default: 3)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        print(
            "ERROR: No API key provided.\n"
            "  Use --api-key YOUR_KEY or set LLM_API_KEY env var."
        )
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base = args.input.rsplit(".", 1)
        output_path = f"{base[0]}_validated.json" if len(base) > 1 else f"{args.input}_validated.json"

    print("=" * 64)
    print("  TAPBench — Long Chain Validator (LLM 1-5 Scoring)")
    print("=" * 64)

    validate_file(
        input_path=args.input,
        output_path=output_path,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        threshold=args.score_threshold,
    )

    print("=" * 64)


if __name__ == "__main__":
    main()
