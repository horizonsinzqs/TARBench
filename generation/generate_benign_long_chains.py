#!/usr/bin/env python3
"""
generate_benign_long_chains.py — LLM-Assisted Long Chain Generator

Reads single-rule benign chains from single_rule_chains.json and combines
them into multi-rule chains (2–10 rules) via LLM-assisted scenario combination.

Here, **exact signature matching**
(action 4-tuple == next trigger 4-tuple) is **not** required. The LLM may
order rules into a multi-step chain whenever the **overall scenario** is
reasonable (same routine, thematic coherence, indirect physical causation,
etc.).

Enhanced 2-stage deduplication pipeline:
  1. Exact fingerprint dedup
  2. Source-overlap dedup (Jaccard >= 0.8)

Optional quality scoring / filtering can be applied offline with
`validate_long_chains.py`; the generator does not invoke it.

After **each successful LLM round**, raw candidates are flushed to a
**checkpoint file** (default: next to `--output`, `*.checkpoint.json`) so an
interrupted run still keeps partial chains. Use `--no-checkpoint` to disable.

Usage
-----
    python generate_benign_long_chains.py ^
        --input data/benign/benign_single_rule_chains.json ^
        --ontology resources/device_ontology.json ^
        --api-key YOUR_API_KEY ^
        --base-url https://api.deepseek.com/v1 ^
        --model deepseek-chat

    # Dry-run (syntax check, no LLM calls):
    python generate_benign_long_chains.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "benign" / "benign_single_rule_chains.json"
DEFAULT_ONTOLOGY_PATH = PROJECT_ROOT / "resources" / "device_ontology.json"

# 默认输出目录（与 --output 默认值配套）
DEFAULT_RESULT_DIR = str(PROJECT_ROOT / "outputs" / "generation")
DEFAULT_OUTPUT_BASENAME = "long_chains_llm"

# 跨场景链在 scenario_group 中的统一标记
CROSS_SCENARIO_GROUP = "__cross_scenario__"
CROSS_SCENARIO_SCENE_COUNT = 4
CROSS_SCENARIO_CHAINS_PER_SCENE = 10
CROSS_SCENARIO_MAX_TARGET_PER_ROUND = 20
CROSS_SCENARIO_MAX_ZERO_ROUNDS = 3
CROSS_SCENARIO_MAX_ERROR_ROUNDS = 3
CROSS_SCENARIO_REQUEST_RETRIES = 2


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _default_output_path() -> str:
    return os.path.join(
        DEFAULT_RESULT_DIR,
        f"{DEFAULT_OUTPUT_BASENAME}_{_timestamp_str()}.json",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ontology-aware parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Ontology:
    """设备与房间本体知识库，从 TARBench ontology JSON 加载。"""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.devices: List[str] = sorted(
            raw["Device_type"].keys(), key=len, reverse=True
        )
        self.rooms: List[str] = sorted(
            raw["Room_type"], key=len, reverse=True
        )
        self.device_attrs: Dict[str, List[str]] = raw["Device_type"]
        self.value_ranges: Dict[str, list] = raw["value_range"]

    def parse_ref(self, text: str) -> Optional[Dict[str, str]]:
        dot = text.find(".")
        if dot == -1:
            return None
        prefix, attr = text[:dot], text[dot + 1:]
        for room in self.rooms:
            if prefix.startswith(room + "_"):
                dev = prefix[len(room) + 1:]
                if dev in self.devices:
                    return {"room": room, "device": dev, "attribute": attr}
        return None

    def parse_trigger(self, s: str) -> Optional[dict]:
        m = re.match(r"(.+?)\s*(==|>=|<=|!=|>|<)\s*(.+)", s.strip())
        if not m:
            return None
        ref = self.parse_ref(m.group(1).strip())
        if ref:
            ref["op"] = m.group(2)
            ref["value"] = m.group(3).strip()
        return ref

    def parse_action(self, s: str) -> Optional[dict]:
        m = re.match(r"(.+?)\s*=\s*(.+)", s.strip())
        if not m:
            return None
        ref = self.parse_ref(m.group(1).strip())
        if ref:
            ref["value"] = m.group(2).strip()
        return ref

    def inventory_text(self) -> str:
        """生成人类可读的设备清单文本，用于嵌入到 LLM 提示词中。"""
        lines = []
        for dev in sorted(self.device_attrs.keys()):
            attrs = self.device_attrs[dev]
            parts = []
            for a in attrs:
                vr = self.value_ranges.get(a, ["?"])
                parts.append(f"{a}: [{', '.join(str(v) for v in vr)}]")
            lines.append(f"- {dev}: {'; '.join(parts)}")
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LLM:
    """大语言模型 API 客户端，兼容 OpenAI 接口规范。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 600,
    ):
        self.key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, temperature: float | None = 0.7) -> str:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if temperature is not None:
            body["temperature"] = temperature
        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _extract_json_array(text: str) -> list:
    """从 LLM 输出文本中提取 JSON 数组。"""
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            pass
    return []


def _normal_length_quotas(
    n_chains: int,
    lo: int = 2,
    hi: int = 10,
    mu: float = 6.0,
    sigma: float = 2.0,
) -> Dict[int, int]:
    """整数长度 lo..hi 上的离散权重 ~ N(mu, sigma)，总和为 n_chains。"""
    if n_chains <= 0:
        return {}
    weights: List[float] = []
    for L in range(lo, hi + 1):
        weights.append(math.exp(-0.5 * ((L - mu) / sigma) ** 2))
    s = sum(weights)
    raw = [n_chains * w / s for w in weights]
    floors = [int(x) for x in raw]
    rem = n_chains - sum(floors)
    fracs = sorted([(raw[i] - floors[i], i) for i in range(len(raw))], reverse=True)
    for k in range(rem):
        floors[fracs[k][1]] += 1
    return {lo + i: floors[i] for i in range(len(floors))}


def _build_length_guidance(n: int, lo: int = 2, hi: int = 10) -> str:
    """本批请求的链数 → 人类可读的长度配额（近似正态）。"""
    if n <= 0:
        return ""
    q = _normal_length_quotas(n, lo=lo, hi=hi)
    lines = "\n".join(
        f"- **{L} rules per chain**: ~{q[L]} chains" for L in sorted(q.keys())
    )
    return f"""### Chain length targets for this batch (2–{hi} rules per chain)
Follow a **bell-shaped (normal-like) distribution**: peak around **5–7** rules,
fewer chains at **2–3** and **9–{hi}**. Approximate counts for your **{n}** chains:
{lines}
Each chain must have between **{lo}** and **{hi}** rules (inclusive)."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM-assisted combination
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COMBINE_SYS = """\
You are an IoT smart-home automation designer.

Your task is to create coherent **multi-rule interaction chains** of **2–10
rules** each (see per-batch length targets in the user message).

Here you only need **ordered rules**, or create **bridge** rules where needed, to
form a **plausible single user story**: same routine, logical flow, thematic
coherence, or indirect physical causation.


### Rules

1. Every rule MUST follow:
   IF {room}_{device}.{attribute} == {value} THEN {room}_{device}.{attribute} = {value}
2. All devices and attributes MUST come from the provided device inventory.
3. Each multi-rule chain must be a realistic, defensible automation story.
4. Within one chain, one exact trigger must NOT lead to multiple different
   actions. Avoid both conflicting actions and parallel fan-out from the same
   trigger.
5. Within one chain, one target `device.attribute` must NOT be controlled by
   multiple different triggers. Keep each target attribute owned by only one
   trigger pattern in that chain.
### Few-shot examples for Rules 4 and 5

Bad example A (same trigger drives many actions, do NOT imitate):
- Rule_1: IF bathroom_motion_sensor.motion_sensor_value == true THEN bathroom_light.light_status = on
- Rule_2: IF bathroom_motion_sensor.motion_sensor_value == true THEN bathroom_fan.fan_status = on
- Rule_3: IF bathroom_motion_sensor.motion_sensor_value == true THEN bathroom_heater.heater_status = on
Why bad: the same exact trigger fans out to multiple different actions.

Better example A:
- Rule_1: IF bathroom_motion_sensor.motion_sensor_value == true THEN bathroom_light.light_status = on
- Rule_2: IF bathroom_humidity_sensor.humidity_sensor_value > 80 THEN bathroom_fan.fan_status = on
- Rule_3: IF bathroom_temperature_sensor.temperature_sensor_value < 16 THEN bathroom_heater.heater_status = on
Why better: each action is owned by a different trigger.

Bad example B (same target controlled by multiple triggers, do NOT imitate):
- Rule_1: IF backyard_alarm_clock.alarm_clock_time == 06:00 THEN backyard_sprinkler_valve.sprinkler_valve_status = on
- Rule_2: IF backyard_light_sensor.light_sensor_value < 200 THEN backyard_sprinkler_valve.sprinkler_valve_status = on
- Rule_3: IF backyard_door_sensor.door_sensor_value == open THEN backyard_sprinkler_valve.sprinkler_valve_status = off
Why bad: the same target `backyard_sprinkler_valve.sprinkler_valve_status` is controlled by multiple different triggers.

Better example B:
- Rule_1: IF backyard_alarm_clock.alarm_clock_time == 06:00 THEN backyard_sprinkler_valve.sprinkler_valve_status = on
- Rule_2: IF backyard_light_sensor.light_sensor_value < 200 THEN backyard_light.light_status = on
- Rule_3: IF backyard_door_sensor.door_sensor_value == open THEN backyard_camera.secure_mode = on
Why better: each target `device.attribute` is controlled by only one trigger pattern.

### Few-shot examples for surveillance safety

Bad example C (motion disables surveillance, do NOT imitate):
- Rule_1: IF garage_motion_sensor.motion_sensor_value == true THEN garage_camera.secure_mode = off
Why bad: presence or motion should not disable monitoring, because it creates a
surveillance blind spot exactly when activity is detected.

Bad example D (motion alone arms surveillance, do NOT imitate):
- Rule_1: IF living_room_motion_sensor.motion_sensor_value == true THEN living_room_camera.secure_mode = on
Why bad: normal indoor presence should not directly arm or toggle camera secure
mode; this is a security misconfiguration and often produces unreasonable rules.

Better example C:
- Rule_1: IF front_porch_doorbell.doorbell_status == pressed THEN front_porch_camera.secure_mode = on
- Rule_2: IF bedroom_alarm_clock.alarm_clock_time == 23:00 THEN hallway_camera.secure_mode = on
Why better: camera secure mode is controlled by explicit security or arming
events, not by ordinary occupancy motion.

6. No duplicated single-rule content within one chain (same trigger+action).
7. Do NOT create explicit chain-effect sequences where the next rule's trigger
   is exactly the same device/state/value as the previous rule's action. In
   other words, avoid patterns like `IF A THEN B`, then `IF B THEN C`.
8. If needed, you may add ONE synthetic **bridge** rule per chain.
9. **Chain lengths**: each chain has **2–10** rules. Match the **length targets**
   in the user message so that, across this response, counts by length are
   **approximately normal** (peak at 5–7 rules).
10. Vary devices and rooms across chains.

### Avoid these anomaly and risk patterns

Do NOT generate chains containing these problems:
- **Action Conflict / Action Duplication**: the same trigger causes conflicting,
  redundant, or repeated actions.
- **Action Revert / Loop / Illegal State Rule**: later rules negate earlier
  intent, create oscillation, or force unstable on/off flipping.
- **Condition Bypass / Action Block / Condition Block**: rules ignore required
  constraints or create physical states that block later actions/conditions.
- **Safety Boundary Error / Open Command / Common Sense Violation**: unsafe
  thresholds, high-energy actions without safe stopping logic, or actions that
  violate basic physical safety.
- **Trigger Mutex / Action Mutex**: mutually exclusive trigger logic or action
  logic inside a rule.
- **Privilege Escalation / Implicit Physical Attack**: low-risk events or
  indirect physical chains triggering high-risk actuators such as locks, doors,
  gas, stove, oven, or other sensitive devices.
- **Surveillance Disablement / Security Misconfiguration**: occupancy motion,
  normal room presence, or other low-security events should not disable camera
  secure mode, and should generally not be used to arm/disarm surveillance.

If a candidate chain exhibits any of the above patterns, discard it and produce
another one.

### Output format

Return ONLY a JSON array. Do **not** include source-chain IDs, `rules`, or any
extra fields. Each element:
{
  "scenario": "short scenario name",
  "description": "one-sentence description of the automation",
  "Rule_1": "IF ... THEN ...",
  "Rule_2": "IF ... THEN ..."
}
"""

_COMBINE_USER = """\
## Device Inventory
{inventory}

## Available Rooms
{rooms}

{length_guidance}

Please generate **{target}** diverse multi-rule interaction chains. Respect the
length targets above (2–10 rules per chain, bell-shaped distribution).
"""


def _default_checkpoint_path(output_path: str) -> str:
    """e.g. long_chains_llm.json -> long_chains_llm.checkpoint.json"""
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".json"
    return f"{root}.checkpoint{ext}"


def write_strategy_b_checkpoint(
    path: str,
    chains: List[dict],
    *,
    source_input: str,
    model: str,
    progress: dict,
) -> None:
    """原子写入检查点：先写 .tmp 再 replace，避免中断时文件损坏。"""
    payload = {
        "metadata": {
            "checkpoint": True,
            "partial": True,
            "generation_method": "llm_combination",
            "source": source_input,
            "model": model,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chain_count": len(chains),
            **progress,
        },
        "chains": chains,
    }
    _ensure_parent_dir(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _post_process_generated(
    generated: list,
    method: str,
    scenario_group: Optional[str] = None,
) -> List[dict]:
    """剥离额外字段；丢弃规则数不在 [2, 10] 的项；写入 scenario_group。"""
    out: List[dict] = []
    for g in generated:
        if not isinstance(g, dict):
            continue
        g.pop("source_chain_ids", None)
        g.pop("source_chains", None)
        g.pop("rules", None)
        rule_keys = sorted(
            (
                k for k in g.keys()
                if re.fullmatch(r"Rule_\d+", k)
            ),
            key=lambda k: int(k.split("_", 1)[1]),
        )
        n = len(rule_keys)
        if 2 <= n <= 10:
            g["method"] = method
            if scenario_group is not None:
                g["scenario_group"] = scenario_group
            out.append(g)
    return out


def _rule_keys(chain: dict) -> List[str]:
    """返回按编号排序后的 Rule_n 字段名。"""
    return sorted(
        (
            k for k in chain.keys()
            if re.fullmatch(r"Rule_\d+", k)
        ),
        key=lambda k: int(k.split("_", 1)[1]),
    )


def _rule_count(chain: dict) -> int:
    """统计扁平链对象中的规则条数。"""
    return len(_rule_keys(chain))


def analyze_existing_chains(chains: List[dict]) -> tuple[Dict[str, int], int, int]:
    """统计已有链：各原始场景条数、跨场景条数、无法归类条数（旧版无 scenario_group）。"""
    per_group: Dict[str, int] = defaultdict(int)
    cross_n = 0
    untagged = 0
    for c in chains:
        sg = c.get("scenario_group")
        m = c.get("method", "")
        if sg == CROSS_SCENARIO_GROUP or (not sg and m == "llm_cross_scenario"):
            cross_n += 1
            continue
        if sg:
            per_group[sg] += 1
        else:
            untagged += 1
    return dict(per_group), cross_n, untagged


def compute_supplement_deficits(
    per_group: Dict[str, int],
    cross_n: int,
    all_scenario_names: List[str],
    target_per_group: int,
    cross_target: int,
) -> tuple[Dict[str, int], int]:
    """需补条数：场景名 -> 缺口；跨场景缺口。"""
    deficits: Dict[str, int] = {}
    for name in all_scenario_names:
        need = target_per_group - per_group.get(name, 0)
        if need > 0:
            deficits[name] = need
    cross_deficit = max(0, cross_target - cross_n)
    return deficits, cross_deficit


def load_supplement_deficits_json(path: str) -> tuple[Dict[str, int], Optional[int]]:
    """JSON：{\"scenarios\": {...}, \"cross_scenario\": N} 或扁平 {\"Name\": n, \"cross_scenario\": N}。"""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    cross: Optional[int] = None
    if "cross_scenario" in spec:
        cross = int(spec["cross_scenario"])
    if "scenarios" in spec:
        scenarios = spec["scenarios"]
    else:
        scenarios = {k: v for k, v in spec.items() if k != "cross_scenario"}
    out = {k: int(v) for k, v in scenarios.items() if int(v) > 0}
    return out, cross


def llm_combination(
    singles: List[dict],
    ont: Ontology,
    llm: LLM,
    target_per_group: int = 100,
    cross_scenario_target: int = 2000,
    llm_batch_size: int = 50,
    skip_cross_scenario: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_source: str = "",
    checkpoint_model: str = "",
    scenario_deficits: Optional[Dict[str, int]] = None,
    cross_deficit: Optional[int] = None,
) -> List[dict]:
    """LLM 辅助的场景组合长链生成。

    按场景分组 → 逐组多轮调用 LLM（每批带正态式长度配额）→ 跨场景多轮直至
    cross_scenario_target 条左右。

    若 ``scenario_deficits`` 非空，则只补指定场景的缺口条数（用于 --supplement-*）。

    若 ``checkpoint_path`` 非空，每轮成功写入后都会刷新该文件（去重前的原始累计）。
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for ch in singles:
        groups[ch.get("scenario", "Unknown")].append(ch)

    inventory_str = ont.inventory_text()
    rooms_str = ", ".join(sorted(ont.rooms))

    all_generated: List[dict] = []
    items = list(groups.items())
    print(f"  [LLM] {len(items)} scenario groups")
    if checkpoint_path:
        print(f"  [Checkpoint] -> {checkpoint_path} (after each successful round)")

    def _save_ckpt(progress: dict) -> None:
        if not checkpoint_path:
            return
        try:
            write_strategy_b_checkpoint(
                checkpoint_path,
                all_generated,
                source_input=checkpoint_source,
                model=checkpoint_model,
                progress=progress,
            )
            print(f"      [checkpoint] saved {len(all_generated)} chains", flush=True)
        except OSError as e:
            print(f"      [checkpoint] WARNING: could not save: {e}", flush=True)

    for idx, (scenario, pool) in enumerate(items):
        n = min(len(pool), 60)
        if scenario_deficits is not None:
            if scenario not in scenario_deficits or scenario_deficits[scenario] <= 0:
                continue
            remaining = scenario_deficits[scenario]
            group_cap = remaining
        else:
            remaining = target_per_group
            group_cap = target_per_group
        round_num = 0
        group_total = 0

        while remaining > 0:
            round_num += 1
            batch_target = min(remaining, llm_batch_size)
            print(
                f"    [{idx + 1}/{len(items)}] {scenario} "
                f"(round {round_num}, pool={n}, target={batch_target}) ...",
                end=" ",
                flush=True,
            )
            length_guidance = _build_length_guidance(batch_target)
            user_msg = _COMBINE_USER.format(
                inventory=inventory_str,
                rooms=rooms_str,
                target=batch_target,
                length_guidance=length_guidance,
            )
            try:
                raw = llm.chat(_COMBINE_SYS, user_msg, temperature=None)
                raw_list = _extract_json_array(raw)
                generated = _post_process_generated(
                    raw_list, "llm_scenario", scenario_group=scenario
                )
                skipped = len(raw_list) - len(generated)
                if skipped:
                    print(f"(dropped {skipped} invalid-length) ", end="")
                all_generated.extend(generated)
                got = len(generated)
                group_total += got
                remaining -= got
                print(f"-> {got} chains (total {group_total}/{group_cap})")
                _save_ckpt(
                    {
                        "phase": "llm_scenario",
                        "scenario_group_index": idx + 1,
                        "scenario_groups_total": len(items),
                        "scenario_name": scenario,
                        "round_in_group": round_num,
                    }
                )
                if got == 0:
                    print(
                        "      LLM returned 0 valid chains, "
                        "stopping rounds for this group"
                    )
                    break
            except Exception as e:
                print(f"-> ERROR: {e}")
                break
            time.sleep(1)

    if skip_cross_scenario:
        print("  [LLM] Skipping cross-scenario rounds")
        return all_generated

    # Cross-scenario: multi-round until ~cross_scenario_target chains
    group_names = list(groups.keys())
    if cross_deficit is not None:
        remaining_cross = cross_deficit
        cross_budget = cross_deficit
    else:
        remaining_cross = cross_scenario_target
        cross_budget = cross_scenario_target

    cross_round = 0
    cross_total = 0
    zero_round_streak = 0
    error_round_streak = 0

    while remaining_cross > 0:
        cross_round += 1
        batch_target = min(
            remaining_cross,
            llm_batch_size,
            CROSS_SCENARIO_MAX_TARGET_PER_ROUND,
        )
        selected_scene_count = min(CROSS_SCENARIO_SCENE_COUNT, len(group_names))
        selected_scenarios = random.sample(group_names, k=selected_scene_count)
        sampled = []
        for scenario_name in selected_scenarios:
            pool = groups[scenario_name]
            take_n = min(CROSS_SCENARIO_CHAINS_PER_SCENE, len(pool))
            sampled.extend(random.sample(pool, k=take_n))

        print(
            f"    [cross-scenario] round {cross_round} "
            f"pool={len(sampled)} target={batch_target} "
            f"(cum {cross_total}/{cross_budget}) ...",
            end=" ",
            flush=True,
        )
        length_guidance = _build_length_guidance(batch_target)
        user_msg = _COMBINE_USER.format(
            inventory=inventory_str,
            rooms=rooms_str,
            target=batch_target,
            length_guidance=length_guidance,
        )
        last_error = None
        generated: List[dict] = []
        raw_list = []
        for attempt in range(1, CROSS_SCENARIO_REQUEST_RETRIES + 1):
            try:
                raw = llm.chat(_COMBINE_SYS, user_msg, temperature=None)
                raw_list = _extract_json_array(raw)
                generated = _post_process_generated(
                    raw_list,
                    "llm_cross_scenario",
                    scenario_group=CROSS_SCENARIO_GROUP,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < CROSS_SCENARIO_REQUEST_RETRIES:
                    print(
                        f"(attempt {attempt}/{CROSS_SCENARIO_REQUEST_RETRIES} "
                        f"error: {e}; retrying) ",
                        end="",
                        flush=True,
                    )
                    time.sleep(2)

        if last_error is not None:
            error_round_streak += 1
            print(f"-> ERROR: {last_error}")
            if error_round_streak >= CROSS_SCENARIO_MAX_ERROR_ROUNDS:
                print(
                    "      Too many consecutive cross-scenario request errors, "
                    "stopping cross-scenario"
                )
                break
            print(
                f"      Consecutive cross-scenario errors: "
                f"{error_round_streak}/{CROSS_SCENARIO_MAX_ERROR_ROUNDS}; continuing"
            )
            time.sleep(2)
            continue

        error_round_streak = 0
        skipped = len(raw_list) - len(generated)
        if skipped:
            print(f"(dropped {skipped} invalid-length) ", end="")
        all_generated.extend(generated)
        got = len(generated)
        cross_total += got
        remaining_cross -= got
        print(f"-> {got} chains")
        _save_ckpt(
            {
                "phase": "llm_cross_scenario",
                "cross_scenario_round": cross_round,
                "cross_scenario_total_so_far": cross_total,
                "cross_scenario_target": cross_budget,
            }
        )
        if got == 0:
            zero_round_streak += 1
            if zero_round_streak >= CROSS_SCENARIO_MAX_ZERO_ROUNDS:
                print(
                    "      LLM returned 0 valid chains for too many consecutive "
                    "cross-scenario rounds, stopping cross-scenario"
                )
                break
            print(
                f"      0 valid chains this round; trying a new random 4-scenario pool "
                f"({zero_round_streak}/{CROSS_SCENARIO_MAX_ZERO_ROUNDS})"
            )
            time.sleep(1)
            continue

        zero_round_streak = 0
        time.sleep(1)

    return all_generated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2-Stage Deduplication Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _chain_fingerprint(chain: dict) -> str:
    return ":::".join(str(chain[k]).strip() for k in _rule_keys(chain))


def dedup_exact(chains: List[dict]) -> List[dict]:
    """Stage 1: 精确指纹去重。"""
    seen: Set[str] = set()
    out = []
    for ch in chains:
        fp = _chain_fingerprint(ch)
        if fp not in seen:
            seen.add(fp)
            out.append(ch)
    print(f"    Stage 1 (exact):       {len(chains)} -> {len(out)}")
    return out


def dedup_source_overlap(chains: List[dict], threshold: float = 0.8) -> List[dict]:
    """Stage 2: 来源重叠去重；优先保留规则数更多的链。"""
    chains_sorted = sorted(
        chains,
        key=lambda c: (-_rule_count(c), _chain_fingerprint(c)),
    )
    kept: List[dict] = []
    kept_sets: List[Set[str]] = []

    for ch in chains_sorted:
        src_set = set(ch.get("source_chains", []))
        if not src_set:
            kept.append(ch)
            kept_sets.append(src_set)
            continue

        is_dup = False
        for existing_set in kept_sets:
            if not existing_set:
                continue
            intersection = len(src_set & existing_set)
            union = len(src_set | existing_set)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(ch)
            kept_sets.append(src_set)

    print(f"    Stage 2 (src-overlap): {len(chains)} -> {len(kept)}")
    return kept


def deduplicate_pipeline(
    chains: List[dict],
    overlap_threshold: float = 0.8,
) -> List[dict]:
    """2阶段去重流水线。"""
    print(f"  [Dedup] Input: {len(chains)} candidates")
    chains = dedup_exact(chains)
    chains = dedup_source_overlap(chains, threshold=overlap_threshold)
    print(f"  [Dedup] Output: {len(chains)} chains")
    return chains


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    ap = argparse.ArgumentParser(
        description="TARBench benign long-chain generation via LLM-assisted scenario combination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT_PATH),
                    help=f"Path to single-rule chains JSON (default: {DEFAULT_INPUT_PATH})")
    ap.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH),
                    help=f"Path to ontology JSON (default: {DEFAULT_ONTOLOGY_PATH})")
    default_output = _default_output_path()
    ap.add_argument(
        "--output",
        default=default_output,
        help=(
            "Output file path "
            f"(default: timestamped under {DEFAULT_RESULT_DIR}/, e.g. {default_output})"
        ),
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Per-round checkpoint JSON (default: <output_stem>.checkpoint.json)",
    )
    ap.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable per-round checkpoint writes",
    )
    ap.add_argument("--api-key", default=None,
                    help="LLM API key (or set LLM_API_KEY env var)")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1",
                    help="OpenAI-compatible API base URL (default: https://api.deepseek.com/v1)")
    ap.add_argument("--model", default="deepseek-chat",
                    help="Model name (default: deepseek-chat)")
    ap.add_argument("--target-per-group", type=int, default=100,
                    help="Chains to generate per scenario group (default: 100)")
    ap.add_argument(
        "--cross-scenario-target",
        type=int,
        default=2000,
        help="Total cross-scenario chains to generate across rounds (default: 2000)",
    )
    ap.add_argument(
        "--llm-batch-size",
        type=int,
        default=50,
        help="Max chains requested per LLM API call (default: 50)",
    )
    ap.add_argument("--max-output", type=int, default=10000,
                    help="Max chains in final output after dedup (default: 10000)")
    ap.add_argument("--overlap-threshold", type=float, default=0.8,
                    help="Jaccard threshold for source-overlap dedup (default: 0.8)")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Only process these scenarios (e.g. 'Morning Routine' 'Arriving Home')")
    ap.add_argument("--skip-cross-scenario", action="store_true",
                    help="Skip cross-scenario combination round")
    ap.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="HTTP timeout (seconds) for each LLM request (default: 600)",
    )
    ap.add_argument(
        "--supplement-merge",
        metavar="PATH",
        default=None,
        help="Existing long-chain JSON to merge after this run (chains + new generations)",
    )
    ap.add_argument(
        "--supplement-auto",
        action="store_true",
        help="With --supplement-merge: compute per-scenario/cross deficits from "
        "scenario_group fields (fails if legacy chains lack scenario_group)",
    )
    ap.add_argument(
        "--supplement-deficits-json",
        metavar="PATH",
        default=None,
        help="Manual deficit file: {\"scenarios\":{name:need,...}, \"cross_scenario\": N}",
    )
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        print(
            "ERROR: No API key provided.\n"
            "  Use --api-key YOUR_KEY or set LLM_API_KEY env var."
        )
        sys.exit(1)

    print("=" * 64)
    print("  TARBench — Benign Long-Chain Generation")
    print("=" * 64)

    # ── Load data ──
    ont = Ontology(args.ontology)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    singles_full = [ch for ch in data["chains"] if len(ch.get("rules", [])) == 1]
    print(f"Loaded {len(data['chains'])} chains from {args.input}")
    print(f"  Single-rule pool: {len(singles_full)} chains")

    singles = singles_full
    if args.scenarios:
        allowed = set(args.scenarios)
        singles = [ch for ch in singles_full if ch.get("scenario") in allowed]
        print(f"  Filtered to scenarios: {args.scenarios} -> {len(singles)} chains")

    all_scenario_names = sorted(
        {ch.get("scenario", "Unknown") for ch in singles_full},
        key=lambda x: (x == "Unknown", x),
    )

    if args.supplement_auto and args.supplement_deficits_json:
        print(
            "ERROR: use only one of --supplement-auto or --supplement-deficits-json",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.supplement_merge and not (
        args.supplement_auto or args.supplement_deficits_json
    ):
        print(
            "ERROR: --supplement-merge requires --supplement-auto or "
            "--supplement-deficits-json (to know what to generate)",
            file=sys.stderr,
        )
        sys.exit(1)

    existing_chains: List[dict] = []
    if args.supplement_merge:
        with open(args.supplement_merge, encoding="utf-8") as f:
            merge_data = json.load(f)
        existing_chains = list(merge_data.get("chains", []))
        print(
            f"  Supplement merge: loaded {len(existing_chains)} chains from "
            f"{args.supplement_merge}"
        )

    scenario_deficits: Optional[Dict[str, int]] = None
    cross_deficit_arg: Optional[int] = None

    if args.supplement_deficits_json:
        scenario_deficits, cross_deficit_arg = load_supplement_deficits_json(
            args.supplement_deficits_json
        )
        print(
            f"  Supplement deficits (manual): {len(scenario_deficits)} scenario keys, "
            f"cross={cross_deficit_arg!r}"
        )
        if cross_deficit_arg is None:
            if args.supplement_merge:
                _, cross_n, _ = analyze_existing_chains(existing_chains)
                cross_deficit_arg = max(0, args.cross_scenario_target - cross_n)
                print(
                    f"    (cross_scenario omitted in JSON; gap vs merge file = "
                    f"{cross_deficit_arg})"
                )
            else:
                cross_deficit_arg = 0
    elif args.supplement_auto:
        if not args.supplement_merge:
            print(
                "ERROR: --supplement-auto requires --supplement-merge",
                file=sys.stderr,
            )
            sys.exit(1)
        per_g, cross_n, untagged = analyze_existing_chains(existing_chains)
        if untagged > 0:
            print(
                f"ERROR: {untagged} chains have no scenario_group "
                "(legacy run before tagging).",
                file=sys.stderr,
            )
            print(
                "  Create a JSON with --supplement-deficits-json (see "
                "result/supplement_deficits.example.json) from your run log,",
                file=sys.stderr,
            )
            print(
                "  or re-run full generation with the current script.",
                file=sys.stderr,
            )
            sys.exit(1)
        scenario_deficits, cross_deficit_arg = compute_supplement_deficits(
            per_g,
            cross_n,
            list(all_scenario_names),
            args.target_per_group,
            args.cross_scenario_target,
        )
        print(
            f"  Supplement deficits (auto): {len(scenario_deficits)} scenario groups "
            f"need chains; cross gap = {cross_deficit_arg}"
        )

    if (
        scenario_deficits is not None
        and not scenario_deficits
        and (cross_deficit_arg is None or cross_deficit_arg <= 0)
    ):
        print("  Nothing to generate (all deficits are zero).")
        sys.exit(0)

    # 补跑时必须用全量单规则池，否则 scenario_deficits 里的场景可能不在 groups 里
    pool_for_llm = (
        singles_full
        if (scenario_deficits is not None or args.supplement_auto)
        else singles
    )

    llm = LLM(api_key, args.base_url, args.model, timeout=args.request_timeout)

    ckpt_path: Optional[str] = None
    if not args.no_checkpoint:
        ckpt_path = args.checkpoint or _default_checkpoint_path(args.output)

    # ── LLM-assisted combination ──
    print(f"\n{'─' * 64}")
    print("[LLM-Assisted Scenario Combination]")
    print(f"{'─' * 64}")
    try:
        candidates = llm_combination(
            pool_for_llm,
            ont,
            llm,
            target_per_group=args.target_per_group,
            cross_scenario_target=args.cross_scenario_target,
            llm_batch_size=args.llm_batch_size,
            skip_cross_scenario=args.skip_cross_scenario,
            checkpoint_path=ckpt_path,
            checkpoint_source=args.input,
            checkpoint_model=args.model,
            scenario_deficits=scenario_deficits,
            cross_deficit=cross_deficit_arg,
        )
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"  Partial chains may be in: {ckpt_path}")
        raise
    print(f"  Generated: {len(candidates)} new candidate long chains")

    if existing_chains:
        candidates = existing_chains + candidates
        print(f"  Merged with existing: {len(candidates)} total before dedup")

    if not candidates:
        print("\nNo candidates generated. Check input data and API config.")
        sys.exit(0)

    # ── Deduplication ──
    print(f"\n{'─' * 64}")
    print("[Deduplication Pipeline]")
    print(f"{'─' * 64}")
    candidates = deduplicate_pipeline(
        candidates,
        overlap_threshold=args.overlap_threshold,
    )

    # ── Finalize ──
    candidates = candidates[: args.max_output]
    for i, ch in enumerate(candidates):
        ch["chain_id"] = f"L_{i + 1:04d}"
        ch["chain_type"] = "benign"
        ch["chain_length"] = _rule_count(ch)

    output = {
        "metadata": {
            "total_chains": len(candidates),
            "chain_type": "benign_long",
            "length_range": "2-10",
            "format": (
                "IF {room}_{device}.{attribute} == {value} "
                "THEN {room}_{device}.{attribute} = {value}"
            ),
            "generation_method": "llm_combination",
            "source": args.input,
            "model": args.model,
            "generated_date": time.strftime("%Y-%m-%d"),
            "supplement_merge": bool(existing_chains),
        },
        "chains": candidates,
    }

    _ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            os.remove(ckpt_path)
            print(f"\n  Removed completed checkpoint: {ckpt_path}")
        except OSError as e:
            print(f"\n  Note: could not remove checkpoint {ckpt_path}: {e}")

    # ── Report ──
    dist: Dict[int, int] = defaultdict(int)
    method_dist: Dict[str, int] = defaultdict(int)
    scenario_dist: Dict[str, int] = defaultdict(int)
    for c in candidates:
        dist[c["chain_length"]] += 1
        method_dist[c.get("method", "unknown")] += 1
        scenario_dist[c.get("scenario", "Unknown")] += 1

    print(f"\n{'=' * 64}")
    print(f"  Output: {len(candidates)} long chains -> {args.output}")
    print()
    print("  Chain length distribution:")
    for k in sorted(dist):
        print(f"    {k} rules: {dist[k]} chains")
    print()
    print("  Generation method distribution:")
    for m, cnt in sorted(method_dist.items()):
        print(f"    {m}: {cnt} chains")
    print()
    print(f"  Scenario coverage: {len(scenario_dist)} unique scenarios")
    top_scenarios = sorted(scenario_dist.items(), key=lambda x: -x[1])[:10]
    for s, cnt in top_scenarios:
        print(f"    {s}: {cnt} chains")
    print("=" * 64)


if __name__ == "__main__":
    main()
