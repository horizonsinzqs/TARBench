import argparse
import json
import random
from pathlib import Path


ROOMS = [
    "living_room",
    "bedroom",
    "bathroom",
    "kitchen",
    "study_room",
    "dining_room",
    "backyard",
    "garage",
    "balcony",
    "hallway",
    "front_porch",
    "laundry_room",
]

SCENARIO_GROUPS = [
    "Air Quality Management",
    "Arriving Home",
    "Bathroom Automation",
    "Child/Elder Safety",
    "Cleaning & Housekeeping",
    "Climate Comfort",
    "Cooking & Kitchen",
    "Energy Saving",
    "Entertainment & Leisure",
    "Garage & Vehicle",
    "Garden & Outdoor",
    "Guest Mode",
    "Laundry Management",
    "Leaving Home",
    "Morning Routine",
    "Night/Sleep Mode",
    "Nighttime Navigation",
    "Package & Delivery",
    "Safety Response",
    "Seasonal Adjustment",
    "Security & Surveillance",
    "Vacation / Extended Away",
    "Weather Response",
]

DEVICE_ATTRIBUTES = {
    "light": ["light_status"],
    "lamp": ["lamp_status", "lamp_brightness"],
    "heater": ["heater_status", "heater_mode"],
    "AC": ["AC_status", "AC_mode", "AC_intensity"],
    "dehumidifier": ["dehumidifier_status", "dehumidifier_mode", "dehumidifier_container_full"],
    "humidifier": ["humidifier_status", "humidifier_mode", "humidifier_container_empty"],
    "smart_lock": ["smart_lock_status"],
    "camera": ["secure_mode", "motion_detected"],
    "smoke_sensor": ["smoke_sensor_value"],
    "alarm_clock": ["alarm_clock_time", "alarm_clock_status"],
    "water_leak_sensor": ["water_leak_sensor_value"],
    "temperature_sensor": ["temperature_sensor_value"],
    "motion_sensor": ["motion_sensor_value"],
    "window": ["window_status"],
    "curtain": ["curtain_status"],
    "air_purifier": ["air_purifier_status", "air_purifier_mode"],
    "gas_sensor": ["gas_sensor_valve"],
    "sprinkler_valve": ["sprinkler_valve_status"],
    "fan": ["fan_status", "fan_mode", "fan_intensity"],
    "oven": ["oven_status", "oven_mode"],
    "plug": ["plug_status"],
    "door_sensor": ["door_sensor_value"],
    "doorbell": ["doorbell_status"],
    "thermostat": ["thermostat_target_temp", "thermostat_status"],
    "robot_vacuum": ["robot_vacuum_status"],
    "speaker": ["speaker_status", "speaker_volume"],
    "garage_door": ["garage_door_status"],
    "cooler": ["cooler_status", "cooler_mode"],
    "siren": ["siren_status"],
    "window_lock": ["window_lock_status"],
    "washing_machine": ["washing_machine_status", "washing_machine_mode"],
    "dishwasher": ["dishwasher_status"],
    "water_valve": ["water_valve_status"],
    "stove": ["stove_status"],
    "coffee_machine": ["coffee_machine_status"],
    "toaster": ["toaster_status"],
    "light_sensor": ["light_sensor_value"],
    "humidity_sensor": ["humidity_sensor_value"],
    "air_quality_sensor": ["air_quality_sensor_value"],
}

ATTRIBUTE_VALUES = {
    "light_status": ["on", "off"],
    "lamp_status": ["on", "off"],
    "lamp_brightness": ["strong", "medium", "weak"],
    "heater_status": ["on", "off"],
    "heater_mode": ["auto", "high", "low", "eco"],
    "AC_status": ["on", "off"],
    "AC_mode": ["cool", "heat", "auto", "fan_only"],
    "AC_intensity": ["low", "medium", "high"],
    "dehumidifier_status": ["on", "off"],
    "dehumidifier_mode": ["strong", "medium", "weak"],
    "dehumidifier_container_full": ["true", "false"],
    "humidifier_status": ["on", "off"],
    "humidifier_mode": ["strong", "medium", "weak"],
    "humidifier_container_empty": ["true", "false"],
    "smart_lock_status": ["locked", "unlocked"],
    "secure_mode": ["on", "off"],
    "motion_detected": ["true", "false"],
    "smoke_sensor_value": ["true", "false"],
    "alarm_clock_status": ["on", "off", "ringing"],
    "water_leak_sensor_value": ["true", "false"],
    "motion_sensor_value": ["true", "false"],
    "window_status": ["open", "closed"],
    "curtain_status": ["open", "closed"],
    "air_purifier_status": ["on", "off"],
    "air_purifier_mode": ["auto", "sleep", "high", "low"],
    "gas_sensor_valve": ["true", "false"],
    "sprinkler_valve_status": ["on", "off"],
    "fan_status": ["on", "off"],
    "fan_mode": ["static", "shaking"],
    "fan_intensity": ["low", "medium", "high"],
    "oven_status": ["on", "off"],
    "oven_mode": ["bake", "grill", "convection"],
    "plug_status": ["on", "off"],
    "door_sensor_value": ["open", "closed"],
    "doorbell_status": ["pressed", "idle"],
    "thermostat_status": ["on", "off"],
    "robot_vacuum_status": ["on", "off", "charging", "self-cleaning"],
    "speaker_status": ["on", "off"],
    "speaker_volume": ["low", "medium", "high"],
    "garage_door_status": ["open", "closed"],
    "cooler_status": ["on", "off"],
    "cooler_mode": ["auto", "high", "low"],
    "siren_status": ["on", "off"],
    "window_lock_status": ["locked", "unlocked"],
    "washing_machine_status": ["on", "off"],
    "washing_machine_mode": ["normal", "quick_wash", "heavy_duty", "spin"],
    "dishwasher_status": ["on", "off"],
    "water_valve_status": ["on", "off"],
    "stove_status": ["on", "off"],
    "coffee_machine_status": ["on", "off"],
    "toaster_status": ["on", "off"],
}

SENSOR_DEVICES = {
    "smoke_sensor",
    "alarm_clock",
    "water_leak_sensor",
    "temperature_sensor",
    "motion_sensor",
    "door_sensor",
    "doorbell",
    "light_sensor",
    "humidity_sensor",
    "air_quality_sensor",
    "gas_sensor",
}

ACTION_FRIENDLY_DEVICES = [
    "light",
    "lamp",
    "heater",
    "AC",
    "dehumidifier",
    "humidifier",
    "smart_lock",
    "camera",
    "window",
    "curtain",
    "air_purifier",
    "sprinkler_valve",
    "fan",
    "oven",
    "plug",
    "thermostat",
    "robot_vacuum",
    "speaker",
    "garage_door",
    "cooler",
    "siren",
    "window_lock",
    "washing_machine",
    "dishwasher",
    "water_valve",
    "stove",
    "coffee_machine",
    "toaster",
]

ROOM_DEVICE_POOLS = {
    "living_room": ["light", "lamp", "heater", "AC", "dehumidifier", "humidifier", "camera", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "plug", "speaker", "robot_vacuum", "window_lock", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "bedroom": ["light", "lamp", "heater", "AC", "dehumidifier", "humidifier", "alarm_clock", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "plug", "speaker", "window_lock", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "bathroom": ["light", "lamp", "heater", "dehumidifier", "humidifier", "water_leak_sensor", "temperature_sensor", "motion_sensor", "window", "fan", "door_sensor", "water_valve", "light_sensor", "humidity_sensor", "speaker"],
    "kitchen": ["light", "lamp", "heater", "AC", "smoke_sensor", "gas_sensor", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "oven", "plug", "door_sensor", "speaker", "dishwasher", "water_valve", "stove", "coffee_machine", "toaster", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "study_room": ["light", "lamp", "heater", "AC", "dehumidifier", "humidifier", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "plug", "speaker", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "dining_room": ["light", "lamp", "heater", "AC", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "speaker", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "backyard": ["light", "motion_sensor", "sprinkler_valve", "camera", "speaker", "siren", "light_sensor", "humidity_sensor", "air_quality_sensor"],
    "garage": ["light", "camera", "temperature_sensor", "motion_sensor", "plug", "speaker", "garage_door", "robot_vacuum", "light_sensor", "door_sensor"],
    "balcony": ["light", "lamp", "temperature_sensor", "motion_sensor", "window", "curtain", "air_purifier", "fan", "plug", "light_sensor", "humidity_sensor", "air_quality_sensor", "window_lock"],
    "hallway": ["light", "lamp", "motion_sensor", "door_sensor", "camera", "air_purifier", "speaker", "light_sensor"],
    "front_porch": ["light", "camera", "motion_sensor", "door_sensor", "doorbell", "smart_lock", "speaker", "siren", "light_sensor"],
    "laundry_room": ["light", "lamp", "dehumidifier", "humidifier", "water_leak_sensor", "temperature_sensor", "motion_sensor", "fan", "door_sensor", "plug", "washing_machine", "water_valve", "light_sensor", "humidity_sensor", "air_quality_sensor"],
}

ROOM_SCENARIO_GROUPS = {
    "living_room": ["Entertainment & Leisure", "Climate Comfort", "Air Quality Management", "Arriving Home"],
    "bedroom": ["Morning Routine", "Night/Sleep Mode", "Climate Comfort", "Seasonal Adjustment"],
    "bathroom": ["Bathroom Automation", "Climate Comfort", "Safety Response", "Nighttime Navigation"],
    "kitchen": ["Cooking & Kitchen", "Morning Routine", "Safety Response", "Air Quality Management"],
    "study_room": ["Energy Saving", "Climate Comfort", "Entertainment & Leisure", "Night/Sleep Mode"],
    "dining_room": ["Entertainment & Leisure", "Arriving Home", "Climate Comfort", "Guest Mode"],
    "backyard": ["Garden & Outdoor", "Weather Response", "Security & Surveillance", "Seasonal Adjustment"],
    "garage": ["Garage & Vehicle", "Security & Surveillance", "Leaving Home", "Arriving Home"],
    "balcony": ["Weather Response", "Air Quality Management", "Seasonal Adjustment", "Energy Saving"],
    "hallway": ["Nighttime Navigation", "Security & Surveillance", "Arriving Home", "Leaving Home"],
    "front_porch": ["Package & Delivery", "Security & Surveillance", "Child/Elder Safety", "Arriving Home"],
    "laundry_room": ["Laundry Management", "Safety Response", "Climate Comfort", "Energy Saving"],
}


def random_time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]):02d}"


def random_value(attr: str, rng: random.Random) -> str:
    if attr == "alarm_clock_time":
        return random_time(rng)
    if attr == "temperature_sensor_value":
        return str(rng.randint(16, 31))
    if attr == "thermostat_target_temp":
        return str(rng.randint(18, 28))
    if attr == "light_sensor_value":
        return str(rng.randint(80, 650))
    if attr == "humidity_sensor_value":
        return str(rng.randint(30, 75))
    if attr == "air_quality_sensor_value":
        return str(rng.randint(60, 180))
    return rng.choice(ATTRIBUTE_VALUES[attr])


def comparison_for(attr: str, value: str, rng: random.Random) -> str:
    if attr == "temperature_sensor_value":
        return rng.choice([">", "<", "=="])
    if attr in {"light_sensor_value", "humidity_sensor_value", "air_quality_sensor_value"}:
        return rng.choice([">", "<"])
    return "=="


def make_entity(room: str, device: str) -> str:
    return f"{room}_{device}"


def choose_trigger(rng: random.Random, room: str):
    room_devices = set(ROOM_DEVICE_POOLS[room])
    candidates = sorted(room_devices & (SENSOR_DEVICES | {"window", "curtain", "smart_lock", "garage_door"}))
    trigger_device = rng.choice(candidates)
    attr = rng.choice(DEVICE_ATTRIBUTES[trigger_device])
    value = random_value(attr, rng)
    op = comparison_for(attr, value, rng)
    entity = make_entity(room, trigger_device)
    return f"{entity}.{attr} {op} {value}"


def choose_action(rng: random.Random, room: str, used_targets: set[str]) -> str:
    room_action_devices = [device for device in ROOM_DEVICE_POOLS[room] if device in ACTION_FRIENDLY_DEVICES]
    for _ in range(100):
        device = rng.choice(room_action_devices)
        attr = rng.choice([a for a in DEVICE_ATTRIBUTES[device] if "container_" not in a and a != "motion_detected"])
        entity = make_entity(room, device)
        target = f"{entity}.{attr}"
        if target in used_targets:
            continue
        used_targets.add(target)
        value = random_value(attr, rng)
        return f"{entity}.{attr} = {value}"
    raise RuntimeError("Unable to choose a unique action target.")


def build_rule(rng: random.Random, room: str, compressed: bool) -> str:
    trigger = choose_trigger(rng, room)
    action_count = rng.randint(2, 4) if compressed else 1
    trigger_target = trigger.split(" ", 1)[0]
    local_targets: set[str] = {trigger_target}
    actions = [choose_action(rng, room, local_targets) for _ in range(action_count)]
    return f"IF {trigger} THEN {', '.join(actions)}"


def scenario_title(room: str, group: str, rng: random.Random) -> str:
    room_label = room.replace("_", " ")
    lead = rng.choice(
        [
            "steady",
            "calm",
            "early",
            "late",
            "balanced",
            "quiet",
            "cozy",
            "fresh",
            "routine",
            "gentle",
        ]
    )
    noun = rng.choice(
        [
            "comfort setup",
            "automation block",
            "response routine",
            "control sequence",
            "room adjustment",
            "home preparation",
            "daily routine",
            "environment tune-up",
        ]
    )
    return f"{room_label} {lead} {noun}"


def description_for(room: str, visible_rule_count: int, compressed_count: int) -> str:
    room_label = room.replace("_", " ")
    return (
        f"A benign {room_label} automation item with {visible_rule_count} visible rules, "
        f"including {compressed_count} compressed rule"
        f"{'' if compressed_count == 1 else 's'} for token-compression evaluation."
    )


def load_dataset(path: Path) -> dict:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "metadata": {
            "dataset": "token_compression_benign",
            "label": "benign",
            "sample_count": 0,
            "target_sample_count": 500,
            "batch_size": 50,
            "sampling_seed": 20260516,
            "method": "token_compression_manual",
        },
        "samples": [],
    }


def generate_sample(sample_index: int, rng: random.Random) -> dict:
    room = rng.choice(ROOMS)
    scenario_group = rng.choice(ROOM_SCENARIO_GROUPS[room])
    visible_rule_count = rng.randint(2, 10)
    compressed_count = rng.randint(1, min(3, visible_rule_count))
    compressed_positions = set(rng.sample(range(visible_rule_count), compressed_count))
    rules = {}

    for idx in range(visible_rule_count):
        compressed = idx in compressed_positions
        rules[f"Rule_{idx + 1}"] = build_rule(rng, room, compressed)

    sample = {
        "sample_index": sample_index,
        "sample_id": f"TC_BENIGN_{sample_index:06d}",
        "chain_id": f"TCB_{sample_index:06d}",
        "chain_type": "benign",
        "label": "benign",
        "scenario": scenario_title(room, scenario_group, rng),
        "scenario_group": scenario_group,
        "method": "token_compression_manual",
        "chain_length": visible_rule_count,
        "rules": rules,
        "description": description_for(room, visible_rule_count, compressed_count),
        "ground_truth": {
            "has_anomaly": False,
            "involved_rules": [],
            "anomaly_type": None,
        },
        "mixed_index": sample_index,
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description="Generate benign token-compression dataset batches.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    dataset = load_dataset(args.output)
    existing = dataset.get("samples", [])
    start_index = len(existing) + 1
    rng = random.Random(args.seed + start_index)

    for sample_index in range(start_index, start_index + args.count):
        existing.append(generate_sample(sample_index, rng))

    dataset["samples"] = existing
    dataset["metadata"]["sample_count"] = len(existing)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {args.count} new samples. Total samples: {len(existing)}")


if __name__ == "__main__":
    main()
