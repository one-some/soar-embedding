import orjson
import csv
import re
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

LABELED_DIR = Path("labeled_v2")
OUT_DIR = Path("processed_v2")
OUT_CSV = OUT_DIR / "logs.csv"

DIRB_PER_SCENARIO = 10_000
BENIGN_RATIO = 10.0

MIN_CONFIDENCE_MALICIOUS = 0.70
MIN_CONFIDENCE_SUSPICIOUS = 0.50

RANDOM_SEED = 42

TEST_SCENARIOS = {"harrison", "santos"}
TRAIN_SCENARIOS = {"fox", "wheeler", "wilson", "shaw", "wardbeck", "russellmitchell"}


def build_content(record: dict) -> str:
    meta = record.get("metadata", {})
    entities = record.get("entities", {})
    parts = []

    full_log = (meta.get("full_log") or "").strip()
    if full_log:
        parts.append(full_log)

    sig = (meta.get("signature") or "").strip()
    if sig:
        parts.append(f"sig:{sig}")

    cat = (meta.get("category") or "").strip()
    if cat:
        parts.append(f"cat:{cat}")

    groups = meta.get("rule_groups") or []
    if groups:
        parts.append(f"groups:{','.join(groups)}")

    src_ip = entities.get("src_ip")
    dst_ip = entities.get("dst_ip")
    src_port = entities.get("src_port")
    dst_port = entities.get("dst_port")
    if src_ip or dst_ip:
        parts.append(
            f"flow:{src_ip or '?'}:{src_port or '?'}"
            f"->{dst_ip or '?'}:{dst_port or '?'}"
        )

    domains = entities.get("domains") or []
    if domains:
        parts.append(f"domains:{','.join(domains[:5])}")

    urls = entities.get("urls") or []
    if urls:
        parts.append(f"urls:{','.join(urls[:3])}")

    return " | ".join(parts)


def extract_url_path_bucket(content: str) -> str:
    match = re.search(r'"(?:GET|POST|PUT|DELETE|HEAD|OPTIONS)\s+(/[^\s"]*)', content)
    if not match:
        return "_no_url"
    path = match.group(1)
    parts = path.strip("/").split("/")
    if len(parts) <= 1:
        return "/" + parts[0] if parts else "/"
    return "/" + "/".join(parts[:2])


OUT_DIR.mkdir(exist_ok=True)
random.seed(RANDOM_SEED)

# pools[scenario] = {"attack": [(ts, content, phase, label, conf), ...], ...}
pools = defaultdict(lambda: {"attack": [], "in_window": [], "normal": []})
drop_reasons = Counter()

for j_path in sorted(LABELED_DIR.glob("*_comprehensive.json")):
    scenario = j_path.stem.replace("_comprehensive", "")
    print(f"{j_path.name}...", end=" ", flush=True)

    with j_path.open("rb") as f:
        data = orjson.loads(f.read())

    n_attack = n_inwindow = n_normal = n_dropped = 0

    for record in data:
        gt = record.get("ground_truth", {})
        label = gt.get("label", "UNKNOWN")
        confidence = gt.get("confidence", 0.0)
        phase = gt.get("attack_phase")

        content = build_content(record)
        if not content:
            drop_reasons["empty_content"] += 1
            n_dropped += 1
            continue

        try:
            ts = int(datetime.fromisoformat(record["timestamp"]).timestamp())
        except Exception:
            drop_reasons["bad_timestamp"] += 1
            n_dropped += 1
            continue

        if label in ("MALICIOUS", "SUSPICIOUS") and phase:
            if label == "SUSPICIOUS" and confidence < MIN_CONFIDENCE_SUSPICIOUS:
                drop_reasons["low_confidence_suspicious"] += 1
                n_dropped += 1
                continue
            if label == "MALICIOUS" and confidence < MIN_CONFIDENCE_MALICIOUS:
                drop_reasons["low_confidence_malicious"] += 1
                n_dropped += 1
                continue

            pools[scenario]["attack"].append((ts, content, phase, label, confidence))
            n_attack += 1

        elif phase:
            # In an attack window but labeled BENIGN/PRE/POST. No confidence
            # filter - inclusion is justified by temporal ground truth.
            pools[scenario]["in_window"].append((ts, content, phase, label, confidence))
            n_inwindow += 1

        elif label == "BENIGN":
            pools[scenario]["normal"].append((ts, content, "none", label, confidence))
            n_normal += 1

        else:
            drop_reasons["no_phase_non_benign"] += 1
            n_dropped += 1

    split = "TEST" if scenario in TEST_SCENARIOS else "TRAIN"
    print(
        f"{len(data):,} records -> attack={n_attack:,} in_window={n_inwindow:,} "
        f"normal={n_normal:,} dropped={n_dropped:,} [{split}]"
    )


sampled_attack = []
attack_stats = defaultdict(lambda: defaultdict(lambda: {"available": 0, "sampled": 0}))

for scenario, pool in pools.items():
    by_phase = defaultdict(list)
    for record in pool["attack"]:
        by_phase[record[2]].append(record)

    for phase, records in by_phase.items():
        available = len(records)
        attack_stats[scenario][phase]["available"] = available

        if phase == "dirb" and available > DIRB_PER_SCENARIO:
            # Stratified sample by URL path bucket so we don't oversample one path.
            by_bucket = defaultdict(list)
            for r in records:
                bucket = extract_url_path_bucket(r[1])
                by_bucket[bucket].append(r)

            per_bucket = max(1, DIRB_PER_SCENARIO // len(by_bucket))
            sampled = []
            for bucket_records in by_bucket.values():
                random.shuffle(bucket_records)
                sampled.extend(bucket_records[:per_bucket])

            if len(sampled) < DIRB_PER_SCENARIO:
                random.shuffle(records)
                sampled = records[:DIRB_PER_SCENARIO]

            attack_stats[scenario][phase]["sampled"] = len(sampled)
            sampled_attack.extend([(scenario, *r) for r in sampled])
        else:
            attack_stats[scenario][phase]["sampled"] = available
            sampled_attack.extend([(scenario, *r) for r in records])

sampled_in_window = []
for scenario, pool in pools.items():
    sampled_in_window.extend([(scenario, *r) for r in pool["in_window"]])

total_attack_like = len(sampled_attack) + len(sampled_in_window)

benign_cap = int(total_attack_like * BENIGN_RATIO)
all_normal = []
for scenario, pool in pools.items():
    all_normal.extend([(scenario, *r) for r in pool["normal"]])

random.shuffle(all_normal)
sampled_normal = all_normal[:benign_cap]


all_rows = []

for scenario, ts, content, phase, orig_label, conf in sampled_attack:
    all_rows.append(
        {
            "Scenario": scenario,
            "Tier": "ATTACK",
            "Label": "Anomaly",
            "Phase": phase,
            "Content": content,
            "Timestamp": ts,
            "Confidence": round(conf, 3),
            "OrigLabel": orig_label,
            "Split": "test" if scenario in TEST_SCENARIOS else "train",
        }
    )

for scenario, ts, content, phase, orig_label, conf in sampled_in_window:
    all_rows.append(
        {
            "Scenario": scenario,
            "Tier": "IN_WINDOW",
            "Label": "Anomaly",
            "Phase": phase,
            "Content": content,
            "Timestamp": ts,
            "Confidence": round(conf, 3),
            "OrigLabel": orig_label,
            "Split": "test" if scenario in TEST_SCENARIOS else "train",
        }
    )

for scenario, ts, content, phase, orig_label, conf in sampled_normal:
    all_rows.append(
        {
            "Scenario": scenario,
            "Tier": "NORMAL",
            "Label": "Normal",
            "Phase": "none",
            "Content": content,
            "Timestamp": ts,
            "Confidence": round(conf, 3),
            "OrigLabel": orig_label,
            "Split": "test" if scenario in TEST_SCENARIOS else "train",
        }
    )

all_rows.sort(key=lambda r: (r["Scenario"], r["Timestamp"]))

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Scenario",
            "Tier",
            "Label",
            "Phase",
            "Content",
            "Timestamp",
            "Confidence",
            "OrigLabel",
            "Split",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)


n_attack = len(sampled_attack)
n_inwindow = len(sampled_in_window)
n_normal = len(sampled_normal)
total = len(all_rows)

print(
    f"\nwrote {total:,} rows: ATTACK={n_attack:,} IN_WINDOW={n_inwindow:,} NORMAL={n_normal:,}"
)
print(f"anomaly: {n_attack + n_inwindow:,} ({100*(n_attack+n_inwindow)/total:.1f}%)")

train_rows = [r for r in all_rows if r["Split"] == "train"]
test_rows = [r for r in all_rows if r["Split"] == "test"]
train_anomaly = sum(1 for r in train_rows if r["Label"] == "Anomaly")
test_anomaly = sum(1 for r in test_rows if r["Label"] == "Anomaly")

print(f"train: {len(train_rows):,} ({train_anomaly:,} anomaly)")
print(f"test: {len(test_rows):,} ({test_anomaly:,} anomaly)")

phase_counts = Counter()
tier_by_phase = defaultdict(Counter)
for r in all_rows:
    if r["Phase"] != "none":
        phase_counts[r["Phase"]] += 1
        tier_by_phase[r["Phase"]][r["Tier"]] += 1

print("\nphase total attack in_win")
for phase, count in phase_counts.most_common():
    atk = tier_by_phase[phase].get("ATTACK", 0)
    inw = tier_by_phase[phase].get("IN_WINDOW", 0)
    print(f"{phase} {count:,} {atk:,} {inw:,}")

if drop_reasons:
    print("\ndropped:")
    for reason, count in drop_reasons.most_common():
        print(f"{reason}: {count:,}")

print(f"\noutput: {OUT_CSV}")
