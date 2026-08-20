import orjson
import json
import re
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

TEXT_EMB_DIM = 384
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

LABELED_DIR = Path("labeled_v2")
OUT_DIR = Path("processed_v2")
# seconds, matches LogBERT
WINDOW_SIZE = 300
MAX_ALERTS_PER_WINDOW = 500
RANDOM_SEED = 42

TEST_RATIO_NORMAL = 0.10
TEST_RATIO_ANOMALY = 0.30
SPLIT_SEED = 42

MIN_CONFIDENCE_MALICIOUS = 0.70
MIN_CONFIDENCE_SUSPICIOUS = 0.50

# dnsteal: labeled windows mark when the attacker ran the dnsteal command, but
# the actual DNS exfil traffic runs on a background cron 2-4 days earlier in
# all 8 scenarios. Labeled windows contain only unrelated AMiner false positives.
EXCLUDED_PHASES: frozenset = frozenset({"dnsteal"})

BAD_PROCESSES = {"Decode]", "started", "HTTP/1.1", "auth", "HORDE", "msg", "HTTP"}
BAD_DOMAINS_RE = re.compile(r"\.(php|html|js|css|txt|log|jpg|png|gif)$", re.I)

_ZERO_EMB = np.zeros(TEXT_EMB_DIM, dtype=np.float32)

TRAIN_RATIO = 0.9
SEED = 777

sig_vocab = {}
process_vocab = {}


def get_sig_id(sig):
    if sig not in sig_vocab:
        sig_vocab[sig] = len(sig_vocab) + 1
    return sig_vocab[sig]


def get_process_id(proc):
    if proc not in process_vocab:
        process_vocab[proc] = len(process_vocab) + 1
    return process_vocab[proc]


def build_alert_text(rec) -> str:
    # signature + category only (not full_log) so semantically identical alerts
    # share an embedding regardless of per-instance IPs/timestamps/session IDs.
    parts = []
    if rec["signature"]:
        parts.append(rec["signature"])
    if rec["category"]:
        parts.append(rec["category"])
    return " ".join(parts) if parts else "unknown alert"


def parse_record(record):
    gt = record.get("ground_truth", {})
    ents = record.get("entities", {})
    meta = record.get("metadata", {})

    try:
        ts = int(datetime.fromisoformat(record["timestamp"]).timestamp())
    except Exception:
        return None

    label = gt.get("label", "UNKNOWN")
    phase = gt.get("attack_phase")
    confidence = gt.get("confidence", 0.0)

    processes = [p for p in ents.get("processes", []) if p and p not in BAD_PROCESSES]
    domains = [d for d in ents.get("domains", []) if d and not BAD_DOMAINS_RE.search(d)]

    full_log = (meta.get("full_log") or "").strip()
    signature = (meta.get("signature") or "").strip()
    category = (meta.get("category") or "").strip()

    return {
        "ts": ts,
        "label": label,
        "phase": phase,
        "confidence": confidence,
        "src_ip": ents.get("src_ip"),
        "dst_ip": ents.get("dst_ip"),
        "processes": processes,
        "domains": domains[:5],
        "full_log": full_log,
        "signature": signature,
        "category": category,
        "rule_groups": meta.get("rule_groups") or [],
        "source": record.get("source", "unknown"),
    }


def classify_tier(rec):
    label, phase, conf = rec["label"], rec["phase"], rec["confidence"]

    if label in ("MALICIOUS", "SUSPICIOUS") and phase:
        if label == "SUSPICIOUS" and conf < MIN_CONFIDENCE_SUSPICIOUS:
            return "FILTERED"
        if label == "MALICIOUS" and conf < MIN_CONFIDENCE_MALICIOUS:
            return "FILTERED"
        return "ATTACK"
    elif phase:
        return "IN_WINDOW"
    elif label == "BENIGN":
        return "NORMAL"
    else:
        return "FILTERED"


def build_logbert_content(rec):
    if rec["full_log"]:
        return rec["full_log"]

    parts = []
    if rec["signature"]:
        parts.append(f"sig:{rec['signature']}")
    if rec["category"]:
        parts.append(f"cat:{rec['category']}")
    if rec["rule_groups"]:
        parts.append(f"groups:{','.join(rec['rule_groups'])}")
    src = rec["src_ip"] or "?"
    dst = rec["dst_ip"] or "?"
    if src != "?" or dst != "?":
        parts.append(f"flow:{src}->{dst}")
    return " | ".join(parts)


def compute_ip_features(ip_str, ip_alerts, n_total, all_ips, ip_history=None):
    n = len(ip_alerts)
    sigs = set()
    peers = set()
    as_src = as_dst = 0
    for a in ip_alerts:
        sigs.add(a["signature"])
        if a["src_ip"] == ip_str:
            as_src += 1
            if a["dst_ip"] and a["dst_ip"] != ip_str:
                peers.add(a["dst_ip"])
        if a["dst_ip"] == ip_str:
            as_dst += 1
            if a["src_ip"] and a["src_ip"] != ip_str:
                peers.add(a["src_ip"])
    feats = [
        np.log1p(n),
        np.log1p(len(peers)),
        np.log1p(len(sigs)),
        as_src / max(as_src + as_dst, 1),
        n / max(n_total, 1),
    ]
    hist = (ip_history or {}).get(ip_str, {})
    feats += [
        np.log1p(hist.get("n_windows", 0)),
        np.log1p(hist.get("n_attack_windows", 0)),
        float(hist.get("last_window_attack", False)),
    ]
    return feats


_HEX_CHARS = frozenset("0123456789abcdef")


def compute_domain_features(dom_str, alert_count, n_alerts):
    f_freq = np.log1p(alert_count)
    f_ratio = alert_count / max(n_alerts, 1)

    labels = dom_str.split(".")
    n_labels = len(labels)
    max_label_len = float(max((len(l) for l in labels), default=0))

    subdom = (
        ".".join(labels[:-2]) if n_labels > 2 else (labels[0] if n_labels == 1 else "")
    )

    if subdom:
        ch_counts = Counter(subdom.lower())
        total = len(subdom)
        entropy = -sum((c / total) * np.log2(c / total) for c in ch_counts.values())
        hex_ratio = sum(1 for c in subdom.lower() if c in _HEX_CHARS) / max(total, 1)
    else:
        entropy = 0.0
        hex_ratio = 0.0

    return [f_freq, f_ratio, entropy, max_label_len, hex_ratio, float(n_labels)]


def build_graph(alerts, ip_history=None, text_to_emb=None):
    if len(alerts) > MAX_ALERTS_PER_WINDOW:
        rng = np.random.RandomState(RANDOM_SEED)
        # Always keep ATTACK alerts. Fill the rest from the others.
        attack_idx = [i for i, a in enumerate(alerts) if a["tier"] == "ATTACK"]
        other_idx = [i for i, a in enumerate(alerts) if a["tier"] != "ATTACK"]
        if len(attack_idx) <= MAX_ALERTS_PER_WINDOW:
            n_other = MAX_ALERTS_PER_WINDOW - len(attack_idx)
            other_sample = rng.choice(
                len(other_idx), min(n_other, len(other_idx)), replace=False
            ).tolist()
            selected = sorted(attack_idx + [other_idx[i] for i in other_sample])
        else:
            selected = sorted(
                rng.choice(attack_idx, MAX_ALERTS_PER_WINDOW, replace=False).tolist()
            )
        alerts = [alerts[i] for i in selected]

    alerts.sort(key=lambda a: a["ts"])
    n = len(alerts)
    data = HeteroData()

    ip_map, proc_map, domain_map = {}, {}, {}
    alert_feats, alert_sig_ids = [], []
    alert_src, alert_dst, alert_proc, alert_domain = (
        ([], []),
        ([], []),
        ([], []),
        ([], []),
    )
    ip_flow, temporal = ([], []), ([], [])
    ip_alert_refs = defaultdict(list)

    text_embs = []
    for i, a in enumerate(alerts):
        sid = get_sig_id(a["signature"])
        pos = i / max(n - 1, 1)
        alert_feats.append([float(sid), pos])
        alert_sig_ids.append(sid)
        if text_to_emb is not None:
            text_embs.append(text_to_emb.get(build_alert_text(a), _ZERO_EMB))
        else:
            text_embs.append(_ZERO_EMB)

        src, dst = a["src_ip"], a["dst_ip"]
        if src:
            if src not in ip_map:
                ip_map[src] = len(ip_map)
            alert_src[0].append(i)
            alert_src[1].append(ip_map[src])
            ip_alert_refs[src].append(a)
        if dst:
            if dst not in ip_map:
                ip_map[dst] = len(ip_map)
            alert_dst[0].append(i)
            alert_dst[1].append(ip_map[dst])
            ip_alert_refs[dst].append(a)
        if src and dst:
            ip_flow[0].append(ip_map[src])
            ip_flow[1].append(ip_map[dst])

        for proc in a["processes"]:
            get_process_id(proc)
            if proc not in proc_map:
                proc_map[proc] = len(proc_map)
            alert_proc[0].append(i)
            alert_proc[1].append(proc_map[proc])

        for dom in a["domains"]:
            if dom not in domain_map:
                domain_map[dom] = len(domain_map)
            alert_domain[0].append(i)
            alert_domain[1].append(domain_map[dom])

        if i > 0:
            temporal[0].append(i - 1)
            temporal[1].append(i)

    data["alert"].x = torch.tensor(alert_feats, dtype=torch.float)
    data["alert"].sig_ids = torch.tensor(alert_sig_ids, dtype=torch.long)
    data["alert"].text_emb = torch.tensor(np.stack(text_embs), dtype=torch.float)
    data["alert"].y = torch.tensor(
        [1 if a["tier"] == "ATTACK" else 0 for a in alerts], dtype=torch.long
    )
    data["alert"].num_nodes = n

    if ip_map:
        all_ips = set(ip_map.keys())
        ip_feats = [
            compute_ip_features(ip, ip_alert_refs[ip], n, all_ips, ip_history)
            for ip in sorted(ip_map, key=ip_map.get)
        ]
        data["ip"].x = torch.tensor(ip_feats, dtype=torch.float)
        data["ip"].num_nodes = len(ip_map)

    if proc_map:
        proc_ids = [get_process_id(p) for p in sorted(proc_map, key=proc_map.get)]
        data["process"].x = torch.tensor(proc_ids, dtype=torch.long).unsqueeze(1)
        data["process"].num_nodes = len(proc_map)

    if domain_map:
        dom_feats = []
        for dom in sorted(domain_map, key=domain_map.get):
            ct = sum(1 for a in alerts if dom in a["domains"])
            dom_feats.append(compute_domain_features(dom, ct, n))
        data["domain"].x = torch.tensor(dom_feats, dtype=torch.float)
        data["domain"].num_nodes = len(domain_map)

    def set_edge(s, r, d, pair):
        if pair[0]:
            data[s, r, d].edge_index = torch.tensor(pair, dtype=torch.long)

    def set_edge_rev(s, r, d, pair):
        if pair[0]:
            data[s, r, d].edge_index = torch.tensor(
                [pair[1], pair[0]], dtype=torch.long
            )

    set_edge("alert", "has_src", "ip", alert_src)
    set_edge("alert", "has_dst", "ip", alert_dst)
    set_edge("alert", "has_process", "process", alert_proc)
    set_edge("alert", "has_domain", "domain", alert_domain)
    set_edge("ip", "flow", "ip", ip_flow)
    set_edge("alert", "next", "alert", temporal)

    # HGT needs these to propagate IP/domain context back to alert nodes.
    set_edge_rev("ip", "rev_src", "alert", alert_src)
    set_edge_rev("ip", "rev_dst", "alert", alert_dst)
    set_edge_rev("domain", "rev_domain", "alert", alert_domain)
    set_edge_rev("process", "rev_proc", "alert", alert_proc)

    return data


def build_llm_summary(alerts, max_alerts=50):
    if len(alerts) > max_alerts:
        sampled = [alerts[0], alerts[-1]]
        mid = random.sample(alerts[1:-1], min(max_alerts - 2, len(alerts) - 2))
        sampled.extend(mid)
        sampled.sort(key=lambda a: a["ts"])
    else:
        sampled = alerts

    lines = []
    for a in sampled:
        parts = []
        if a["full_log"]:
            parts.append(a["full_log"][:200])
        else:
            if a["signature"]:
                parts.append(f"[{a['signature']}]")
            if a["category"]:
                parts.append(f"({a['category']})")
        if a["src_ip"]:
            parts.append(f"src={a['src_ip']}")
        if a["dst_ip"]:
            parts.append(f"dst={a['dst_ip']}")
        if a["processes"]:
            parts.append(f"proc={','.join(a['processes'][:3])}")
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _collect_unique_alert_texts() -> set:
    texts = set()
    for j_path in sorted(LABELED_DIR.glob("*_comprehensive.json")):
        with j_path.open("rb") as f:
            data = orjson.loads(f.read())
        for r in data:
            parsed = parse_record(r)
            if not parsed:
                continue
            if classify_tier(parsed) == "FILTERED":
                continue
            texts.add(build_alert_text(parsed))
    return texts


def build_text_embedding_cache(device: str = "cpu") -> dict:
    unique_texts = sorted(_collect_unique_alert_texts())
    print(f"Embedding {len(unique_texts)} unique alert texts on {device}")

    model = SentenceTransformer(TEXT_MODEL_NAME)
    embeddings = model.encode(
        unique_texts,
        batch_size=512,
        device=device,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return {t: embeddings[i] for i, t in enumerate(unique_texts)}


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_to_emb = build_text_embedding_cache(device=device)

    all_windows = []

    for j_path in sorted(LABELED_DIR.glob("*_comprehensive.json")):
        scenario = j_path.stem.replace("_comprehensive", "")
        print(f"{scenario}...", end=" ", flush=True)

        with j_path.open("rb") as f:
            data = orjson.loads(f.read())

        records = []
        n_filtered = 0
        for r in data:
            parsed = parse_record(r)
            if not parsed:
                n_filtered += 1
                continue
            parsed["tier"] = classify_tier(parsed)
            if parsed["tier"] == "FILTERED":
                n_filtered += 1
                continue
            records.append(parsed)

        records.sort(key=lambda r: r["ts"])
        if not records:
            print(f"no records (filtered {n_filtered})")
            continue

        min_ts = records[0]["ts"]

        windows_by_id = defaultdict(list)
        for r in records:
            win_id = (r["ts"] - min_ts) // WINDOW_SIZE
            windows_by_id[win_id].append(r)

        n_attack = 0
        ip_history = {}
        for win_id in sorted(windows_by_id):
            alerts = windows_by_id[win_id]

            # Anomaly = at least one ATTACK alert. IN_WINDOW (BENIGN during
            # attack time) alone doesn't count - avoids labeling temporal
            # coincidence as attack.
            phases = sorted(
                set(a["phase"] for a in alerts if a["phase"] and a["tier"] == "ATTACK")
            )
            tiers = Counter(a["tier"] for a in alerts)
            is_anomaly = int(tiers.get("ATTACK", 0) > 0)

            # Drop windows whose only attack phase is excluded (e.g. dnsteal).
            if phases and set(phases) <= EXCLUDED_PHASES:
                continue
            if is_anomaly:
                n_attack += 1

            logbert_contents = [build_logbert_content(a) for a in alerts]
            logbert_contents = [c for c in logbert_contents if c]

            graph = build_graph(alerts, ip_history=ip_history, text_to_emb=text_to_emb)
            graph.y = torch.tensor([is_anomaly])

            window_ips = set()
            for a in alerts:
                if a["src_ip"]:
                    window_ips.add(a["src_ip"])
                if a["dst_ip"]:
                    window_ips.add(a["dst_ip"])
            for ip in window_ips:
                h = ip_history.setdefault(
                    ip,
                    {
                        "n_windows": 0,
                        "n_attack_windows": 0,
                        "last_window_attack": False,
                    },
                )
                h["n_windows"] += 1
                if is_anomaly:
                    h["n_attack_windows"] += 1
                h["last_window_attack"] = bool(is_anomaly)
            graph.scenario = scenario
            graph.n_alerts_original = len(alerts)

            llm_text = build_llm_summary(alerts)

            all_windows.append(
                {
                    "scenario": scenario,
                    "window_id": int(win_id),
                    "window_start": min_ts + win_id * WINDOW_SIZE,
                    "label": is_anomaly,
                    "phases": phases,
                    "tiers": dict(tiers),
                    "n_alerts": len(alerts),
                    "n_ips": graph["ip"].num_nodes if "ip" in graph.node_types else 0,
                    "n_procs": (
                        graph["process"].num_nodes
                        if "process" in graph.node_types
                        else 0
                    ),
                    "_logbert_contents": logbert_contents,
                    "_graph": graph,
                    "_llm_text": llm_text,
                }
            )

        print(
            f"{len(data):,} records -> {len(windows_by_id)} windows ({n_attack} attack), filtered {n_filtered}"
        )

    # Stratified split: within each scenario, hold out TEST_RATIO_NORMAL of
    # normals and TEST_RATIO_ANOMALY of anomalies.
    rng = random.Random(SPLIT_SEED)

    for scenario in sorted(set(w["scenario"] for w in all_windows)):
        normal_idx = [
            i
            for i, w in enumerate(all_windows)
            if w["scenario"] == scenario and w["label"] == 0
        ]
        anomaly_idx = [
            i
            for i, w in enumerate(all_windows)
            if w["scenario"] == scenario and w["label"] == 1
        ]

        rng.shuffle(normal_idx)
        rng.shuffle(anomaly_idx)

        n_test_normal = max(1, int(len(normal_idx) * TEST_RATIO_NORMAL))
        n_test_anomaly = (
            max(1, int(len(anomaly_idx) * TEST_RATIO_ANOMALY)) if anomaly_idx else 0
        )

        test_idx = set(normal_idx[:n_test_normal]) | set(anomaly_idx[:n_test_anomaly])

        for i in range(len(all_windows)):
            if all_windows[i]["scenario"] == scenario:
                all_windows[i]["split"] = "test" if i in test_idx else "train"
                all_windows[i]["_graph"].split = all_windows[i]["split"]

    n_total = len(all_windows)
    n_anomaly = sum(w["label"] for w in all_windows)
    n_train = sum(1 for w in all_windows if w["split"] == "train")
    n_test = sum(1 for w in all_windows if w["split"] == "test")
    train_anom = sum(
        1 for w in all_windows if w["split"] == "train" and w["label"] == 1
    )
    test_anom = sum(1 for w in all_windows if w["split"] == "test" and w["label"] == 1)

    print(f"{n_total} windows ({n_anomaly} anomaly)")
    print(
        f"train {n_train} ({train_anom} anomaly), test {n_test} ({test_anom} anomaly)"
    )
    print(f"vocab: {len(sig_vocab)} sigs, {len(process_vocab)} processes")

    manifest = []
    for i, w in enumerate(all_windows):
        manifest.append(
            {
                "idx": i,
                "scenario": w["scenario"],
                "split": w["split"],
                "window_id": w["window_id"],
                "window_start": w["window_start"],
                "label": w["label"],
                "phases": w["phases"],
                "tiers": w["tiers"],
                "n_alerts": w["n_alerts"],
                "n_ips": w["n_ips"],
                "n_procs": w["n_procs"],
            }
        )

    manifest_path = OUT_DIR / "windows_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    graphs = [w["_graph"] for w in all_windows]
    torch.save(graphs, OUT_DIR / "graph_windows.pt")

    graph_meta = {
        "sig_vocab": sig_vocab,
        "process_vocab": process_vocab,
        "n_sigs": len(sig_vocab) + 1,
        "n_processes": len(process_vocab) + 1,
    }
    with open(OUT_DIR / "graph_meta.json", "w") as f:
        json.dump(graph_meta, f, indent=2)

    # LogBERT runs Drain over these raw contents in its own data_process.py.
    logbert_dir = OUT_DIR / "logbert_windows"
    logbert_dir.mkdir(exist_ok=True)

    train_normal = [
        (i, w)
        for i, w in enumerate(all_windows)
        if w["split"] == "train" and w["label"] == 0
    ]
    train_anomaly = [
        (i, w)
        for i, w in enumerate(all_windows)
        if w["split"] == "train" and w["label"] == 1
    ]
    test_normal = [
        (i, w)
        for i, w in enumerate(all_windows)
        if w["split"] == "test" and w["label"] == 0
    ]
    test_anomaly = [
        (i, w)
        for i, w in enumerate(all_windows)
        if w["split"] == "test" and w["label"] == 1
    ]

    def write_logbert_windows(windows_with_idx, path):
        with open(path, "w") as f:
            for idx, w in windows_with_idx:
                f.write(
                    json.dumps({"idx": idx, "contents": w["_logbert_contents"]}) + "\n"
                )

    write_logbert_windows(train_normal, logbert_dir / "train_normal.jsonl")
    write_logbert_windows(train_anomaly, logbert_dir / "train_anomaly.jsonl")
    write_logbert_windows(test_normal, logbert_dir / "test_normal.jsonl")
    write_logbert_windows(test_anomaly, logbert_dir / "test_abnormal.jsonl")

    llm_data = []
    llm_train_data = []
    for i, w in enumerate(all_windows):
        entry = {
            "idx": i,
            "label": w["label"],
            "phases": w["phases"],
            "scenario": w["scenario"],
            "n_alerts": w["n_alerts"],
            "text": w["_llm_text"],
        }
        if w["split"] == "test":
            llm_data.append(entry)
        elif w["split"] == "train":
            # Need anomaly text in train for honest OCSVM scoring.
            llm_train_data.append(entry)

    with open(OUT_DIR / "llm_windows.json", "w") as f:
        json.dump(llm_data, f, indent=2)
    with open(OUT_DIR / "llm_windows_train.json", "w") as f:
        json.dump(llm_train_data, f, indent=2)

    for i, (m, g) in enumerate(zip(manifest, graphs)):
        assert m["label"] == g.y.item(), f"Label mismatch at window {i}"
        assert m["scenario"] == g.scenario, f"Scenario mismatch at window {i}"
        assert m["split"] == g.split, f"Split mismatch at window {i}"

    print(f"wrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
