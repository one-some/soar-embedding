import orjson
import torch
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import HeteroData

LABELED_DIR = Path("labeled_v2")
OUT_PATH = Path("processed_v2/graph_windows.pt")
META_PATH = Path("processed_v2/graph_meta.json")

WINDOW_SIZE = 300
MAX_ALERTS_PER_WINDOW = 500
RANDOM_SEED = 42

TEST_SCENARIOS = {"harrison", "santos"}

BAD_PROCESSES = {"Decode]", "started", "HTTP/1.1", "auth", "HORDE", "msg", "HTTP"}
BAD_USERS = {"HTTP", "0", "4294967295", "msg", "33"}

sig_vocab = {}
process_vocab = {}


def get_sig_id(sig: str) -> int:
    if sig not in sig_vocab:
        sig_vocab[sig] = len(sig_vocab) + 1
    return sig_vocab[sig]


def get_process_id(proc: str) -> int:
    if proc not in process_vocab:
        process_vocab[proc] = len(process_vocab) + 1
    return process_vocab[proc]


def clean_processes(procs):
    return [p for p in procs if p and p not in BAD_PROCESSES]


def extract_record(record):
    gt = record.get("ground_truth", {})
    meta = record.get("metadata", {})
    ents = record.get("entities", {})

    label = gt.get("label", "UNKNOWN")
    phase = gt.get("attack_phase")
    confidence = gt.get("confidence", 0.0)

    try:
        ts = int(datetime.fromisoformat(record["timestamp"]).timestamp())
    except Exception:
        return None

    sig = (meta.get("signature") or "").strip()
    src_ip = ents.get("src_ip")
    dst_ip = ents.get("dst_ip")
    processes = clean_processes(ents.get("processes", []))
    domains = ents.get("domains", [])
    domains = [
        d
        for d in domains
        if d and not re.search(r"\.\w{2,4}$", d) or "." in d.split("/")[0]
    ]

    is_attack = (label in ("MALICIOUS", "SUSPICIOUS") and phase) or (
        phase and label in ("BENIGN", "PRE_ATTACK", "POST_ATTACK")
    )

    return {
        "ts": ts,
        "sig": sig,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "processes": processes,
        "domains": domains[:5],
        "is_attack": is_attack,
        "phase": phase or "none",
        "label": label,
        "confidence": confidence,
    }


def compute_ip_features(ip_str, ip_alerts, n_total_alerts, all_ips_in_window):
    n_alerts = len(ip_alerts)
    sigs = set()
    peer_ips = set()
    as_src = 0
    as_dst = 0

    for a in ip_alerts:
        sigs.add(a["sig"])
        if a["src_ip"] == ip_str:
            as_src += 1
            if a["dst_ip"] and a["dst_ip"] != ip_str:
                peer_ips.add(a["dst_ip"])
        if a["dst_ip"] == ip_str:
            as_dst += 1
            if a["src_ip"] and a["src_ip"] != ip_str:
                peer_ips.add(a["src_ip"])

    return [
        np.log1p(n_alerts),
        np.log1p(len(peer_ips)),
        np.log1p(len(sigs)),
        as_src / max(as_src + as_dst, 1),
        n_alerts / max(n_total_alerts, 1),
    ]


IP_FEAT_DIM = 5
DOMAIN_FEAT_DIM = 2


def build_window_graph(alerts, scenario, split):
    if len(alerts) > MAX_ALERTS_PER_WINDOW:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(alerts), MAX_ALERTS_PER_WINDOW, replace=False)
        alerts = [alerts[i] for i in sorted(idx)]

    alerts.sort(key=lambda a: a["ts"])
    n = len(alerts)
    data = HeteroData()

    ip_map = {}
    proc_map = {}
    domain_map = {}

    alert_src = ([], [])
    alert_dst = ([], [])
    alert_proc = ([], [])
    alert_domain = ([], [])
    ip_flow = ([], [])
    temporal = ([], [])

    alert_feats = []
    alert_sig_ids = []

    ip_alert_refs = defaultdict(list)

    for i, a in enumerate(alerts):
        sig_id = get_sig_id(a["sig"])
        pos = i / max(n - 1, 1)
        alert_feats.append([float(sig_id), pos])
        alert_sig_ids.append(sig_id)

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
    data["alert"].num_nodes = n

    if ip_map:
        all_ips = set(ip_map.keys())
        ip_feats = [
            compute_ip_features(ip, ip_alert_refs[ip], n, all_ips)
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
            dom_feats.append([np.log1p(ct), ct / max(n, 1)])
        data["domain"].x = torch.tensor(dom_feats, dtype=torch.float)
        data["domain"].num_nodes = len(domain_map)

    def set_edge(src_type, rel, dst_type, pair):
        if pair[0]:
            data[src_type, rel, dst_type].edge_index = torch.tensor(
                pair, dtype=torch.long
            )

    set_edge("alert", "has_src", "ip", alert_src)
    set_edge("alert", "has_dst", "ip", alert_dst)
    set_edge("alert", "has_process", "process", alert_proc)
    set_edge("alert", "has_domain", "domain", alert_domain)
    set_edge("ip", "flow", "ip", ip_flow)
    set_edge("alert", "next", "alert", temporal)

    phases = set(a["phase"] for a in alerts) - {"none"}
    is_anomaly = int(any(a["is_attack"] for a in alerts))

    data.y = torch.tensor([is_anomaly])
    data.scenario = scenario
    data.split = split
    data.n_alerts_original = len(alerts)

    return data, sorted(phases)


def main():
    np.random.seed(RANDOM_SEED)
    OUT_PATH.parent.mkdir(exist_ok=True)

    all_graphs = []
    all_meta = []

    for j_path in sorted(LABELED_DIR.glob("*_comprehensive.json")):
        scenario = j_path.stem.replace("_comprehensive", "")
        split = "test" if scenario in TEST_SCENARIOS else "train"
        print(f"{scenario} [{split}]...", end=" ", flush=True)

        with j_path.open("rb") as f:
            data = orjson.loads(f.read())

        records = [r for r in (extract_record(rec) for rec in data) if r]
        if not records:
            print("no records")
            continue

        records.sort(key=lambda r: r["ts"])
        min_ts = records[0]["ts"]

        windows = defaultdict(list)
        for r in records:
            win_id = (r["ts"] - min_ts) // WINDOW_SIZE
            windows[win_id].append(r)

        n_attack = 0
        for win_id in sorted(windows):
            alerts = windows[win_id]
            graph, phases = build_window_graph(alerts, scenario, split)
            all_graphs.append(graph)
            if graph.y.item() == 1:
                n_attack += 1
            all_meta.append(
                {
                    "scenario": scenario,
                    "window_id": int(win_id),
                    "label": graph.y.item(),
                    "phases": phases,
                    "split": split,
                    "n_alerts": len(alerts),
                    "n_ips": graph["ip"].num_nodes if "ip" in graph.node_types else 0,
                    "n_procs": (
                        graph["process"].num_nodes
                        if "process" in graph.node_types
                        else 0
                    ),
                }
            )

        print(f"{len(windows)} windows ({n_attack} attack)")

    n = len(all_graphs)
    print(f"\ntotal: {n} ({sum(1 for g in all_graphs if g.y.item() == 1)} anomaly)")
    print(
        f"train: {sum(1 for g in all_graphs if g.split == 'train')}, "
        f"test: {sum(1 for g in all_graphs if g.split == 'test')}"
    )
    print(f"vocab: {len(sig_vocab)} sigs, {len(process_vocab)} processes")

    torch.save(all_graphs, OUT_PATH)

    meta_out = {
        "sig_vocab": sig_vocab,
        "process_vocab": process_vocab,
        "n_sigs": len(sig_vocab) + 1,
        "n_processes": len(process_vocab) + 1,
        "windows": all_meta,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"saved: {OUT_PATH}, {META_PATH}")


if __name__ == "__main__":
    main()
