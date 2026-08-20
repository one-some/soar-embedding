import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
)
from collections import defaultdict
from torch_geometric.loader import DataLoader

from graph_model import GraphAnomalyModel

GRAPHS_PATH = Path("../processed_v2/graph_windows.pt")
META_PATH = Path("../processed_v2/graph_meta.json")
MANIFEST_PATH = Path("../processed_v2/windows_manifest.json")
MODEL_DIR = Path("graph_output")
MODEL_PATH = MODEL_DIR / "best_graph.pt"

# architecture
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
HOMO = True
POOL = "max"

# training
BATCH_SIZE = 64
FINETUNE_LR = 5e-4
FINETUNE_EPOCHS = 200
FINETUNE_PATIENCE = 25
OVERSAMPLE_RATIO = 10
SEED = None
# cap BENIGN/ATTACK nodes contributing to the loss per window
MAX_POS_NODES_PER_WINDOW = 64
MAX_NEG_NODES_PER_WINDOW = 64

# run mode
EVAL_ONLY = False
NO_SAVE = False
TAG = ""

# ablation toggles (all off = full model)
NO_TEMPORAL = False
DISABLE_TEXT = False
DISABLE_IP = False
DISABLE_PROC = False
DISABLE_DOMAIN = False
DISABLE_ALERT_POS = False
DISABLE_SIG = False
RANDOM_TEXT = False
USE_SAGE = False


def load_data(strip_temporal=False):
    graphs = torch.load(GRAPHS_PATH, weights_only=False)

    if strip_temporal:
        key = ("alert", "next", "alert")
        for g in graphs:
            if key in g.edge_types:
                del g[key]

    train_normal = [g for g in graphs if g.split == "train" and g.y.item() == 0]
    train_anomaly = [g for g in graphs if g.split == "train" and g.y.item() == 1]
    test_normal = [g for g in graphs if g.split == "test" and g.y.item() == 0]
    test_anomaly = [g for g in graphs if g.split == "test" and g.y.item() == 1]

    print(f"train: {len(train_normal)} normal, {len(train_anomaly)} anomaly")
    print(f"test:  {len(test_normal)} normal, {len(test_anomaly)} anomaly")
    return train_normal, train_anomaly, test_normal, test_anomaly, graphs


def get_metadata_and_vocab(graphs):
    node_types = set()
    edge_types = set()
    for g in graphs[:500]:
        node_types.update(g.node_types)
        edge_types.update(g.edge_types)

    with open(META_PATH) as f:
        meta = json.load(f)

    return (list(node_types), list(edge_types)), meta["n_sigs"], meta["n_processes"]


def _node_loss(model, batch, device):
    batch = batch.to(device)
    node_logits = model.classify_nodes(batch)
    node_labels = batch["alert"].y.float().to(device)
    win_batch = batch["alert"].batch

    sel_logits, sel_labels = [], []
    for w in win_batch.unique():
        mask = win_batch == w
        logits_w = node_logits[mask]
        labels_w = node_labels[mask]

        pos_idx = (labels_w == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_w == 0).nonzero(as_tuple=True)[0]

        if len(pos_idx) > MAX_POS_NODES_PER_WINDOW:
            perm = torch.randperm(len(pos_idx), device=device)
            pos_idx = pos_idx[perm[:MAX_POS_NODES_PER_WINDOW]]

        n_neg = max(len(pos_idx), 1) if len(pos_idx) > 0 else MAX_NEG_NODES_PER_WINDOW
        n_neg = min(n_neg, MAX_NEG_NODES_PER_WINDOW, len(neg_idx))
        if len(neg_idx) > n_neg:
            perm = torch.randperm(len(neg_idx), device=device)
            neg_idx = neg_idx[perm[:n_neg]]

        all_idx = torch.cat([pos_idx, neg_idx])
        sel_logits.append(logits_w[all_idx])
        sel_labels.append(labels_w[all_idx])

    if not sel_logits:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_cat = torch.cat(sel_logits)
    labels_cat = torch.cat(sel_labels)
    return nn.functional.binary_cross_entropy_with_logits(logits_cat, labels_cat)


def finetune(
    model, train_normal, train_anomaly, device, oversample_ratio=OVERSAMPLE_RATIO
):
    print("finetune (node classification):")
    n_anom_eff = len(train_anomaly) * oversample_ratio
    n_normal_sample = min(len(train_normal), n_anom_eff * 2)
    print(
        f"{len(train_anomaly)} anomaly x{oversample_ratio} = {n_anom_eff}, {n_normal_sample} normal/epoch"
    )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)

    best_loss = float("inf")
    patience = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        normal_idx = np.random.choice(len(train_normal), n_normal_sample, replace=False)
        normal_sample = [train_normal[i] for i in normal_idx]
        finetune_data = normal_sample + train_anomaly * oversample_ratio
        np.random.shuffle(finetune_data)

        loader = DataLoader(finetune_data, batch_size=BATCH_SIZE, shuffle=True)
        total_loss, n = 0, 0

        for batch in loader:
            optimizer.zero_grad()
            loss = _node_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg = total_loss / n
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch} loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= FINETUNE_PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    print(f"best loss={best_loss:.4f}")


def evaluate(
    model,
    test_normal,
    test_anomaly,
    train_normal,
    train_anomaly,
    device,
    save_scores=True,
):
    model.eval()

    def score_set(data_list):
        if not data_list:
            return []
        loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model.classify(batch)
                probs = torch.sigmoid(logits)
                scores.extend(probs.cpu().tolist())
        return scores

    normal_scores = score_set(test_normal)
    anomaly_scores = score_set(test_anomaly)
    train_normal_scores = score_set(train_normal)
    train_anomaly_scores = score_set(train_anomaly)

    all_scores = normal_scores + anomaly_scores
    all_labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)

    if not anomaly_scores:
        print("no anomaly windows")
        return

    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)

    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        preds = [1 if s > t else 0 for s in all_scores]
        f = f1_score(all_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    preds_at_best = [1 if s > best_thresh else 0 for s in all_scores]

    print(
        f"AUROC={auroc:.4f} AUPRC={auprc:.4f} F1={best_f1:.4f} (thresh={best_thresh:.2f})"
    )
    print(f"normal:  mean={np.mean(normal_scores):.4f} std={np.std(normal_scores):.4f}")
    print(
        f"anomaly: mean={np.mean(anomaly_scores):.4f} std={np.std(anomaly_scores):.4f}"
    )
    print(
        classification_report(
            all_labels, preds_at_best, target_names=["Normal", "Anomaly"]
        )
    )

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    test_anomaly_meta = [
        w for w in manifest if w["split"] == "test" and w["label"] == 1
    ]

    if test_anomaly_meta and len(anomaly_scores) == len(test_anomaly_meta):
        print("per-phase recall:")
        phase_results = defaultdict(lambda: {"detected": 0, "total": 0})
        for m, score in zip(test_anomaly_meta, anomaly_scores):
            for phase in m.get("phases", []):
                phase_results[phase]["total"] += 1
                if score > best_thresh:
                    phase_results[phase]["detected"] += 1
        for phase in sorted(phase_results):
            r = phase_results[phase]
            recall = r["detected"] / r["total"] if r["total"] else 0
            print(f"{phase}: {r['detected']}/{r['total']} ({recall:.0%})")

    results = []
    for scores, lbl, name in [
        (train_normal_scores, 0, "train_normal"),
        (train_anomaly_scores, 1, "train_anomaly"),
        (normal_scores, 0, "test_normal"),
        (anomaly_scores, 1, "test_anomaly"),
    ]:
        for s in scores:
            results.append({"score": s, "label": lbl, "set": name})
    if save_scores:
        with open(MODEL_DIR / "graph_scores.json", "w") as f:
            json.dump(results, f, indent=2)


def main():
    global MODEL_PATH
    if TAG:
        MODEL_PATH = MODEL_DIR / f"best_graph_{TAG}.pt"

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    MODEL_DIR.mkdir(exist_ok=True)

    train_normal, train_anomaly, test_normal, test_anomaly, all_graphs = load_data(
        strip_temporal=NO_TEMPORAL
    )
    metadata, n_sigs, n_procs = get_metadata_and_vocab(all_graphs)
    print(f"vocab: {n_sigs} sigs, {n_procs} processes")

    enc_kwargs = {}
    if HOMO:
        enc_kwargs.update(
            disable_text=DISABLE_TEXT,
            disable_ip=DISABLE_IP,
            disable_proc=DISABLE_PROC,
            disable_domain=DISABLE_DOMAIN,
            disable_alert_pos=DISABLE_ALERT_POS,
            disable_sig=DISABLE_SIG,
            random_text=RANDOM_TEXT,
            use_sage=USE_SAGE,
        )
    model = GraphAnomalyModel(
        HIDDEN_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        n_sigs=n_sigs,
        n_processes=n_procs,
        metadata=metadata,
        text_emb_dim=384,
        pool=POOL,
        homo=HOMO,
        **enc_kwargs,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    if EVAL_ONLY:
        model.load_state_dict(
            torch.load(MODEL_PATH, weights_only=False, map_location=device)
        )
    else:
        finetune(
            model,
            train_normal,
            train_anomaly,
            device,
            oversample_ratio=OVERSAMPLE_RATIO,
        )

    evaluate(
        model,
        test_normal,
        test_anomaly,
        train_normal,
        train_anomaly,
        device,
        save_scores=not NO_SAVE,
    )


if __name__ == "__main__":
    main()
