import json
import sys

sys.path.insert(0, ".")
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader
from graph_model import GraphAnomalyModel

HIDDEN, LAYERS, HEADS = 128, 4, 4
SEEDS = list(range(1, 9))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load("../processed_v2/graph_windows.pt", weights_only=False)
with open("../processed_v2/graph_meta.json") as f:
    meta = json.load(f)
md = (
    list({k for g in graphs[:500] for k in g.node_types}),
    list({k for g in graphs[:500] for k in g.edge_types}),
)

test_n = [g for g in graphs if g.split == "test" and g.y.item() == 0]
test_a = [g for g in graphs if g.split == "test" and g.y.item() == 1]
y = np.array([0] * len(test_n) + [1] * len(test_a))

per_seed_scores = []
per_seed_auprc = []
per_seed_auroc = []
for s in SEEDS:
    m = (
        GraphAnomalyModel(
            HIDDEN,
            HEADS,
            LAYERS,
            n_sigs=meta["n_sigs"],
            n_processes=meta["n_processes"],
            metadata=md,
            pool="max",
            homo=True,
        )
        .to(device)
        .eval()
    )
    m.load_state_dict(
        torch.load(
            f"graph_output/best_graph_homogt128_4L_s{s}.pt",
            weights_only=False,
            map_location=device,
        )
    )
    loader = DataLoader(test_n + test_a, batch_size=64, shuffle=False)
    scores = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            scores.extend(torch.sigmoid(m.classify(b)).cpu().tolist())
    s_arr = np.array(scores)
    per_seed_scores.append(s_arr)
    per_seed_auprc.append(average_precision_score(y, s_arr))
    per_seed_auroc.append(roc_auc_score(y, s_arr))

per_seed_auprc = np.array(per_seed_auprc)
per_seed_auroc = np.array(per_seed_auroc)
print("Per-seed (saved ckpt) metrics:")
for i, s in enumerate(SEEDS):
    print(f"s{s}: AUROC={per_seed_auroc[i]:.4f}  AUPRC={per_seed_auprc[i]:.4f}")
print(f"mean AUROC = {per_seed_auroc.mean():.4f} +/- {per_seed_auroc.std(ddof=1):.4f}")
print(f"mean AUPRC = {per_seed_auprc.mean():.4f} +/- {per_seed_auprc.std(ddof=1):.4f}")

# threshold sweep
thresholds = np.arange(0.0, 1.01, 0.01)
P_arr, R_arr, F_arr = [], [], []
for s_scores in per_seed_scores:
    p_row, r_row, f_row = [], [], []
    for t in thresholds:
        pred = (s_scores > t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        p_row.append(p)
        r_row.append(r)
        f_row.append(f)
    P_arr.append(p_row)
    R_arr.append(r_row)
    F_arr.append(f_row)
P_mean = np.mean(P_arr, axis=0)
R_mean = np.mean(R_arr, axis=0)
F_mean = np.mean(F_arr, axis=0)

rows = ["threshold,precision,recall,f1"]
for i, t in enumerate(thresholds):
    rows.append(f"{t:.2f},{P_mean[i]:.4f},{R_mean[i]:.4f},{F_mean[i]:.4f}")
with open("sweep.csv", "w") as f:
    f.write("\n".join(rows) + "\n")
best_i = int(np.argmax(F_mean))
print(f"\nsweep.csv written (per-seed averaged).")
print(
    f"optimum: t* = {thresholds[best_i]:.2f}  P={P_mean[best_i]:.3f}  R={R_mean[best_i]:.3f}  F1={F_mean[best_i]:.3f}"
)

# PR curve
rec_grid = np.linspace(0.0, 1.0, 51)
prec_grids = []
for s_scores in per_seed_scores:
    pc, rc, _ = precision_recall_curve(y, s_scores)
    order = np.argsort(rc)
    rc = rc[order]
    pc = pc[order]
    # right-continuous step interpolation (standard PR curve shape)
    pgrid = []
    for g_rec in rec_grid:
        # largest precision among points with recall >= g_rec
        mask = rc >= g_rec
        pgrid.append(pc[mask].max() if mask.any() else 0.0)
    prec_grids.append(pgrid)
prec_mean = np.mean(prec_grids, axis=0)

# downsample to ~20 points for tikz
keep_idx = np.linspace(0, len(rec_grid) - 1, 21).astype(int)
keep_idx = sorted(set(keep_idx.tolist()))
print(f"\nProposed Model PR curve coords (per-seed averaged, {len(keep_idx)} points):")
coords = " ".join(f"({rec_grid[i]:.2f},{prec_mean[i]:.2f})" for i in keep_idx)
print(" " + coords)
print(
    f"\nLegend AUPRC label: {per_seed_auprc.mean():.3f} (mean across {len(SEEDS)} seeds)"
)
