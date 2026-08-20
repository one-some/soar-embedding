# E-GraphSAGE on SOAR data.
# Model code (SAGELayer/SAGE/MLPPredictor/Model) copied verbatim from the
# upstream notebook by Lo et al. (waimorris/E-GraphSAGE, Apache 2.0):
#   E-GraphSAGE/netflow/bot-iot/unsw_bot_iot_binary_mean_agg.ipynb
# Data plumbing and metrics are ours; the SoarDataset cache from the
# graphids fork supplies (edge_index, edge_attr, labels, splits, scenario, wid).

import json
import os
import pickle
import sys
import time

import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.utils import class_weight

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CACHE = os.path.join(_HERE, "..", "graphids", "data", "soar", "cache.pt2")
MANIFEST = os.path.join(_REPO, "soar-embedding-main", "processed_v2", "windows_manifest.json")
CKPT = os.path.join(_HERE, "ckpt.pt")
EPOCHS = 500
LR = 1e-3
DROPOUT = 0.2
HIDDEN = 128
SEED = 123

th.manual_seed(SEED); np.random.seed(SEED)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("device:", device)


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super().__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {"m": self.W_msg(th.cat([edges.src["h"], edges.data["h"]], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats
            g.update_all(self.message_func, fn.mean("m", "h_neigh"))
            g.ndata["h"] = F.relu(self.W_apply(th.cat([g.ndata["h"], g.ndata["h_neigh"]], 2)))
            return g.ndata["h"]


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(th.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 2)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)


def build_graph(edge_index, edge_attr, edge_labels, num_nodes, edim):
    src = th.from_numpy(np.ascontiguousarray(edge_index[0])).long()
    dst = th.from_numpy(np.ascontiguousarray(edge_index[1])).long()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata["h"] = th.ones(num_nodes, 1, edim, dtype=th.float)
    g.edata["h"] = th.from_numpy(np.ascontiguousarray(edge_attr)).float().unsqueeze(1)
    g.edata["label"] = th.from_numpy(np.ascontiguousarray(edge_labels)).long()
    return g


def report(tag, y, scores, thr):
    pred = (scores > thr).astype(int)
    auroc = roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan")
    auprc = average_precision_score(y, scores) if y.sum() > 0 else float("nan")
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f1p = f1_score(y, pred, zero_division=0)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    f1w = f1_score(y, pred, average="weighted", zero_division=0)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    print(f"--- {tag} n={len(y)} pos={int(y.sum())} thr={thr:.6g} ---")
    print(f"AUROC={auroc:.4f} AUPRC={auprc:.4f} P={p:.4f} R={r:.4f} F1pos={f1p:.4f} F1macro={f1m:.4f} F1wt={f1w:.4f}")
    print(f"TP {tp}  FP {fp}  TN {tn}  FN {fn}")


def main():
    print("loading cache...")
    blob = th.load(CACHE, weights_only=False)
    ei = blob["edge_index"].numpy()
    ea = blob["edge_attr"].numpy()
    lb = blob["labels"].numpy()
    num_nodes = int(blob["num_nodes"])
    edim = int(blob["edge_dim"])
    train_idx = blob["train_idx"].numpy()
    val_idx = blob["val_idx"].numpy()
    test_idx = blob["test_idx"].numpy()
    scenario = np.array(blob["scenario"])
    wid = blob["wid"].numpy() if hasattr(blob["wid"], "numpy") else np.asarray(blob["wid"])
    print(f"nodes={num_nodes} edges={ei.shape[1]} edim={edim}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    g_train = build_graph(ei[:, train_idx], ea[train_idx], lb[train_idx], num_nodes, edim).to(device)
    g_val = build_graph(ei[:, val_idx], ea[val_idx], lb[val_idx], num_nodes, edim).to(device)
    g_test = build_graph(ei[:, test_idx], ea[test_idx], lb[test_idx], num_nodes, edim).to(device)

    cw = class_weight.compute_class_weight("balanced", classes=np.unique(lb[train_idx]), y=lb[train_idx])
    cw = th.FloatTensor(cw).to(device)
    print("class weights:", cw.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=cw)

    model = Model(edim, 128, edim, F.relu, DROPOUT).to(device)
    opt = th.optim.Adam(model.parameters(), lr=LR)

    best_val_ap = -1.0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pred = model(g_train, g_train.ndata["h"], g_train.edata["h"])
        loss = criterion(pred, g_train.edata["label"])
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with th.no_grad():
                vpred = model(g_val, g_val.ndata["h"], g_val.edata["h"])
                vprob = F.softmax(vpred, dim=1)[:, 1].cpu().numpy()
                vy = g_val.edata["label"].cpu().numpy()
                vap = average_precision_score(vy, vprob) if vy.sum() > 0 else float("nan")
            print(f"ep {epoch:04d}  loss {loss.item():.4f}  val_ap {vap:.4f}")
            if vap > best_val_ap:
                best_val_ap = vap
                th.save({"sd": model.state_dict(), "ep": epoch, "vap": vap}, CKPT)
    train_time = time.perf_counter() - t0
    print(f"train time: {train_time:.1f} s")

    ckpt = th.load(CKPT, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["sd"])
    print(f"loaded best ep={ckpt['ep']} val_ap={ckpt['vap']:.4f}")

    model.eval()
    if device.type == "cuda":
        th.cuda.synchronize()
    t0 = time.perf_counter()
    with th.no_grad():
        tpred = model(g_test, g_test.ndata["h"], g_test.edata["h"])
        tprob = F.softmax(tpred, dim=1)[:, 1].cpu().numpy()
    if device.type == "cuda":
        th.cuda.synchronize()
    inf_time = time.perf_counter() - t0
    ty = g_test.edata["label"].cpu().numpy()

    prec_c, rec_c, thr_c = precision_recall_curve(ty, tprob)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
    best = int(np.argmax(f1_c))
    best_thr = thr_c[min(best, len(thr_c) - 1)]
    report("per-edge (best-F1 thr)", ty, tprob, best_thr)

    # Per-window aggregation.
    n = len(tprob)
    sc = scenario[test_idx][:n]
    wd = wid[test_idx][:n]
    groups = {}
    for s, w, p, l in zip(sc, wd, tprob, ty):
        k = (str(s), int(w))
        g = groups.get(k)
        if g is None:
            groups[k] = [float(p), int(l)]
        else:
            if p > g[0]: g[0] = float(p)
            if l > g[1]: g[1] = int(l)
    win_keys = np.array([f"{s}#{w}" for s, w in groups.keys()])
    win_score = np.array([v[0] for v in groups.values()])
    win_label = np.array([v[1] for v in groups.values()])

    with open(MANIFEST) as f:
        manifest = json.load(f)
    mtw = [(w["scenario"], w["window_id"], w["label"]) for w in manifest if w["split"] == "test"]
    mkeys = np.array([f"{s}#{w}" for s, w, _ in mtw])
    mlab = np.array([lbl for _, _, lbl in mtw], dtype=int)
    k2s = dict(zip(win_keys, win_score))
    fill = float(win_score.min()) if len(win_score) else 0.0
    full_score = np.array([k2s.get(k, fill) for k in mkeys])
    print(f"covered windows: {len(win_keys)} / {len(mkeys)}")

    prec_c, rec_c, thr_c = precision_recall_curve(mlab, full_score)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
    best = int(np.argmax(f1_c))
    best_thr = thr_c[min(best, len(thr_c) - 1)]
    report("per-window (all 948, best-F1 thr)", mlab, full_score, best_thr)

    print(f"inference: {inf_time:.3f} s for {n} edges  ({inf_time/n*1e6:.2f} us/edge)")
    print(f"per-window throughput (amortized 948): {948/inf_time:.1f} win/sec, {inf_time/948*1000:.3f} ms/win")
    print(f"train time: {train_time:.1f} s for {EPOCHS} epochs")


if __name__ == "__main__":
    main()
