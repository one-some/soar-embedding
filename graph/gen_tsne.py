import json
import sys

sys.path.insert(0, ".")
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from graph_model import GraphAnomalyModel

CKPT = "graph_output/best_graph_homogt128_4L_s1.pt"
OUT_PDF = "tsne_alerts.pdf"
HIDDEN, LAYERS, HEADS = 128, 4, 4
N_PER_CLASS = 3000
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load("../processed_v2/graph_windows.pt", weights_only=False)
with open("../processed_v2/graph_meta.json") as f:
    meta = json.load(f)
md = (
    list({k for g in graphs[:500] for k in g.node_types}),
    list({k for g in graphs[:500] for k in g.edge_types}),
)

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
m.load_state_dict(torch.load(CKPT, weights_only=False, map_location=device))

# pull per-alert embeddings + logits over the test split
test_graphs = [g for g in graphs if g.split == "test"]
print(f"test windows: {len(test_graphs)}")
loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

all_emb, all_logit, all_label = [], [], []
with torch.no_grad():
    for b in loader:
        b = b.to(device)
        # [N_alerts, hidden]
        emb = m.encoder.forward_nodes(b)
        # [N_alerts]
        logit = m.node_classifier(emb).squeeze(-1)
        all_emb.append(emb.cpu().numpy())
        all_logit.append(logit.cpu().numpy())
        all_label.append(b["alert"].y.cpu().numpy())

emb = np.concatenate(all_emb, axis=0)
logit = np.concatenate(all_logit, axis=0)
label = np.concatenate(all_label, axis=0)
print(
    f"per-alert embeddings: shape={emb.shape}  attack={int(label.sum())}/{len(label)}"
)

# balanced sample (3000 attack + 3000 benign)
rng = np.random.default_rng(SEED)
att_idx = np.where(label == 1)[0]
ben_idx = np.where(label == 0)[0]
n_att = min(N_PER_CLASS, len(att_idx))
n_ben = min(N_PER_CLASS, len(ben_idx))
sel_att = rng.choice(att_idx, n_att, replace=False)
sel_ben = rng.choice(ben_idx, n_ben, replace=False)
sel = np.concatenate([sel_att, sel_ben])
rng.shuffle(sel)
emb_s = emb[sel]
logit_s = logit[sel]
label_s = label[sel]
print(
    f"balanced sample: {len(emb_s)} alerts ({(label_s==1).sum()} attack, {(label_s==0).sum()} benign)"
)

# PCA init, perplexity 30, auto learning rate, 1000 iters
print(
    "running t-SNE (standard config: perplexity=30, init=pca, learning_rate=auto, max_iter=1000)..."
)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=SEED,
    n_jobs=-1,
)
xy = tsne.fit_transform(emb_s)

# color by classifier logit (diverging cmap)
fig, ax = plt.subplots(figsize=(5.5, 5.0))
vmax = float(np.max(np.abs(logit_s)))
sc = ax.scatter(
    xy[:, 0],
    xy[:, 1],
    c=logit_s,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    s=6,
    alpha=0.7,
    linewidths=0,
)
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.set_xticks([])
ax.set_yticks([])
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label("per-alert classifier logit\n(<0: benign, >0: attack)")
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
plt.savefig(OUT_PDF.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"wrote {OUT_PDF} and PNG")

# quick separability stats (KNN purity within the sample)
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=11).fit(xy)
_, idx = nbrs.kneighbors(xy)
neigh_labels = label_s[idx[:, 1:]]
purity = (neigh_labels == label_s[:, None]).mean()
print(f"10-NN label purity in t-SNE space: {purity:.3f}")
