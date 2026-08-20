import time
import numpy as np
import torch
import json
import sys

sys.path.insert(0, ".")
from graph_model import GraphAnomalyModel

CKPT = "graph_output/best_graph_homogt128_4L_s1.pt"
HOMO = True
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

graphs = torch.load("../processed_v2/graph_windows.pt", weights_only=False)
metadata = (
    list({k for g in graphs[:500] for k in g.node_types}),
    list({k for g in graphs[:500] for k in g.edge_types}),
)
with open("../processed_v2/graph_meta.json") as f:
    meta = json.load(f)

test_graphs = [g for g in graphs if g.split == "test"]
print(f"test windows: {len(test_graphs)}")

model = (
    GraphAnomalyModel(
        HIDDEN_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        n_sigs=meta["n_sigs"],
        n_processes=meta["n_processes"],
        metadata=metadata,
        pool="max",
        homo=HOMO,
    )
    .to(device)
    .eval()
)
model.load_state_dict(torch.load(CKPT, weights_only=False, map_location=device))
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# warmup
with torch.no_grad():
    for g in test_graphs[:10]:
        _ = model.classify(g.to(device))

# per-window timing
times = []
with torch.no_grad():
    for g in test_graphs:
        g = g.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.classify(g)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

t = np.asarray(times)
print(f"per-window mean   {t.mean()*1000:.3f} ms")
print(f"per-window median {np.median(t)*1000:.3f} ms")
print(f"per-window p95    {np.percentile(t,95)*1000:.3f} ms")
print(f"throughput        {1.0/t.mean():.1f} win/sec")
print(f"total             {t.sum()*1000:.1f} ms over {len(t)} windows")
