import argparse
import time
import torch
import warnings
import numpy as np
from utils.loaddata import load_batch_level_dataset, transform_graph
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
from utils.config import build_args

warnings.filterwarnings("ignore")

args = build_args()
device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
args.num_hidden = 256
args.num_layers = 3
set_random_seed(0)

data = load_batch_level_dataset(args.dataset)
args.n_dim = data["n_feat"]
args.e_dim = data["e_feat"]
model = build_model(args)
model.load_state_dict(torch.load(f"./checkpoints/checkpoint-{args.dataset}.pt", map_location=device))
model = model.to(device).eval()
pooler = Pooling("mean")

graphs = data["dataset"]
splits = data["splits"]
test_idxs = [i for i, s in enumerate(splits) if s == "test"]
n_test = len(test_idxs)
print(f"test windows: {n_test}")

# Warmup.
with torch.no_grad():
    for i in test_idxs[:5]:
        g = transform_graph(graphs[i][0], args.n_dim, args.e_dim).to(device)
        _ = pooler(g, model.embed(g))

times = []
with torch.no_grad():
    for i in test_idxs:
        g = transform_graph(graphs[i][0], args.n_dim, args.e_dim).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = model.embed(g)
        out = pooler(g, emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

t = np.asarray(times)
print(f"per-window: mean {t.mean()*1000:.3f} ms  median {np.median(t)*1000:.3f} ms  p95 {np.percentile(t,95)*1000:.3f} ms")
print(f"throughput: {1.0/t.mean():.1f} win/sec  ({n_test} windows in {t.sum():.2f} s)")
