import os
import pickle as pkl
import dgl
import torch
from tqdm import tqdm

# Resolve paths relative to repo root so this runs from baselines/magic_apt/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
SOAR_CACHE = os.path.join(_HERE, '..', 'data', 'soar')
GRAPHS_PT = os.path.join(_REPO, 'processed_v2', 'graph_windows.pt')

EDGE_RELS = [
    ('alert', 'has_src', 'ip'),
    ('alert', 'has_dst', 'ip'),
    ('alert', 'has_process', 'process'),
    ('alert', 'has_domain', 'domain'),
    ('alert', 'next', 'alert'),
    ('ip', 'flow', 'ip'),
    ('ip', 'rev_src', 'alert'),
    ('ip', 'rev_dst', 'alert'),
    ('domain', 'rev_domain', 'alert'),
    ('process', 'rev_proc', 'alert'),
]


def _build_vocab(graphs):
    n_sigs, n_procs = 0, 0
    for g in graphs:
        if 'alert' in g.node_types and g['alert'].num_nodes > 0:
            n_sigs = max(n_sigs, int(g['alert'].sig_ids.max()) + 1)
        if 'process' in g.node_types and g['process'].num_nodes > 0:
            n_procs = max(n_procs, int(g['process'].x.squeeze(-1).max()) + 1)
    return {
        'n_sigs': n_sigs,
        'ip_id': n_sigs,
        'proc_base': n_sigs + 1,
        'domain_id': n_sigs + 1 + n_procs,
        'n_node_types': n_sigs + 1 + n_procs + 1,
        'n_edge_types': len(EDGE_RELS),
    }


def _flatten(g, vocab):
    counts = {nt: (g[nt].num_nodes if nt in g.node_types else 0)
              for nt in ('alert', 'ip', 'process', 'domain')}
    offsets, cum = {}, 0
    for nt in ('alert', 'ip', 'process', 'domain'):
        offsets[nt] = cum
        cum += counts[nt]
    n_total = cum

    types = torch.zeros(n_total, dtype=torch.long)
    if counts['alert']:
        types[offsets['alert']:offsets['alert'] + counts['alert']] = g['alert'].sig_ids.long()
    if counts['ip']:
        types[offsets['ip']:offsets['ip'] + counts['ip']] = vocab['ip_id']
    if counts['process']:
        pids = g['process'].x.squeeze(-1).long()
        types[offsets['process']:offsets['process'] + counts['process']] = vocab['proc_base'] + pids
    if counts['domain']:
        types[offsets['domain']:offsets['domain'] + counts['domain']] = vocab['domain_id']

    src_chunks, dst_chunks, etype_chunks = [], [], []
    for etype_id, key in enumerate(EDGE_RELS):
        s, _, d = key
        if key not in g.edge_types or not hasattr(g[key], 'edge_index'):
            continue
        ei = g[key].edge_index
        if ei.numel() == 0:
            continue
        src_chunks.append(ei[0].long() + offsets[s])
        dst_chunks.append(ei[1].long() + offsets[d])
        etype_chunks.append(torch.full((ei.size(1),), etype_id, dtype=torch.long))

    if src_chunks:
        src = torch.cat(src_chunks)
        dst = torch.cat(dst_chunks)
        etypes = torch.cat(etype_chunks)
    else:
        src = dst = etypes = torch.zeros(0, dtype=torch.long)

    dg = dgl.graph((src, dst), num_nodes=n_total)
    dg.ndata['type'] = types
    dg.edata['type'] = etypes
    return dg


def _build_cache():
    os.makedirs(SOAR_CACHE, exist_ok=True)
    pkl_path = os.path.join(SOAR_CACHE, 'graphs.pkl')
    if os.path.exists(pkl_path):
        return pkl_path
    print('Building soar cache from', GRAPHS_PT)
    raw = torch.load(GRAPHS_PT, weights_only=False)
    vocab = _build_vocab(raw)
    items = []
    for g in tqdm(raw):
        dg = _flatten(g, vocab)
        items.append((dg, int(g.y.item()), g.split))
    with open(pkl_path, 'wb') as f:
        pkl.dump((items, vocab), f)
    return pkl_path


def load_soar_dataset():
    _build_cache()
    with open(os.path.join(SOAR_CACHE, 'graphs.pkl'), 'rb') as f:
        items, vocab = pkl.load(f)
    graphs = [(dg, lbl) for dg, lbl, _ in items]
    splits = [sp for _, _, sp in items]
    train_index = [i for i, (_, lbl, sp) in enumerate(items) if sp == 'train' and lbl == 0]
    full_index = list(range(len(items)))
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(
        len(items), vocab['n_node_types'], vocab['n_edge_types']))
    return {
        'dataset': graphs,
        'splits': splits,
        'train_index': train_index,
        'full_index': full_index,
        'n_feat': vocab['n_node_types'],
        'e_feat': vocab['n_edge_types'],
    }
