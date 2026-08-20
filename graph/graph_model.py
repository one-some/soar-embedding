import torch
import torch.nn as nn
from torch_geometric.nn import (
    HGTConv,
    TransformerConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.data import HeteroData


class HGTEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        n_sigs=500,
        n_processes=50,
        metadata=None,
        text_emb_dim=384,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sig_embed = nn.Embedding(n_sigs, hidden_dim, padding_idx=0)
        self.alert_pos = nn.Linear(1, hidden_dim)
        self.alert_text = nn.Linear(text_emb_dim, hidden_dim)

        # 5 within-window behavioral + 3 cross-window history features
        self.ip_proj = nn.Linear(8, hidden_dim)

        self.proc_embed = nn.Embedding(n_processes, hidden_dim, padding_idx=0)

        # freq, ratio, subdomain entropy, max label len, hex ratio, n_labels
        self.domain_proj = nn.Linear(6, hidden_dim)

        self.convs = nn.ModuleList(
            [
                HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def _build_node_features(self, data):
        x_dict = {}

        if "alert" in data.node_types and data["alert"].num_nodes > 0:
            sig_h = self.sig_embed(data["alert"].sig_ids)
            pos_h = self.alert_pos(data["alert"].x[:, 1:2])
            text_h = self.alert_text(data["alert"].text_emb)
            x_dict["alert"] = sig_h + pos_h + text_h

        if "ip" in data.node_types and data["ip"].num_nodes > 0:
            x_dict["ip"] = self.ip_proj(data["ip"].x)

        if "process" in data.node_types and data["process"].num_nodes > 0:
            proc_ids = data["process"].x.squeeze(-1).long()
            x_dict["process"] = self.proc_embed(proc_ids)

        if "domain" in data.node_types and data["domain"].num_nodes > 0:
            x_dict["domain"] = self.domain_proj(data["domain"].x)

        edge_index_dict = {}
        for key in data.edge_types:
            if not hasattr(data[key], "edge_index"):
                continue
            src_type, _, dst_type = key
            if src_type in x_dict and dst_type in x_dict:
                edge_index_dict[key] = data[key].edge_index

        if edge_index_dict:
            for conv, norm in zip(self.convs, self.norms):
                x_dict_new = conv(x_dict, edge_index_dict)
                x_dict = {
                    k: norm(x_dict_new[k])
                    + x_dict.get(k, torch.zeros_like(x_dict_new[k]))
                    for k in x_dict_new
                }

        return x_dict

    def forward_nodes(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._build_node_features(data)
        if "alert" in x_dict:
            return x_dict["alert"]
        return torch.cat([v for v in x_dict.values()], dim=0)


class HomoGTEncoder(nn.Module):
    # ablation of HGTEncoder: same per-type input projections, then collapse to a
    # single node type and edge type and run a homogeneous TransformerConv stack

    NODE_ORDER = ("alert", "ip", "process", "domain")

    def __init__(
        self,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        n_sigs=500,
        n_processes=50,
        metadata=None,
        text_emb_dim=384,
        disable_text=False,
        disable_ip=False,
        disable_proc=False,
        disable_domain=False,
        disable_alert_pos=False,
        disable_sig=False,
        random_text=False,
        use_sage=False,
    ):
        super().__init__()
        del metadata
        self.hidden_dim = hidden_dim
        self.disable_text = disable_text
        self.disable_ip = disable_ip
        self.disable_proc = disable_proc
        self.disable_domain = disable_domain
        self.disable_alert_pos = disable_alert_pos
        self.disable_sig = disable_sig
        self.random_text = random_text
        self.sig_embed = nn.Embedding(n_sigs, hidden_dim, padding_idx=0)
        self.alert_pos = nn.Linear(1, hidden_dim)
        self.alert_text = nn.Linear(text_emb_dim, hidden_dim)
        self.ip_proj = nn.Linear(8, hidden_dim)
        self.proc_embed = nn.Embedding(n_processes, hidden_dim, padding_idx=0)
        self.domain_proj = nn.Linear(6, hidden_dim)
        if random_text:
            rand = torch.randn(n_sigs, text_emb_dim)
            self.register_buffer("random_text_table", rand)

        assert hidden_dim % num_heads == 0
        head_dim = hidden_dim // num_heads
        if use_sage:
            self.convs = nn.ModuleList(
                [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    TransformerConv(hidden_dim, head_dim, heads=num_heads)
                    for _ in range(num_layers)
                ]
            )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def _project(self, data):
        x_dict = {}
        if "alert" in data.node_types and data["alert"].num_nodes > 0:
            sig_h = (
                torch.zeros(
                    data["alert"].num_nodes,
                    self.hidden_dim,
                    device=data["alert"].sig_ids.device,
                )
                if self.disable_sig
                else self.sig_embed(data["alert"].sig_ids)
            )
            pos_h = (
                0 if self.disable_alert_pos else self.alert_pos(data["alert"].x[:, 1:2])
            )
            if self.disable_text:
                text_h = 0
            elif self.random_text:
                text_h = self.alert_text(self.random_text_table[data["alert"].sig_ids])
            else:
                text_h = self.alert_text(data["alert"].text_emb)
            x_dict["alert"] = sig_h + pos_h + text_h
        if "ip" in data.node_types and data["ip"].num_nodes > 0 and not self.disable_ip:
            x_dict["ip"] = self.ip_proj(data["ip"].x)
        if (
            "process" in data.node_types
            and data["process"].num_nodes > 0
            and not self.disable_proc
        ):
            x_dict["process"] = self.proc_embed(data["process"].x.squeeze(-1).long())
        if (
            "domain" in data.node_types
            and data["domain"].num_nodes > 0
            and not self.disable_domain
        ):
            x_dict["domain"] = self.domain_proj(data["domain"].x)
        return x_dict

    def _flatten(self, x_dict, data):
        order = [t for t in self.NODE_ORDER if t in x_dict]
        offsets, cum, chunks = {}, 0, []
        for t in order:
            offsets[t] = cum
            chunks.append(x_dict[t])
            cum += x_dict[t].size(0)
        x = torch.cat(chunks, dim=0)

        edge_chunks = []
        for key in data.edge_types:
            if not hasattr(data[key], "edge_index"):
                continue
            src_t, _, dst_t = key
            if src_t not in offsets or dst_t not in offsets:
                continue
            ei = data[key].edge_index.clone()
            ei[0] = ei[0] + offsets[src_t]
            ei[1] = ei[1] + offsets[dst_t]
            edge_chunks.append(ei)
        if edge_chunks:
            edge_index = torch.cat(edge_chunks, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        return x, edge_index, offsets

    def _build_node_features(self, data):
        x_dict = self._project(data)
        if not x_dict:
            return x_dict
        x, edge_index, offsets = self._flatten(x_dict, data)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            x = norm(h) + x
        out = {}
        for t, off in offsets.items():
            n = x_dict[t].size(0)
            out[t] = x[off : off + n]
        return out

    def forward_nodes(self, data):
        x_dict = self._build_node_features(data)
        if "alert" in x_dict:
            return x_dict["alert"]
        return torch.cat(list(x_dict.values()), dim=0)


class GraphAnomalyModel(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        n_sigs=500,
        n_processes=50,
        metadata=None,
        text_emb_dim=384,
        pool="max",
        homo=False,
        **enc_kwargs
    ):
        super().__init__()
        self.pool = pool
        enc_cls = HomoGTEncoder if homo else HGTEncoder
        if homo:
            self.encoder = enc_cls(
                hidden_dim,
                num_heads,
                num_layers,
                n_sigs,
                n_processes,
                metadata,
                text_emb_dim,
                **enc_kwargs
            )
        else:
            self.encoder = enc_cls(
                hidden_dim,
                num_heads,
                num_layers,
                n_sigs,
                n_processes,
                metadata,
                text_emb_dim,
            )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def classify_nodes(self, data) -> torch.Tensor:
        node_embs = self.encoder.forward_nodes(data)
        return self.node_classifier(node_embs).squeeze(-1)

    def classify(self, data) -> torch.Tensor:
        # max-pool over per-alert logits so one ATTACK alert can flag the window
        node_logits = self.classify_nodes(data)
        batch = (
            data["alert"].batch
            if hasattr(data["alert"], "batch")
            else torch.zeros(
                node_logits.size(0), dtype=torch.long, device=node_logits.device
            )
        )
        pool_fn = global_mean_pool if self.pool == "mean" else global_max_pool
        return pool_fn(node_logits.unsqueeze(-1), batch).squeeze(-1)
