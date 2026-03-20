"""
GraphSAGE-based link-prediction anomaly detector.

Maintains an in-memory heterogeneous graph of (User, Device, Resource) nodes
and the edges between them.  At inference time, it computes node embeddings
via a 2-layer GraphSAGE encoder and scores the plausibility of the
(user, device) and (user, resource) links with dot-product similarity.

Low link probability  =>  the access pattern is unusual  =>  higher risk.

The module exposes ``GNNLinkScorer`` which wraps model loading, graph
management, and inference behind a single ``.score()`` call.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

# ---------------------------------------------------------------------------
# Hyper-parameters / defaults
# ---------------------------------------------------------------------------
DEFAULT_IN_CHANNELS = 32
DEFAULT_HIDDEN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 32


# ---------------------------------------------------------------------------
# PyTorch Geometric model
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder for homogeneous graph."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GraphSAGEEncoder. "
                "Install it with: pip install torch-geometric"
            )
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """Dot-product link scorer on top of GraphSAGE embeddings."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    @staticmethod
    def predict_link(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((z_src * z_dst).sum(dim=-1))


# ---------------------------------------------------------------------------
# In-memory graph store
# ---------------------------------------------------------------------------
class _GraphStore:
    """
    Lightweight in-memory graph that maps string node IDs to integer indices
    and accumulates edges.  Converts to a PyG ``Data`` object on demand.
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.node_to_idx: dict[str, int] = {}
        self.edges: list[tuple[int, int]] = []
        self._dirty = True
        self._cached_data: Optional[Data] = None

    def _get_or_add(self, node_id: str) -> int:
        if node_id not in self.node_to_idx:
            self.node_to_idx[node_id] = len(self.node_to_idx)
            self._dirty = True
        return self.node_to_idx[node_id]

    def add_edge(self, src: str, dst: str) -> None:
        s = self._get_or_add(src)
        d = self._get_or_add(dst)
        self.edges.append((s, d))
        self.edges.append((d, s))
        self._dirty = True

    def to_pyg(self) -> "Data":
        if self._cached_data is not None and not self._dirty:
            return self._cached_data

        num_nodes = len(self.node_to_idx)
        if num_nodes == 0:
            x = torch.zeros((1, self.feature_dim))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            torch.manual_seed(42)
            x = torch.randn(num_nodes, self.feature_dim)
            if self.edges:
                edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

        self._cached_data = Data(x=x, edge_index=edge_index)
        self._dirty = False
        return self._cached_data

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_to_idx

    def idx(self, node_id: str) -> int:
        return self.node_to_idx[node_id]


# ---------------------------------------------------------------------------
# High-level scorer
# ---------------------------------------------------------------------------
class GNNLinkScorer:
    """
    Wraps the serialised GraphSAGE model, keeps an in-memory graph of
    observed (user, device, resource) relationships, and exposes a
    ``score(user_id, device_type, resource_id)`` method returning risk in [0, 1].
    """

    def __init__(
        self,
        model_path: str,
        meta_path: str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        in_ch = meta.get("in_channels", DEFAULT_IN_CHANNELS)
        hidden_ch = meta.get("hidden_channels", DEFAULT_HIDDEN_CHANNELS)
        out_ch = meta.get("out_channels", DEFAULT_OUT_CHANNELS)

        self.model = LinkPredictor(in_ch, hidden_ch, out_ch)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

        self.graph = _GraphStore(feature_dim=in_ch)

        if "initial_edges" in meta:
            for src, dst in meta["initial_edges"]:
                self.graph.add_edge(src, dst)

    def _ensure_nodes(self, user_id: str, device_type: str, resource_id: str) -> None:
        u_key = f"user:{user_id}"
        d_key = f"device:{device_type}"
        r_key = f"resource:{resource_id}"
        self.graph.add_edge(u_key, d_key)
        self.graph.add_edge(u_key, r_key)
        self.graph.add_edge(d_key, r_key)

    def score(self, user_id: str, device_type: str, resource_id: str) -> float:
        """
        Return a risk score in [0, 1].

        Computes GraphSAGE embeddings for all nodes in the current graph, then
        evaluates the link probability for (user, resource) and (user, device).
        Risk = 1 - mean(link_probabilities).
        """
        u_key = f"user:{user_id}"
        d_key = f"device:{device_type}"
        r_key = f"resource:{resource_id}"

        is_new_user = not self.graph.has_node(u_key)

        self._ensure_nodes(user_id, device_type, resource_id)
        data = self.graph.to_pyg()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        if is_new_user or edge_index.shape[1] < 2:
            return 0.5

        with torch.no_grad():
            z = self.model(x, edge_index)

        u_idx = self.graph.idx(u_key)
        d_idx = self.graph.idx(d_key)
        r_idx = self.graph.idx(r_key)

        link_ud = float(LinkPredictor.predict_link(z[u_idx], z[d_idx]).item())
        link_ur = float(LinkPredictor.predict_link(z[u_idx], z[r_idx]).item())

        mean_link = (link_ud + link_ur) / 2.0
        risk = 1.0 - mean_link
        return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_gnn_scorer(
    base_dir: str,
    model_filename: str = "graphsage_link_model.pt",
    meta_filename: str = "graphsage_meta.json",
) -> Optional[GNNLinkScorer]:
    """Return a ``GNNLinkScorer`` or ``None`` if artefacts are missing."""
    if not _HAS_PYG:
        return None
    model_path = os.path.join(base_dir, model_filename)
    meta_path = os.path.join(base_dir, meta_filename)
    if not os.path.isfile(model_path) or not os.path.isfile(meta_path):
        return None
    return GNNLinkScorer(model_path, meta_path)
