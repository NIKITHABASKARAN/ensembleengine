#!/usr/bin/env python
"""
Training script for the Deep Path behavioural models.

Generates synthetic user-resource-device interaction data, trains:

1. **LSTM sequential anomaly model** — predicts the next resource a user
   will access given their recent history.
2. **GraphSAGE link-prediction model** — scores how plausible a
   (user, device, resource) triple is based on a co-occurrence graph.

Outputs four artefact files to the project root:

    lstm_action_model.pt   – LSTM state dict
    lstm_vocab.json        – resource vocabulary + hyper-parameters
    graphsage_link_model.pt – GraphSAGE state dict
    graphsage_meta.json    – graph meta-data + hyper-parameters

Run
---
    python train_deep_models.py
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.lstm_sequential import LSTMActionPredictor

# ---------------------------------------------------------------------------
# Check for optional torch_geometric dependency
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data
    from models.gnn_link_predictor import LinkPredictor

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    print(
        "[WARN] torch_geometric not installed — GraphSAGE training will be skipped.\n"
        "       Install with:  pip install torch-geometric"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================================================
#  Synthetic data generation
# ===========================================================================
NUM_USERS = 500
NUM_RESOURCES = 50
NUM_DEVICES = 10
SEQUENCES_PER_USER = 8
SEQ_LEN = 20
ANOMALY_RATE = 0.15

RESOURCES = [f"resource_{i}" for i in range(1, NUM_RESOURCES + 1)]
DEVICES = [f"device_{i}" for i in range(1, NUM_DEVICES + 1)]
USERS = [f"user_{i}" for i in range(1, NUM_USERS + 1)]


def _build_user_profiles() -> dict:
    """
    Each user has a *preferred* subset of resources (3-8) and devices (1-3).
    Normal sequences are drawn from these subsets; anomalous sequences inject
    random resources/devices outside the profile.
    """
    profiles: dict[str, dict] = {}
    for uid in USERS:
        n_res = random.randint(3, 8)
        n_dev = random.randint(1, 3)
        profiles[uid] = {
            "resources": random.sample(RESOURCES, n_res),
            "devices": random.sample(DEVICES, n_dev),
        }
    return profiles


def _generate_sequences(profiles: dict) -> list[dict]:
    """
    Return a list of dicts, each with:
        user_id    : str
        sequence   : list[str]   (resource IDs of length SEQ_LEN+1)
        device     : str
        is_anomaly : bool
    The last element of ``sequence`` is the *target* (next resource).
    """
    records: list[dict] = []
    for uid, prof in profiles.items():
        for _ in range(SEQUENCES_PER_USER):
            is_anomaly = random.random() < ANOMALY_RATE

            if is_anomaly:
                normal_len = random.randint(SEQ_LEN // 2, SEQ_LEN - 1)
                seq = [random.choice(prof["resources"]) for _ in range(normal_len)]
                anomalous_len = SEQ_LEN + 1 - normal_len
                outside = [r for r in RESOURCES if r not in prof["resources"]]
                if not outside:
                    outside = RESOURCES
                seq += [random.choice(outside) for _ in range(anomalous_len)]
                device = random.choice(
                    [d for d in DEVICES if d not in prof["devices"]] or DEVICES
                )
            else:
                seq = [random.choice(prof["resources"]) for _ in range(SEQ_LEN + 1)]
                device = random.choice(prof["devices"])

            records.append(
                {
                    "user_id": uid,
                    "sequence": seq,
                    "device": device,
                    "is_anomaly": is_anomaly,
                }
            )
    random.shuffle(records)
    return records


# ===========================================================================
#  LSTM training
# ===========================================================================
class SequenceDataset(Dataset):
    """Yields (input_seq, target_idx) pairs."""

    def __init__(self, records: list[dict], resource_to_idx: dict[str, int]):
        self.inputs: list[list[int]] = []
        self.targets: list[int] = []
        for rec in records:
            encoded = [resource_to_idx.get(r, 0) for r in rec["sequence"]]
            self.inputs.append(encoded[:-1])
            self.targets.append(encoded[-1])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


def train_lstm(records: list[dict]) -> None:
    """Train the LSTM action predictor and save artefacts."""
    print("=" * 60)
    print("  LSTM Sequential Anomaly Model — Training")
    print("=" * 60)

    resource_to_idx: dict[str, int] = {"<PAD>": 0}
    for r in RESOURCES:
        resource_to_idx[r] = len(resource_to_idx)
    vocab_size = len(resource_to_idx)

    embed_dim = 32
    hidden_dim = 64
    num_layers = 2
    lr = 1e-3
    epochs = 20
    batch_size = 64

    dataset = SequenceDataset(records, resource_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMActionPredictor(vocab_size, embed_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total * 100
        print(f"  Epoch {epoch:>3}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.1f}%")

    model_path = os.path.join(BASE_DIR, "lstm_action_model.pt")
    vocab_path = os.path.join(BASE_DIR, "lstm_vocab.json")

    torch.save(model.state_dict(), model_path)

    meta = {
        "resource_to_idx": resource_to_idx,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_len": SEQ_LEN,
    }
    with open(vocab_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved  {model_path}")
    print(f"  Saved  {vocab_path}\n")


# ===========================================================================
#  GraphSAGE training
# ===========================================================================
def _build_graph_data(
    records: list[dict],
) -> tuple["Data", dict[str, int], list[tuple[str, str]]]:
    """
    Build a PyG ``Data`` object from the interaction records.

    Nodes are prefixed strings: ``user:X``, ``device:Y``, ``resource:Z``.
    Edges connect (user, device) and (user, resource) for every record.
    """
    node_set: dict[str, int] = {}
    edge_src: list[int] = []
    edge_dst: list[int] = []
    raw_edges: list[tuple[str, str]] = []

    def _idx(name: str) -> int:
        if name not in node_set:
            node_set[name] = len(node_set)
        return node_set[name]

    for rec in records:
        u = f"user:{rec['user_id']}"
        d = f"device:{rec['device']}"
        for res_name in rec["sequence"]:
            r = f"resource:{res_name}"
            ui, di, ri = _idx(u), _idx(d), _idx(r)

            edge_src += [ui, di, ui, ri, di, ri]
            edge_dst += [di, ui, ri, ui, ri, di]
            raw_edges += [(u, d), (u, r), (d, r)]

    num_nodes = len(node_set)
    feature_dim = 32
    torch.manual_seed(SEED)
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data, node_set, raw_edges


def _negative_sample(
    num_nodes: int, pos_edge_index: torch.Tensor, num_neg: int
) -> torch.Tensor:
    """Simple uniform negative sampling."""
    pos_set = set()
    for i in range(pos_edge_index.size(1)):
        s, d = int(pos_edge_index[0, i]), int(pos_edge_index[1, i])
        pos_set.add((s, d))

    neg_src, neg_dst = [], []
    while len(neg_src) < num_neg:
        s = random.randint(0, num_nodes - 1)
        d = random.randint(0, num_nodes - 1)
        if s != d and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)
    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def train_graphsage(records: list[dict]) -> None:
    """Train the GraphSAGE link predictor and save artefacts."""
    if not _HAS_PYG:
        print("  [SKIP] GraphSAGE training — torch_geometric is not installed.\n")
        return

    print("=" * 60)
    print("  GraphSAGE Link Predictor — Training")
    print("=" * 60)

    data, node_set, raw_edges = _build_graph_data(records)

    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    lr = 1e-3
    epochs = 30

    model = LinkPredictor(in_channels, hidden_channels, out_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    num_pos = data.edge_index.size(1) // 2
    num_neg = num_pos
    neg_edge_index = _negative_sample(data.x.size(0), data.edge_index, num_neg)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)

        pos_src = data.edge_index[0, :num_pos]
        pos_dst = data.edge_index[1, :num_pos]
        pos_scores = LinkPredictor.predict_link(z[pos_src], z[pos_dst])

        neg_scores = LinkPredictor.predict_link(
            z[neg_edge_index[0]], z[neg_edge_index[1]]
        )

        pos_labels = torch.ones(pos_scores.size(0))
        neg_labels = torch.zeros(neg_scores.size(0))
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (scores > 0.5).float()
            acc = (preds == labels).float().mean().item() * 100

        print(f"  Epoch {epoch:>3}/{epochs}  |  Loss: {loss.item():.4f}  |  Acc: {acc:.1f}%")

    model_path = os.path.join(BASE_DIR, "graphsage_link_model.pt")
    meta_path = os.path.join(BASE_DIR, "graphsage_meta.json")

    torch.save(model.state_dict(), model_path)

    sample_edges = list({(s, d) for s, d in raw_edges})
    if len(sample_edges) > 2000:
        sample_edges = random.sample(sample_edges, 2000)

    meta = {
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "num_nodes": data.x.size(0),
        "num_edges": data.edge_index.size(1),
        "initial_edges": sample_edges,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved  {model_path}")
    print(f"  Saved  {meta_path}\n")


# ===========================================================================
#  Main
# ===========================================================================
def main() -> None:
    print("\nGenerating synthetic interaction data …")
    profiles = _build_user_profiles()
    records = _generate_sequences(profiles)
    print(
        f"  {len(records)} sequences  |  {NUM_USERS} users  |  "
        f"{NUM_RESOURCES} resources  |  {NUM_DEVICES} devices\n"
    )

    train_lstm(records)
    train_graphsage(records)

    print("Training complete.")


if __name__ == "__main__":
    main()
