"""
LSTM-based sequential anomaly detector.

Maintains a per-user rolling history of resource accesses and uses an LSTM
language-model to predict the next resource.  A low predicted probability for
the *actual* next resource implies anomalous behaviour.

The module exposes ``LSTMSequentialScorer`` which wraps model loading,
history management, and inference behind a single ``.score()`` call.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyper-parameters / defaults
# ---------------------------------------------------------------------------
DEFAULT_SEQ_LEN = 20
DEFAULT_EMBED_DIM = 32
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------
class LSTMActionPredictor(nn.Module):
    """Next-resource predictor trained on sequences of encoded resource IDs."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor of shape (batch, seq_len)

        Returns
        -------
        logits : Tensor of shape (batch, vocab_size)
        """
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over the resource vocabulary."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# High-level scorer (loads model, manages history, runs inference)
# ---------------------------------------------------------------------------
class LSTMSequentialScorer:
    """
    Wrapper that loads a serialised LSTM model and its vocabulary mapping,
    keeps an in-memory per-user history of resource accesses, and exposes a
    ``score(user_id, resource_id)`` method that returns a risk float in [0, 1].
    """

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        seq_len: int = DEFAULT_SEQ_LEN,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.user_history: dict[str, deque] = {}

        with open(vocab_path, "r") as f:
            meta = json.load(f)
        self.resource_to_idx: dict[str, int] = meta["resource_to_idx"]
        self.idx_to_resource: dict[int, str] = {
            int(v): k for k, v in self.resource_to_idx.items()
        }
        self.vocab_size: int = meta["vocab_size"]

        self.model = LSTMActionPredictor(
            vocab_size=self.vocab_size,
            embed_dim=meta.get("embed_dim", DEFAULT_EMBED_DIM),
            hidden_dim=meta.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            num_layers=meta.get("num_layers", DEFAULT_NUM_LAYERS),
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, resource_id: str) -> int:
        return self.resource_to_idx.get(resource_id, 0)

    def _get_sequence(self, user_id: str) -> list[int]:
        history = self.user_history.get(user_id, deque(maxlen=self.seq_len))
        seq = list(history)
        if len(seq) < self.seq_len:
            seq = [0] * (self.seq_len - len(seq)) + seq
        return seq

    def score(self, user_id: str, resource_id: str) -> float:
        """
        Return a risk score in [0, 1].

        * Appends ``resource_id`` to the user's rolling history.
        * If insufficient history exists (< 3 events), returns a neutral 0.5.
        * Otherwise predicts the probability the model assigns to the actual
          next resource.  Risk = 1 - P(actual_resource).
        """
        encoded = self._encode(resource_id)

        if user_id not in self.user_history:
            self.user_history[user_id] = deque(maxlen=self.seq_len)

        history_len = len(self.user_history[user_id])

        if history_len < 3:
            self.user_history[user_id].append(encoded)
            return 0.5

        seq = self._get_sequence(user_id)
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        proba = self.model.predict_proba(x)
        predicted_prob = float(proba[0, encoded].item())
        risk = 1.0 - predicted_prob

        self.user_history[user_id].append(encoded)
        return float(np.clip(risk, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Factory — safe loader that returns None when artefacts are missing
# ---------------------------------------------------------------------------
def load_lstm_scorer(
    base_dir: str,
    model_filename: str = "lstm_action_model.pt",
    vocab_filename: str = "lstm_vocab.json",
) -> Optional[LSTMSequentialScorer]:
    """Return an ``LSTMSequentialScorer`` or ``None`` if files are missing."""
    model_path = os.path.join(base_dir, model_filename)
    vocab_path = os.path.join(base_dir, vocab_filename)
    if not os.path.isfile(model_path) or not os.path.isfile(vocab_path):
        return None
    return LSTMSequentialScorer(model_path, vocab_path)
