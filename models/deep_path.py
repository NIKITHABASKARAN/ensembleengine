"""
Deep Path orchestrator.

The "Detective" layer is computationally heavier (LSTM + GNN) and is only
invoked when the Fast Path ensemble score falls in the ambiguous **Medium**
band (0.4 < score < 0.7).  Outside that band the three Fast-Path models
already agree strongly enough to make a decision, so the extra cost is
avoided.

When triggered the Deep Path runs both the sequential anomaly detector
(LSTM) and the relational anomaly detector (GraphSAGE) and returns their
risk scores alongside a blended final score.
"""

from __future__ import annotations

from typing import Optional

DEEP_PATH_LOW = 0.4
DEEP_PATH_HIGH = 0.7

BLEND_WEIGHT_FAST = 0.70
BLEND_WEIGHT_LSTM = 0.15
BLEND_WEIGHT_GNN = 0.15


def should_trigger_deep_path(fast_path_score: float) -> bool:
    """Return ``True`` when the fast-path score is in the Medium band."""
    if fast_path_score is None:
        return False
    return DEEP_PATH_LOW < fast_path_score < DEEP_PATH_HIGH


def run_deep_path(
    fast_path_score: float,
    user_id: Optional[str],
    resource_id: Optional[str],
    device_type: str,
    lstm_scorer,
    gnn_scorer,
    activity_sequence: Optional[list] = None,
) -> dict:
    """
    Conditionally execute the Deep Path models.

    Parameters
    ----------
    fast_path_score : float
        The weighted ensemble score from the Fast Path (0-1).
    user_id : str or None
        Opaque user identifier.  If ``None`` the Deep Path is skipped.
    resource_id : str or None
        Resource the user is attempting to access.  If ``None`` the Deep Path
        is skipped.
    device_type : str
        Device category string (e.g. ``"desktop"``).
    lstm_scorer : LSTMSequentialScorer or None
        Loaded LSTM scorer instance (``None`` when model files are absent).
    gnn_scorer : GNNLinkScorer or None
        Loaded GNN scorer instance (``None`` when model files are absent).
    activity_sequence : list or None
        User's recent activity sequence for behavioral analysis.

    Returns
    -------
    dict with keys:
        triggered : bool
        lstm_risk  : float or None
        gnn_risk   : float or None
        sequence_risk : float or None  (LSTM sequence anomaly score)
        relational_risk : float or None (GNN relational score)
        blended_score : float or None   (only when triggered)
    """
    if (
        not should_trigger_deep_path(fast_path_score)
        or user_id is None
        or resource_id is None
        or lstm_scorer is None
        or gnn_scorer is None
    ):
        return {
            "triggered": False,
            "lstm_risk": None,
            "gnn_risk": None,
            "sequence_risk": None,
            "relational_risk": None,
            "blended_score": None,
        }

    # LSTM sequential analysis - calculate sequence anomaly risk
    sequence_risk = lstm_scorer.score(user_id, resource_id, activity_sequence)
    
    # GNN relational analysis - score user-device-resource relationship
    relational_risk = gnn_scorer.score(user_id, device_type, resource_id)

    blended = (
        BLEND_WEIGHT_FAST * fast_path_score
        + BLEND_WEIGHT_LSTM * sequence_risk
        + BLEND_WEIGHT_GNN * relational_risk
    )

    return {
        "triggered": True,
        "lstm_risk": round(sequence_risk, 4),
        "gnn_risk": round(relational_risk, 4),
        "sequence_risk": round(sequence_risk, 4),
        "relational_risk": round(relational_risk, 4),
        "blended_score": round(blended, 4),
    }
