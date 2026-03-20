"""
Trust Entropy — quantify how much the models *disagree*.

Uses **Shannon Entropy** over the normalised vector of model risk
probabilities.  When all models output the same probability the
normalised entropy is 0 ("Low" uncertainty — full agreement).  When
every model outputs a wildly different value the normalised entropy
approaches 1 ("High" uncertainty — strong disagreement).

Formula
-------
Given *n* model risk probabilities  p₁, p₂, … , pₙ  (each in [0, 1]):

1. Normalise into a discrete distribution:
       qᵢ = pᵢ / Σpⱼ

2. Compute raw Shannon Entropy:
       H(X) = −Σ qᵢ · log₂(qᵢ)

3. Normalise to [0, 1] so the result is independent of model count:
       H_norm = H(X) / log₂(n)

Uncertainty levels
------------------
    H_norm < 0.3   →  "Low"     (models strongly agree)
    H_norm < 0.7   →  "Medium"  (moderate disagreement)
    H_norm ≥ 0.7   →  "High"    (models strongly disagree)
"""

from __future__ import annotations

import numpy as np


def compute_trust_entropy(
    probabilities: list[float],
) -> tuple[float, str]:
    """
    Compute normalised Shannon Entropy over model risk probabilities.

    Parameters
    ----------
    probabilities : list[float]
        Raw risk probabilities output by each model.  Values should be in
        [0, 1].  Models whose output is ``None`` (e.g. Deep Path not
        triggered) must be excluded *before* calling this function.

    Returns
    -------
    (normalised_entropy, uncertainty_level) : (float, str)
        ``normalised_entropy`` is in [0, 1].
        ``uncertainty_level`` is one of ``"Low"``, ``"Medium"``, ``"High"``.
    """
    n = len(probabilities)
    if n <= 1:
        return 0.0, "Low"

    probs = np.array(probabilities, dtype=np.float64)

    probs = np.clip(probs, 0.0, 1.0)

    total = probs.sum()
    if total < 1e-12:
        return 1.0, "High"

    q = probs / total
    q = q[q > 0]

    entropy = -float(np.sum(q * np.log2(q)))

    max_entropy = np.log2(n)
    if max_entropy < 1e-12:
        normalised = 0.0
    else:
        normalised = entropy / max_entropy

    normalised = float(np.clip(normalised, 0.0, 1.0))

    if normalised < 0.3:
        level = "Low"
    elif normalised < 0.7:
        level = "Medium"
    else:
        level = "High"

    return round(normalised, 4), level
