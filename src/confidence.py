"""
Token-level confidence scoring (Section 6.1 of the paper).

  U(m) = (1/k) * sum_{j=1}^{k} H_{(j)}(m)        [top-k mean entropy]
  C(m) = exp(-U(m))                                [confidence ∈ (0, 1]]

where H_{(j)}(m) is the j-th largest per-token entropy.
Larger C(m) means higher confidence (lower uncertainty).
"""

import math
from typing import List, Optional

from src.api import TokenInfo


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(top_logprobs: List[tuple]) -> Optional[float]:
    """
    Compute Shannon entropy from a top-k logprob distribution.

    The top-k probabilities may not sum to 1.  We assign the residual mass
    1 - sum(probs) to an implicit "other" bucket before computing entropy.
    """
    if not top_logprobs:
        return None

    probs = [math.exp(lp) for _, lp in top_logprobs]
    s = sum(probs)
    if s < 1.0:
        probs.append(1.0 - s)          # residual bucket
    elif s > 1.0:
        probs = [p / s for p in probs] # re-normalise if floats overflow

    H = 0.0
    for p in probs:
        if p > 1e-12:
            H -= p * math.log(p)
    return H


# ---------------------------------------------------------------------------
# Per-message confidence
# ---------------------------------------------------------------------------

def compute_confidence(token_infos: List[TokenInfo], top_k: int = 10) -> Optional[float]:
    """
    Compute C(m) = exp(-U(m)) as defined in Section 6.1.

    Args:
        token_infos: Per-token logprob data returned by the LLM.
        top_k:       Number of highest-entropy tokens to average (k in the paper).

    Returns:
        Confidence score in (0, 1], or None if logprob data is unavailable.
    """
    entropies: List[float] = []
    for ti in token_infos:
        H = entropy_from_top_logprobs(ti.top_logprobs)
        if H is not None:
            entropies.append(H)

    if not entropies:
        return None

    top_entropies = sorted(entropies, reverse=True)[:top_k]
    U = sum(top_entropies) / len(top_entropies)
    return math.exp(-U)
