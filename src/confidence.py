"""
Token-level confidence scoring (Section 6.1 of the paper).

  U(m) = (1/k) * sum_{j=1}^{k} H_{(j)}(m)        [top-k mean entropy]
  C(m) = exp(-U(m))                                [confidence ∈ (0, 1]]

where H_{(j)}(m) is the j-th largest per-token entropy.
Larger C(m) means higher confidence (lower uncertainty).
"""

import math
from typing import List, Literal, Optional

from src.llm_client import LogprobsUnavailableError, TokenInfo
from src.entropy_contract import EXACT_FULL_VOCAB_ENTROPY_SOURCE


ConfidenceEntropyMode = Literal["exact_full_vocab", "top_logprobs_tail_bucket"]


class FullVocabularyLogprobsUnavailableError(LogprobsUnavailableError):
    """Exact paper entropy was requested but the backend returned a truncated distribution."""


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(
    top_logprobs: List[tuple],
    *,
    mode: ConfidenceEntropyMode = "top_logprobs_tail_bucket",
    exhaustive: bool = False,
    residual_tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Compute Shannon entropy from a top-k logprob distribution.

    ``exact_full_vocab`` follows the paper and refuses a truncated
    distribution. ``top_logprobs_tail_bucket`` is an explicitly named
    diagnostic approximation that assigns all omitted probability mass to one
    synthetic bucket; it is a lower bound on the true tail entropy.
    """
    if not top_logprobs:
        return None

    probs = [math.exp(lp) for _, lp in top_logprobs]
    s = sum(probs)
    residual = max(0.0, 1.0 - s)
    complete = exhaustive or residual <= residual_tolerance
    if mode == "exact_full_vocab" and not complete:
        raise FullVocabularyLogprobsUnavailableError(
            "Exact full-vocabulary entropy was requested, but the backend returned "
            f"a truncated distribution with residual mass {residual:.6g}."
        )
    if mode not in {"exact_full_vocab", "top_logprobs_tail_bucket"}:
        raise ValueError(f"Unknown confidence entropy mode: {mode}")

    if s < 1.0 and mode == "top_logprobs_tail_bucket":
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

def compute_confidence(
    token_infos: List[TokenInfo],
    top_k: int = 10,
    *,
    entropy_mode: ConfidenceEntropyMode = "top_logprobs_tail_bucket",
) -> float:
    """
    Compute C(m) = exp(-U(m)) as defined in Section 6.1.

    Args:
        token_infos: Per-token logprob data returned by the LLM.
        top_k:       Number of highest-entropy tokens to average (k in the paper).

    Returns:
        Confidence score in (0, 1].

    Raises:
        LogprobsUnavailableError: If the backend did not expose usable
            top-logprob distributions.  Silent fallback would invalidate the
            confidence defense and is therefore intentionally disallowed.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not token_infos:
        raise LogprobsUnavailableError(
            "No token logprobs were returned; confidence cannot be computed."
        )

    entropies: List[float] = []
    reduced_entropy_contract: Optional[tuple[int, str]] = None
    for ti in token_infos:
        if entropy_mode == "exact_full_vocab" and ti.full_vocab_entropy is not None:
            if (
                ti.full_vocab_size is None
                or ti.full_vocab_size <= 0
                or ti.full_vocab_entropy_source != EXACT_FULL_VOCAB_ENTROPY_SOURCE
            ):
                raise FullVocabularyLogprobsUnavailableError(
                    "Backend-reduced full-vocabulary entropy is missing a valid "
                    "full_vocab_size/raw_generation_logits source contract."
                )
            token_contract = (ti.full_vocab_size, ti.full_vocab_entropy_source)
            if reduced_entropy_contract is None:
                reduced_entropy_contract = token_contract
            elif token_contract != reduced_entropy_contract:
                raise FullVocabularyLogprobsUnavailableError(
                    "Backend-reduced entropy tokens disagree on vocabulary size or source."
                )
            H = ti.full_vocab_entropy
        else:
            H = entropy_from_top_logprobs(
                ti.top_logprobs,
                mode=entropy_mode,
                exhaustive=ti.top_logprobs_exhaustive,
            )
        if H is not None:
            entropies.append(H)

    if not entropies:
        raise LogprobsUnavailableError(
            "No usable top_logprobs distributions were returned; confidence cannot be computed."
        )

    top_entropies = sorted(entropies, reverse=True)[:top_k]
    U = sum(top_entropies) / len(top_entropies)
    return math.exp(-U)


def confidence_weight(confidence: Optional[float], floor: float = 0.0) -> Optional[float]:
    """Map message confidence to the down-weighting metadata used in Section 6.2."""
    if confidence is None:
        return None
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1].")
    value = max(0.0, min(1.0, float(confidence)))
    return max(floor, value)
