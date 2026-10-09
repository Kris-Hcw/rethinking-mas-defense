"""Lightweight exact-entropy manifest and result-row validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


EXACT_FULL_VOCAB_ENTROPY_SOURCE = "raw_generation_logits"
RAW_MODEL_OUTPUT_HEAD_ENTROPY_DOMAIN = "raw_model_output_head"
PADDED_OUTPUT_HEAD_TOKENIZER_POLICY = (
    "contiguous_tokenizer_prefix_with_padded_output_head"
)
PADDED_OUTPUT_ID_TREATMENT = "included_in_raw_output_head_softmax_entropy"
RUNTIME_VERSION_FIELDS = (
    ("runtime_python_version", "python"),
    ("runtime_torch_version", "torch"),
    ("runtime_torch_distribution_version", "torch_distribution"),
    ("runtime_transformers_version", "transformers"),
    ("runtime_fastapi_version", "fastapi"),
    ("runtime_uvicorn_version", "uvicorn"),
)


class ExactEntropyContractError(ValueError):
    """The declared exact full-vocabulary entropy contract is incomplete."""


def _validate_padded_output_head_contract(
    model: Mapping[str, object],
    confidence: Mapping[str, object],
    *,
    vocab_size: int,
) -> None:
    """Fail closed when a manifest declares tokenizer/output-head padding."""

    model_fields = (
        "tokenizer_vocab_size",
        "output_head_padding_size",
        "tokenizer_id_layout",
    )
    confidence_fields = (
        "entropy_domain",
        "tokenizer_vocab_policy",
        "padded_output_id_treatment",
    )
    if not any(field in model for field in model_fields) and not any(
        field in confidence for field in confidence_fields
    ):
        return

    try:
        raw_tokenizer_size = model["tokenizer_vocab_size"]
        raw_padding_size = model["output_head_padding_size"]
        tokenizer_id_layout = model["tokenizer_id_layout"]
        entropy_domain = confidence["entropy_domain"]
        tokenizer_policy = confidence["tokenizer_vocab_policy"]
        padded_id_treatment = confidence["padded_output_id_treatment"]
    except KeyError as exc:
        raise ExactEntropyContractError(
            "Exact-entropy manifest has an incomplete padded output-head entropy contract."
        ) from exc
    if isinstance(raw_tokenizer_size, bool) or isinstance(raw_padding_size, bool):
        raise ExactEntropyContractError(
            "Exact-entropy manifest has an invalid padded output-head entropy contract."
        )
    try:
        tokenizer_size = int(raw_tokenizer_size)
        padding_size = int(raw_padding_size)
    except (TypeError, ValueError) as exc:
        raise ExactEntropyContractError(
            "Exact-entropy manifest has an invalid padded output-head entropy contract."
        ) from exc
    valid = (
        tokenizer_size > 0
        and padding_size > 0
        and tokenizer_size + padding_size == vocab_size
        and tokenizer_id_layout == "contiguous_prefix"
        and entropy_domain == RAW_MODEL_OUTPUT_HEAD_ENTROPY_DOMAIN
        and tokenizer_policy == PADDED_OUTPUT_HEAD_TOKENIZER_POLICY
        and padded_id_treatment == PADDED_OUTPUT_ID_TREATMENT
    )
    if not valid:
        raise ExactEntropyContractError(
            "Exact-entropy manifest has an invalid padded output-head entropy contract."
        )


def exact_entropy_contract_from_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactEntropyContractError(
            f"Cannot read exact-entropy backend manifest {manifest_path}: {exc}"
        ) from exc
    try:
        model = payload["model"]
        confidence = payload["confidence"]
        raw_vocab_size = model["vocab_size"]
        mode = confidence["mode"]
        source = confidence["source"]
        runtime = payload["runtime"]
    except (KeyError, TypeError) as exc:
        raise ExactEntropyContractError(
            "Exact-entropy backend manifest requires model.vocab_size, "
            "confidence.mode, confidence.source, and runtime version fields."
        ) from exc
    if isinstance(raw_vocab_size, bool):
        raise ExactEntropyContractError("Exact-entropy vocab_size must be a positive integer.")
    try:
        vocab_size = int(raw_vocab_size)
    except (TypeError, ValueError) as exc:
        raise ExactEntropyContractError(
            "Exact-entropy vocab_size must be a positive integer."
        ) from exc
    if vocab_size <= 0:
        raise ExactEntropyContractError("Exact-entropy vocab_size must be a positive integer.")
    if mode != "exact_full_vocab":
        raise ExactEntropyContractError(
            "Exact-entropy backend manifest confidence.mode must be exact_full_vocab."
        )
    if source != EXACT_FULL_VOCAB_ENTROPY_SOURCE:
        raise ExactEntropyContractError(
            "Exact-entropy backend manifest confidence.source must be "
            f"{EXACT_FULL_VOCAB_ENTROPY_SOURCE}."
        )
    if not isinstance(model, dict) or not isinstance(confidence, dict):
        raise ExactEntropyContractError(
            "Exact-entropy backend manifest model and confidence must be objects."
        )
    _validate_padded_output_head_contract(
        model,
        confidence,
        vocab_size=vocab_size,
    )
    runtime_contract: dict[str, str] = {}
    if not isinstance(runtime, dict):
        raise ExactEntropyContractError(
            "Exact-entropy backend manifest runtime must be an object."
        )
    for identity_field, manifest_field in RUNTIME_VERSION_FIELDS:
        value = runtime.get(manifest_field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or "<" in value
            or ">" in value
        ):
            raise ExactEntropyContractError(
                "Exact-entropy backend manifest requires a resolved non-placeholder "
                f"runtime.{manifest_field} version."
            )
        runtime_contract[identity_field] = value.strip()
    return {
        "full_vocab_size": vocab_size,
        "full_vocab_entropy_source": source,
        **runtime_contract,
    }


def rows_match_exact_entropy_contract(
    rows: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
) -> bool:
    try:
        expected_size = int(expected["full_vocab_size"])
        expected_source = str(expected["full_vocab_entropy_source"])
    except (KeyError, TypeError, ValueError):
        return False
    if expected_size <= 0 or expected_source != EXACT_FULL_VOCAB_ENTROPY_SOURCE:
        return False
    expected_runtime = {
        identity_field: expected.get(identity_field)
        for identity_field, _manifest_field in RUNTIME_VERSION_FIELDS
        if identity_field in expected
    }
    if any(not isinstance(value, str) or not value for value in expected_runtime.values()):
        return False
    return bool(rows) and all(
        row.get("confidence_entropy_mode") == "exact_full_vocab"
        and row.get("full_vocab_size") == expected_size
        and row.get("full_vocab_entropy_source") == expected_source
        and all(row.get(key) == value for key, value in expected_runtime.items())
        for row in rows
    )
