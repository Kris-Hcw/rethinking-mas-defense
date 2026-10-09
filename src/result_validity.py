"""Fail-closed validation for formal attack result rows."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional, Sequence

from src.gsm8k_eval import normalize_numeric_answer


_NO_ATTACK_TYPES = {"none", "clean"}
MATRIX_DIAGNOSTIC_EVIDENCE_SCOPE = (
    "validated_matrix_diagnostic_not_paper_table"
)
MAJORITY_TIE_POLICY = "first_agent_order"
_FORMAL_IDENTITY_FIELDS = (
    "dataset",
    "model",
    "topology",
    "attack_type",
    "defense_mode",
    "num_agents",
    "rounds",
    "majority_tie_policy",
    "config_hash",
    "code_fingerprint_sha256",
    "confidence_entropy_mode",
    "model_revision",
    "embedding_revision",
    "embedding_identity",
    "embedding_manifest_sha256",
    "backend_revision",
    "backend_manifest_sha256",
    "sampling_top_k",
    "full_vocab_size",
    "full_vocab_entropy_source",
    "runtime_python_version",
    "runtime_torch_version",
    "runtime_torch_distribution_version",
    "runtime_transformers_version",
    "runtime_fastapi_version",
    "runtime_uvicorn_version",
)
_VISIBLE_WRAPPER_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*<ANSWER>\s*:\s*([^\s<]+)\s*$",
    re.IGNORECASE,
)


class FormalResultValidationError(ValueError):
    """Raised when an artifact cannot be used as current formal evidence."""


def majority_vote_first_agent_order(
    answers: Sequence[Optional[str]],
) -> tuple[Optional[str], dict[str, int]]:
    """Match the recovered author implementation's deterministic tie behavior."""

    counts: dict[str, int] = {}
    for answer in answers:
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None, {}
    winning_count = max(counts.values())
    winner = next(
        answer
        for answer in answers
        if answer is not None and counts.get(answer) == winning_count
    )
    return winner, counts


def _recorded_answers_complete(row: dict[str, Any]) -> bool:
    """Reject current-schema rows that silently drop an agent's parsed answer."""

    final_answers = row.get("per_agent_answers")
    if isinstance(final_answers, list) and any(
        not isinstance(answer, str) or not answer.strip() for answer in final_answers
    ):
        return False

    round_results = row.get("round_results")
    if round_results is None:
        # Legacy selection-only fixtures/artifacts carry no agent outputs.
        return True
    if not isinstance(round_results, list) or not round_results:
        return False
    for round_result in round_results:
        if not isinstance(round_result, dict):
            return False
        agents = round_result.get("agents")
        if not isinstance(agents, list) or not agents:
            return False
        if any(
            not isinstance(agent, dict)
            or not isinstance(agent.get("answer"), str)
            or not agent["answer"].strip()
            for agent in agents
        ):
            return False
    return True


def _attack_type(row: dict[str, Any]) -> Optional[str]:
    value = row.get("attack_type")
    if value is None:
        value = row.get("attack_kind")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _attacker_ids(row: dict[str, Any]) -> Optional[list[int]]:
    values = row.get("attacker_ids")
    if not isinstance(values, (list, tuple, set)):
        return None
    try:
        resolved = [int(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(resolved) != len(set(resolved)):
        return None
    return resolved


def _round_result_selections(
    row: dict[str, Any],
    attacker_ids: list[int],
    rounds: int,
) -> Optional[list[Any]]:
    round_results = row.get("round_results")
    if not isinstance(round_results, list) or not round_results:
        return None
    if len(round_results) != rounds:
        return []

    expected_ids = set(attacker_ids)
    selections: list[Any] = []
    for round_result in round_results:
        if not isinstance(round_result, dict):
            return []
        agents = round_result.get("agents")
        if not isinstance(agents, list):
            return []
        attackers = [
            agent
            for agent in agents
            if isinstance(agent, dict) and agent.get("role") == "attacker"
        ]
        try:
            actual_ids = [int(agent.get("agent_id")) for agent in attackers]
        except (TypeError, ValueError):
            return []
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            return []
        selections.extend(agent.get("selection") for agent in attackers)
    return selections


def _legacy_selections(row: dict[str, Any], attacker_ids: list[int]) -> list[Any]:
    explicit = row.get("attack_selections")
    if isinstance(explicit, list):
        return list(explicit)

    histories = row.get("agent_histories")
    if not isinstance(histories, list):
        return []

    selections: list[Any] = []
    if histories and all(isinstance(history, dict) for history in histories):
        for history in histories:
            selection_history = history.get("attack_selection_history")
            if not isinstance(selection_history, list):
                return []
            selections.extend(selection_history)
        return selections

    for attacker_id in attacker_ids:
        if attacker_id < 0 or attacker_id >= len(histories):
            return []
        history = histories[attacker_id]
        if not isinstance(history, list):
            return []
        for item in history:
            if not isinstance(item, dict):
                return []
            selections.append(item.get("selection"))
    return selections


def _normalize_observable_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.search(r"\d", text):
        numeric = normalize_numeric_answer(text)
        if numeric is not None:
            return numeric
    return text.replace(",", "").rstrip(".").upper()


def _selection_valid(
    selection: Any,
    *,
    attack_type: str,
    expected_target: Any = None,
) -> bool:
    if not isinstance(selection, dict):
        return False
    reason = str(selection.get("selection_reason", "")).lower()
    if "fallback" in reason:
        return False
    selected_metrics = selection.get("selected_metrics")
    if not isinstance(selected_metrics, dict):
        return False
    formally_valid = selected_metrics.get("formally_valid")
    if formally_valid is None:
        formally_valid = bool(
            selected_metrics.get("constraint_satisfied")
            and selected_metrics.get("target_answered")
        )
    if formally_valid is not True:
        return False
    if attack_type != "benign_wrapper":
        return True

    wrapper = selected_metrics.get("wrapper")
    payload = selected_metrics.get("payload")
    message = selected_metrics.get("message")
    token_count = selected_metrics.get("payload_token_count")
    tokenizer_model = selected_metrics.get("payload_tokenizer_model")
    visible_answers = (
        _VISIBLE_WRAPPER_ANSWER_RE.findall(payload)
        if isinstance(payload, str)
        else []
    )
    visible_answer = (
        _normalize_observable_answer(visible_answers[0])
        if len(visible_answers) == 1
        else None
    )
    recorded_answer = _normalize_observable_answer(selected_metrics.get("answer"))
    target_answer = _normalize_observable_answer(expected_target)
    if (
        selected_metrics.get("wrapper_serialization_valid") is not True
        or not isinstance(wrapper, str)
        or not wrapper.strip()
        or not isinstance(payload, str)
        or not payload.strip()
        or message != f"{wrapper.strip()} || {payload.strip()}"
        or str(message).count(" || ") != 1
        or "payload_token_count_approx" in selected_metrics
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count < 0
        or not isinstance(tokenizer_model, str)
        or not tokenizer_model.strip()
        or len(re.findall(r"<ANSWER>\s*:", payload, re.IGNORECASE)) != 1
        or selected_metrics.get("target_answered") is not True
        or visible_answer is None
        or visible_answer != recorded_answer
        or visible_answer != target_answer
    ):
        return False
    return True


def attack_result_valid(row: dict[str, Any]) -> bool:
    """Validate every attacker-round selection for any active attack.

    Clean rows are selection-free.  An active attack fails closed when its
    type, round count, attacker set, selection count, or formal evidence is
    missing or inconsistent.  Historical Table 1 shapes are accepted only
    when they carry the same complete evidence.
    """

    if not _recorded_answers_complete(row):
        return False
    attack_type = _attack_type(row)
    if attack_type in _NO_ATTACK_TYPES:
        return True
    if attack_type is None or row.get("attack_valid") is False:
        return False

    attacker_ids = _attacker_ids(row)
    try:
        rounds = int(row.get("rounds", 0))
    except (TypeError, ValueError):
        return False
    if not attacker_ids or rounds <= 0:
        return False

    selections = _round_result_selections(row, attacker_ids, rounds)
    if selections is None:
        selections = _legacy_selections(row, attacker_ids)
    expected = rounds * len(attacker_ids)
    return len(selections) == expected and all(
        _selection_valid(
            selection,
            attack_type=attack_type,
            expected_target=row.get("target_wrong"),
        )
        for selection in selections
    )


def _resolved_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() != "unresolved"
    )


def validate_formal_result_rows(
    rows: list[dict[str, Any]],
    *,
    expected_sample_count: Optional[int] = None,
    expected_config_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Validate a complete, current-schema evaluator JSONL artifact.

    This gate is intentionally stronger than ``attack_result_valid``.  It binds
    every row to one formal execution identity, exact full-vocabulary entropy,
    unique sample/query coverage, and the fixed round/agent structure before a
    generic matrix or aggregation utility may consume it.
    """

    if not isinstance(rows, list) or not rows:
        raise FormalResultValidationError("formal result rows are empty")
    if expected_sample_count is not None and len(rows) != int(expected_sample_count):
        raise FormalResultValidationError(
            f"formal sample count mismatch: expected {expected_sample_count}, found {len(rows)}"
        )

    first = rows[0]
    for field in _FORMAL_IDENTITY_FIELDS:
        value = first.get(field)
        if field in {"num_agents", "rounds", "sampling_top_k", "full_vocab_size"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise FormalResultValidationError(
                    f"formal identity field {field} is missing or not an integer"
                )
        elif not _resolved_text(value):
            raise FormalResultValidationError(
                f"formal identity field {field} is missing or unresolved"
            )
    if first["confidence_entropy_mode"] != "exact_full_vocab":
        raise FormalResultValidationError(
            "formal rows require confidence_entropy_mode=exact_full_vocab"
        )
    if first["majority_tie_policy"] != MAJORITY_TIE_POLICY:
        raise FormalResultValidationError(
            f"formal rows require majority_tie_policy={MAJORITY_TIE_POLICY}"
        )
    if int(first["sampling_top_k"]) < 0 or int(first["full_vocab_size"]) <= 0:
        raise FormalResultValidationError(
            "formal backend sampling/vocabulary identity is invalid"
        )
    if expected_config_hash is not None and first["config_hash"] != expected_config_hash:
        raise FormalResultValidationError("formal config_hash does not match the condition")

    num_agents = int(first["num_agents"])
    rounds = int(first["rounds"])
    if num_agents <= 0 or rounds <= 0:
        raise FormalResultValidationError("formal num_agents/rounds must be positive")

    steps: set[int] = set()
    query_ids: set[str] = set()
    for row in rows:
        for field in _FORMAL_IDENTITY_FIELDS:
            if row.get(field) != first[field]:
                raise FormalResultValidationError(
                    f"mixed formal identity field {field}"
                )
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalResultValidationError("formal row has an invalid step") from exc
        if step in steps:
            raise FormalResultValidationError(f"duplicate formal step={step}")
        steps.add(step)

        question = row.get("question")
        query_id = row.get("query_id")
        if not isinstance(question, str) or not isinstance(query_id, str):
            raise FormalResultValidationError(
                f"formal row step={step} lacks question/query_id identity"
            )
        computed_query_id = hashlib.md5(question.encode("utf-8")).hexdigest()
        if query_id != computed_query_id:
            raise FormalResultValidationError(
                f"formal row step={step} query_id does not match its question"
            )
        if query_id in query_ids:
            raise FormalResultValidationError(f"duplicate formal query_id={query_id}")
        query_ids.add(query_id)

        bbh_answer_domain: Optional[set[str]] = None
        if row.get("dataset") == "bbh":
            raw_options = row.get("answer_options")
            if not isinstance(raw_options, list):
                raise FormalResultValidationError(
                    f"formal BBH answer domain is missing at step={step}"
                )
            normalized_options = [str(value).strip().upper() for value in raw_options]
            if (
                not 2 <= len(normalized_options) <= 4
                or len(set(normalized_options)) != len(normalized_options)
                or any(len(value) != 1 or value not in "ABCD" for value in normalized_options)
            ):
                raise FormalResultValidationError(
                    f"formal BBH answer domain is invalid at step={step}"
                )
            bbh_answer_domain = set(normalized_options)
            normalized_gold = str(
                row.get("normalized_ground_truth", row.get("gold", ""))
            ).strip().upper()
            normalized_pred = str(
                row.get("normalized_prediction", row.get("pred", ""))
            ).strip().upper()
            if normalized_gold not in bbh_answer_domain:
                raise FormalResultValidationError(
                    f"formal BBH answer domain excludes gold at step={step}"
                )
            if normalized_pred not in bbh_answer_domain:
                raise FormalResultValidationError(
                    f"formal BBH answer domain excludes prediction at step={step}"
                )
            if _attack_type(row) not in _NO_ATTACK_TYPES:
                target_wrong = str(row.get("target_wrong", "")).strip().upper()
                if (
                    target_wrong not in bbh_answer_domain
                    or target_wrong == normalized_gold
                ):
                    raise FormalResultValidationError(
                        f"formal BBH answer domain has an invalid attack target at step={step}"
                    )

        if not attack_result_valid(row):
            raise FormalResultValidationError(
                f"formal attack result is invalid at step={step}"
            )
        attacker_ids = _attacker_ids(row)
        if attacker_ids is None:
            raise FormalResultValidationError(
                f"formal row step={step} has invalid attacker_ids"
            )
        if any(agent_id < 0 or agent_id >= num_agents for agent_id in attacker_ids):
            raise FormalResultValidationError(
                f"formal row step={step} has out-of-range attacker_ids"
            )

        round_results = row.get("round_results")
        if not isinstance(round_results, list) or len(round_results) != rounds:
            raise FormalResultValidationError(
                f"formal row step={step} has incomplete round_results"
            )
        observed_rounds: set[int] = set()
        expected_agent_ids = set(range(num_agents))
        expected_attacker_ids = set(attacker_ids)
        for round_result in round_results:
            try:
                round_id = int(round_result["round"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FormalResultValidationError(
                    f"formal row step={step} has an invalid round"
                ) from exc
            if round_id in observed_rounds:
                raise FormalResultValidationError(
                    f"formal row step={step} has duplicate round={round_id}"
                )
            observed_rounds.add(round_id)
            agents = round_result.get("agents")
            if not isinstance(agents, list) or len(agents) != num_agents:
                raise FormalResultValidationError(
                    f"formal row step={step} round={round_id} has incomplete agents"
                )
            try:
                agent_ids = [int(agent["agent_id"]) for agent in agents]
            except (KeyError, TypeError, ValueError) as exc:
                raise FormalResultValidationError(
                    f"formal row step={step} round={round_id} has invalid agent IDs"
                ) from exc
            if len(set(agent_ids)) != len(agent_ids) or set(agent_ids) != expected_agent_ids:
                raise FormalResultValidationError(
                    f"formal row step={step} round={round_id} agent coverage is invalid"
                )
            for agent, agent_id in zip(agents, agent_ids):
                expected_role = (
                    "attacker" if agent_id in expected_attacker_ids else "benign"
                )
                if agent.get("role") != expected_role:
                    raise FormalResultValidationError(
                        f"formal row step={step} round={round_id} role mismatch"
                    )
                if bbh_answer_domain is not None:
                    agent_answer = str(agent.get("answer", "")).strip().upper()
                    if agent_answer not in bbh_answer_domain:
                        raise FormalResultValidationError(
                            "formal BBH answer domain excludes an agent answer at "
                            f"step={step} round={round_id} agent={agent_id}"
                        )
        if observed_rounds != set(range(1, rounds + 1)):
            raise FormalResultValidationError(
                f"formal row step={step} round coverage is invalid"
            )

        final_answers = row.get("per_agent_answers")
        if not isinstance(final_answers, list) or len(final_answers) != num_agents:
            raise FormalResultValidationError(
                f"formal row step={step} final answer coverage is invalid"
            )
        expected_pred, expected_vote_counts = majority_vote_first_agent_order(
            final_answers
        )
        if (
            row.get("pred") != expected_pred
            or row.get("vote_counts") != expected_vote_counts
        ):
            raise FormalResultValidationError(
                f"formal row step={step} majority vote does not match "
                f"{MAJORITY_TIE_POLICY}"
            )
        if bbh_answer_domain is not None and any(
            str(answer).strip().upper() not in bbh_answer_domain
            for answer in final_answers
        ):
            raise FormalResultValidationError(
                f"formal BBH answer domain excludes a final answer at step={step}"
            )

    if steps != set(range(1, len(rows) + 1)):
        raise FormalResultValidationError("formal sample step coverage is not contiguous")

    query_digest = hashlib.sha256(
        "\n".join(
            str(row["query_id"])
            for row in sorted(rows, key=lambda item: int(item["step"]))
        ).encode("utf-8")
    ).hexdigest()
    return {
        **{field: first[field] for field in _FORMAL_IDENTITY_FIELDS},
        "sample_count": len(rows),
        "query_ids_sha256": query_digest,
        "source_rows_validated": True,
    }


def validate_matrix_final_summary(
    value: dict[str, Any],
    *,
    expected_sample_count: Optional[int] = None,
    expected_config_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Validate the non-paper-table matrix summary emitted from formal rows."""

    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise FormalResultValidationError("matrix final summary schema is not current")
    if value.get("source_rows_validated") is not True:
        raise FormalResultValidationError("matrix final summary lacks validated source rows")
    if value.get("evidence_scope") != MATRIX_DIAGNOSTIC_EVIDENCE_SCOPE:
        raise FormalResultValidationError("matrix final summary evidence scope is invalid")
    if value.get("formal_paper_table_eligible") is not False:
        raise FormalResultValidationError(
            "matrix final summary must not claim formal paper-table eligibility"
        )
    try:
        sample_count = int(value["sample_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalResultValidationError("matrix final summary sample_count is invalid") from exc
    if sample_count <= 0 or (
        expected_sample_count is not None and sample_count != int(expected_sample_count)
    ):
        raise FormalResultValidationError("matrix final summary sample_count mismatch")
    if expected_config_hash is not None and value.get("config_hash") != expected_config_hash:
        raise FormalResultValidationError("matrix final summary config_hash mismatch")
    for field in _FORMAL_IDENTITY_FIELDS:
        if field not in value:
            raise FormalResultValidationError(
                f"matrix final summary lacks identity field {field}"
            )
    if value.get("confidence_entropy_mode") != "exact_full_vocab":
        raise FormalResultValidationError("matrix final summary is not exact entropy")
    if not _resolved_text(value.get("query_ids_sha256")):
        raise FormalResultValidationError("matrix final summary lacks query identity digest")
    return value
