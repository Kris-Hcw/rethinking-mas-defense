"""
Local dataset loader.

All datasets are loaded from JSONL files prepared by the user.
Each line must be a JSON object with the following fields:

  {
    "question": str,         # full question text (with answer choices appended for MCQA)
    "choices":  List[str],   # e.g. ["A. ...", "B. ...", "C. ...", "D. ..."]
                             # empty list for GSM8K
    "gold":     str,         # "A"/"B"/"C"/"D" for MCQA, numeric string for GSM8K
    "subject":  str          # subject / category tag (used for logging)
  }

Prepare your data files before running evaluation:
  data/mmlu_test.jsonl
  data/gsm8k_test.jsonl
  data/bbh_test.jsonl
"""

import json
import random
import re
from typing import List, Optional

from src.gsm8k_eval import normalize_numeric_answer


_LABELLED_CHOICE = re.compile(r"^\s*([A-D])\.\s*(.*)$", re.IGNORECASE)


def load_local(
    path: str,
    n_samples: Optional[int] = None,
    seed: int = 0,
    dataset: Optional[str] = None,
    preserve_order: bool = False,
) -> List[dict]:
    """
    Load evaluation records from a local JSONL file.

    Args:
        path:      Path to the JSONL file.
        n_samples: Maximum number of records to return.
                   Records are shuffled before capping so every run with
                   the same seed yields the same subset.
        seed:      Random seed for shuffling.
        preserve_order: Keep source order when true; default shuffle behavior
                        remains unchanged.

    Returns:
        List of dicts with keys: question, choices, gold, subject.
    """
    records: List[dict] = []

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {e}") from e

            raw_question = str(obj.get("input") or obj.get("question") or "")
            raw_choices = [str(c) for c in (obj.get("choices") or obj.get("answer_choices") or [])]
            raw_target = str(obj.get("target") or obj.get("gold") or "").strip()
            question = raw_question
            choices = raw_choices
            gold = raw_target
            adapter = "identity"
            if dataset == "gsm8k":
                normalized = normalize_numeric_answer(raw_target)
                if normalized is None:
                    continue
                gold = normalized
                adapter = "gsm8k_numeric"
            if dataset == "bbh":
                question, choices, gold, adapter = _adapt_bbh(
                    raw_question, raw_choices, raw_target
                )
            subject = str(obj.get("subject") or obj.get("task") or "unknown")
            bbh_task_name = str(obj.get("task") or subject.replace("bbh_", "", 1))
            output_type = "unresolved"
            if dataset == "bbh" and choices and 2 <= len(choices) <= 4:
                output_type = "multiple_choice"

            if not question or not gold:
                continue

            record = {
                "question": question,
                "choices": choices,
                "gold": gold,
                "subject": subject,
                "raw_question": raw_question,
                "raw_target": raw_target,
                "raw_choices": raw_choices,
                "answer_adapter": adapter,
            }
            if dataset == "bbh":
                record["bbh_task_name"] = bbh_task_name
                record["output_type"] = output_type
                parsed_choices = [_LABELLED_CHOICE.match(choice) for choice in choices]
                if choices and all(match is not None for match in parsed_choices):
                    record["answer_options"] = [
                        match.group(1).upper()
                        for match in parsed_choices
                        if match is not None
                    ]
            records.append(record)

    if not preserve_order:
        rng = random.Random(seed)
        rng.shuffle(records)
    return records[:n_samples] if n_samples is not None else records


def _adapt_bbh(
    question: str, raw_choices: List[str], target: str
) -> tuple[str, List[str], str, str]:
    """Adapt raw BBH labels at evaluation time without rewriting the dataset."""
    if not raw_choices:
        return question, [], target, "bbh_identity"

    parsed = [_LABELLED_CHOICE.match(choice) for choice in raw_choices]
    if all(match is not None for match in parsed):
        labels = [match.group(1).upper() for match in parsed if match is not None]
        if not 2 <= len(labels) <= 4 or len(set(labels)) != len(labels):
            raise ValueError(
                "BBH evaluator adapter requires two to four unique labelled choices."
            )
        if target.upper() in labels:
            return question, raw_choices, target.upper(), "bbh_labelled_choices"

    # Raw BBH tasks carry semantic labels (for example "yes"/"no" or
    # task-specific strings).  Preserve those strings in raw_choices and map
    # only the evaluator-facing answer to stable A/B/C/D letters.
    labels = [choice.strip() for choice in raw_choices]
    if target not in labels:
        return question, raw_choices, target, "bbh_unmapped_target"
    letters = "ABCD"
    if len(labels) > len(letters):
        raise ValueError("BBH evaluator adapter supports at most four answer choices.")
    mapping = {label: letters[index] for index, label in enumerate(labels)}
    formatted = [f"{letters[index]}. {label}" for index, label in enumerate(labels)]
    adapted_question = question.rstrip() + "\n\nChoose one:\n" + "\n".join(formatted)
    return adapted_question, formatted, mapping[target], "bbh_raw_label_adapter"
