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
from typing import List, Optional


def load_local(
    path: str,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> List[dict]:
    """
    Load evaluation records from a local JSONL file.

    Args:
        path:      Path to the JSONL file.
        n_samples: Maximum number of records to return.
                   Records are shuffled before capping so every run with
                   the same seed yields the same subset.
        seed:      Random seed for shuffling.

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

            question = str(obj.get("question") or "")
            choices  = [str(c) for c in (obj.get("choices") or [])]
            gold     = str(obj.get("gold") or "").strip()
            subject  = str(obj.get("subject") or "unknown")

            if not question or not gold:
                continue

            records.append({
                "question": question,
                "choices":  choices,
                "gold":     gold,
                "subject":  subject,
            })

    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n_samples] if n_samples is not None else records
