"""GSM8K numeric answer extraction and normalization."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


_ANSWER_TAG_RE = re.compile(r"<ANSWER>\s*:\s*([^\n<]+)", re.IGNORECASE)
_GSM8K_FINAL_RE = re.compile(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")


def normalize_numeric_answer(value: object) -> Optional[str]:
    """Return a canonical GSM8K numeric string, or None if no number is found."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    tagged = _GSM8K_FINAL_RE.search(text)
    if tagged:
        text = tagged.group(1)
    else:
        answer_tag = _ANSWER_TAG_RE.search(text)
        if answer_tag:
            text = answer_tag.group(1)
        else:
            matches = _NUMBER_RE.findall(text)
            if not matches:
                return None
            text = matches[-1]

    cleaned = text.strip().replace("$", "").replace(",", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    cleaned = match.group(0).replace("$", "").replace(",", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def extract_gsm8k_prediction(text: str) -> Optional[str]:
    """Extract the model's final numeric answer without using MCQA letter parsing."""
    return normalize_numeric_answer(text)
