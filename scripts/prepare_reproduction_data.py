"""Prepare the repository's raw MMLU, GSM8K, and BBH files for evaluate.py.

The source datasets stay untouched.  Prepared files use the JSONL schema that
the evaluator consumes and are sampled deterministically for a low-cost run.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List


def _write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(rows: List[dict], n_samples: int, seed: int) -> List[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n_samples]


def prepare_mmlu(source: Path, n_samples: int, seed: int) -> List[dict]:
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return _sample(rows, n_samples, seed)


def prepare_gsm8k(source: Path, n_samples: int, seed: int) -> List[dict]:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for obj in map(json.loads, filter(str.strip, handle)):
            match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", obj.get("answer", ""))
            if not match:
                continue
            gold = match.group(1).replace(",", "")
            rows.append({"question": obj["question"], "choices": [], "gold": gold, "subject": "gsm8k"})
    return _sample(rows, n_samples, seed)


def _normalise_target(value: str) -> str:
    value = str(value).strip()
    if len(value) >= 3 and value[0] == "(" and value[-1] == ")":
        value = value[1:-1].strip()
    return value


def prepare_bbh(root: Path, n_samples: int, seed: int) -> List[dict]:
    by_task = []
    for source in sorted(root.glob("*.json")):
        if source.name.lower() == "readme.json":
            continue
        obj = json.loads(source.read_text(encoding="utf-8"))
        task_rows = []
        for example in obj.get("examples", []):
            target = _normalise_target(example.get("target", ""))
            question = str(example.get("input", "")).strip()
            if question and target:
                task_rows.append({"question": question, "choices": [], "gold": target, "subject": source.stem})
        if task_rows:
            random.Random(seed + len(by_task)).shuffle(task_rows)
            by_task.append(task_rows)
    # Round-robin keeps the reduced run representative of the available BBH tasks.
    rows = []
    while len(rows) < n_samples and any(by_task):
        for task_rows in by_task:
            if task_rows and len(rows) < n_samples:
                rows.append(task_rows.pop())
    random.Random(seed).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    _write(args.output_dir / "mmlu.jsonl", prepare_mmlu(args.data_dir / "MMLU" / "mmlu_test.jsonl", args.n_samples, args.seed))
    _write(args.output_dir / "gsm8k.jsonl", prepare_gsm8k(args.data_dir / "GSM8K" / "test.jsonl", args.n_samples, args.seed))
    _write(args.output_dir / "bbh.jsonl", prepare_bbh(args.data_dir / "BBH", args.n_samples, args.seed))


if __name__ == "__main__":
    main()
