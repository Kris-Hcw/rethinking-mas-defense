"""
Result analysis — compute accuracy and confidence statistics from JSONL logs.

Usage:
  # Single file
  python analyze.py results/mmlu_llama3-8b_slow_drift_pruning_sparse_random.jsonl

  # Compare multiple files (summary table)
  python analyze.py results/*.jsonl --summary

  # Interactive per-sample viewer
  python analyze.py results/mmlu_llama3-8b_slow_drift_pruning_sparse_random.jsonl --view
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
class C:
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def _c(color: str, text: str) -> str:
    return color + text + C.RESET


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(rows: List[dict]) -> Dict[str, Any]:
    total = correct = 0
    attack_success = 0   # fraction of examples where pred == target_wrong

    for r in rows:
        gold = r.get("gold")
        pred = r.get("pred")
        target = r.get("target_wrong")

        if gold is None or pred is None:
            continue
        total += 1
        if pred == gold:
            correct += 1
        if target and pred == target:
            attack_success += 1

    return {
        "total":    total,
        "correct":  correct,
        "accuracy": correct / total if total else 0.0,
        "asr":      attack_success / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Summary table across multiple files
# ---------------------------------------------------------------------------

def _short_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def print_summary(paths: List[str]) -> None:
    fmt = "{:<60s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}"
    print(_c(C.BOLD, fmt.format("File", "Total", "Acc", "ASR", "Attack")))
    print("-" * 90)
    for path in sorted(paths):
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            print(f"  (not found: {path})")
            continue
        stats = compute_accuracy(rows)
        attack = rows[0].get("attack_type", "?") if rows else "?"
        defense = rows[0].get("defense_mode", "?") if rows else "?"
        tag = f"{attack}/{defense}"
        print(fmt.format(
            _short_name(path)[:60],
            str(stats["total"]),
            f"{stats['accuracy']:.3f}",
            f"{stats['asr']:.3f}",
            tag,
        ))


# ---------------------------------------------------------------------------
# Per-sample interactive viewer
# ---------------------------------------------------------------------------

TRUNCATE = 300

def _trunc(s: Optional[str], n: int = TRUNCATE) -> str:
    if not s:
        return ""
    s = " ".join(str(s).split())
    return s[:n] + " [...]" if len(s) > n else s


_REASON_RE = re.compile(r"<REASON>\s*:\s*(.*?)(?:\n\s*<ANSWER>|$)", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<ANSWER>\s*:\s*(\S+)", re.IGNORECASE)


def _parse(resp: str):
    m_r = _REASON_RE.search(resp or "")
    m_a = _ANSWER_RE.search(resp or "")
    return (
        (m_r.group(1).strip() if m_r else resp or ""),
        (m_a.group(1).strip() if m_a else "?"),
    )


def view_log(path: str) -> None:
    rows = load_jsonl(path)
    print(f"{len(rows)} samples in {path}\n")

    for row in rows:
        gold     = row.get("gold", "?")
        pred     = row.get("pred", "?")
        correct  = pred == gold
        attack   = row.get("attack_type", "none")
        defense  = row.get("defense_mode", "none")
        target   = row.get("target_wrong", "?")
        atk_ids  = row.get("attacker_ids", [])
        n_agents = row.get("num_agents", 0)
        n_rounds = row.get("rounds", 0)
        model    = row.get("model", "?")

        status_col = C.GREEN if correct else C.RED
        print("\n" + "=" * 90)
        print(_c(C.BOLD, f"Sample #{row.get('step')}  |  Subject: {row.get('subject')}"))
        print(_c(C.CYAN, f"Model: {model}  agents={n_agents}  rounds={n_rounds}"))
        print(_c(C.YELLOW, f"Attack: {attack}  target={target}  attackers={atk_ids}"))
        print(_c(C.CYAN, f"Defense: {defense}"))
        print(_c(status_col, f"Pred: {pred}  Gold: {gold}  {'✓ CORRECT' if correct else '✗ WRONG'}"))
        print(_c(C.BLUE, "-" * 50))

        histories = row.get("agent_histories", [])
        confs     = row.get("per_agent_conf", [None] * n_agents)

        for rnd in range(n_rounds):
            print(_c(C.BOLD, f"\n  --- Round {rnd+1} ---"))
            for ag_idx in range(n_agents):
                is_atk = ag_idx in atk_ids
                role_col = C.RED if is_atk else C.GREEN
                role_lbl = "ATTACKER" if is_atk else "benign"

                try:
                    interaction = histories[ag_idx][rnd]
                except (IndexError, TypeError):
                    print(f"    [{ag_idx}] no history")
                    continue

                resp  = interaction.get("response", "")
                reason, answer = _parse(resp)
                conf_val = confs[ag_idx] if confs and ag_idx < len(confs) else None
                conf_str = f"{conf_val:.3f}" if isinstance(conf_val, float) else "n/a"

                print(_c(role_col, f"    Agent {ag_idx} [{role_lbl}]") +
                      f"  conf={conf_str}  answer={answer}")
                print(f"      Reason: {_trunc(reason)}")

        print(_c(C.BOLD, f"\n  Votes: {row.get('vote_counts')}"))
        print()

        inp = input("Enter to continue, 'q' to quit... ").strip().lower()
        if inp == "q":
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--summary", action="store_true",
                   help="Print a summary table across all provided files.")
    p.add_argument("--view", action="store_true",
                   help="Interactive per-sample viewer (first file only).")
    args = p.parse_args()

    if args.view:
        view_log(args.files[0])
        return

    if args.summary or len(args.files) > 1:
        print_summary(args.files)
        return

    # Single file: detailed stats
    rows = load_jsonl(args.files[0])
    stats = compute_accuracy(rows)
    attack  = rows[0].get("attack_type", "?") if rows else "?"
    defense = rows[0].get("defense_mode", "?") if rows else "?"

    print(f"File:     {args.files[0]}")
    print(f"Samples:  {stats['total']}")
    print(f"Attack:   {attack}")
    print(f"Defense:  {defense}")
    print(f"Accuracy: {stats['accuracy']:.4f}  ({stats['correct']}/{stats['total']})")
    print(f"ASR:      {stats['asr']:.4f}  (fraction pred == target_wrong)")


if __name__ == "__main__":
    main()
