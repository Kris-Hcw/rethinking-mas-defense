"""
Main evaluation script.

Usage examples:

  # No attack, no defense (baseline)
  python evaluate.py --data_file data/mmlu_test.jsonl \
      --dataset mmlu --model llama3-8b --attack none --defense none

  # Slow Drift + confidence pruning
  python evaluate.py --data_file data/mmlu_test.jsonl \
      --dataset mmlu --model llama3-8b \
      --attack slow_drift --defense pruning

  # Benign Wrapper + down-weighting
  python evaluate.py --data_file data/gsm8k_test.jsonl \
      --dataset gsm8k --model llama3-8b \
      --attack benign_wrapper --defense downweight

  # GPT-4o-mini (requires OPENAI_API_KEY)
  python evaluate.py --data_file data/mmlu_test.jsonl \
      --dataset mmlu --model gpt-4o-mini \
      --base_url "" --api_key $OPENAI_API_KEY \
      --attack slow_drift --defense pruning
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from typing import List, Optional

from src.api import AsyncLLMClient
from src.data import load_local
from src.mas import DebateMAS, DefenseConfig, AttackConfig
from src.topology import build_adjacency, adjacency_to_list


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MAS safety evaluation")

    # Data
    p.add_argument("--data_file", required=True,
                   help="Path to local JSONL data file (see src/data.py for format).")
    p.add_argument("--dataset",   required=True,
                   choices=["mmlu", "gsm8k", "bbh"],
                   help="Dataset type — controls answer format and target selection.")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Cap on number of samples (shuffled by --seed before capping).")
    p.add_argument("--seed",      type=int, default=0)

    # Model / server
    p.add_argument("--base_url",  default="http://localhost:8001/v1",
                   help="vLLM or OpenAI API base URL. Pass empty string for OpenAI default.")
    p.add_argument("--api_key",   default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model",     default="llama3-8b")
    p.add_argument("--max_concurrency", type=int, default=64)
    p.add_argument("--timeout",         type=float, default=300.0)

    # MAS
    p.add_argument("--n_agents",     type=int,   default=5)
    p.add_argument("--n_rounds",     type=int,   default=3)
    p.add_argument("--temperature",  type=float, default=0.3)
    p.add_argument("--max_tokens",   type=int,   default=512)

    # Topology
    p.add_argument("--topology",         default="sparse_random",
                   choices=["star", "chain", "sparse_random", "full"])
    p.add_argument("--topology_density", type=float, default=0.3)
    p.add_argument("--topology_seed",    type=int,   default=0)

    # Attack
    p.add_argument("--attack",       default="none",
                   choices=["none", "overt", "slow_drift", "benign_wrapper", "chaos_seeding"])
    p.add_argument("--n_attackers",  type=int,   default=2)
    p.add_argument("--attacker_ids", default=None,
                   help="Comma-separated attacker IDs, e.g. '1,3'. Overrides --n_attackers.")
    p.add_argument("--attack_temperature", type=float, default=0.9)
    p.add_argument("--attack_max_tokens",  type=int,   default=512)

    # Defense
    p.add_argument("--defense",         default="none",
                   choices=["none", "pruning", "downweight"])
    p.add_argument("--prune_threshold", type=float, default=0.4,
                   help="Confidence threshold δ for pruning defense.")
    p.add_argument("--top_k_conf",      type=int,   default=10)

    # Output
    p.add_argument("--out_file", default=None,
                   help="Output JSONL path. Auto-generated from config if omitted.")
    p.add_argument("--eval_concurrency", type=int, default=32)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_out_file(args) -> str:
    stem = os.path.splitext(os.path.basename(args.data_file))[0]
    parts = [stem, args.model.replace("/", "_"), args.attack, args.defense]
    if args.defense == "pruning":
        parts.append(f"d{args.prune_threshold}")
    parts.append(args.topology)
    return "results/" + "_".join(parts) + ".jsonl"


def _attacker_ids(args) -> List[int]:
    if args.attacker_ids:
        return [int(x) for x in args.attacker_ids.split(",")]
    return list(range(min(args.n_attackers, args.n_agents)))


def _stable_seed(question: str, base: int) -> int:
    h = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    return (base + h) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    out_path = args.out_file or _auto_out_file(args)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Loading data from: {args.data_file}")
    records = load_local(args.data_file, n_samples=args.n_samples, seed=args.seed)
    print(f"  {len(records)} samples loaded.")

    adj = build_adjacency(args.topology, args.n_agents,
                          density=args.topology_density, seed=args.topology_seed)

    client = AsyncLLMClient(
        base_url=args.base_url or None,
        api_key=args.api_key,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
    )
    defense = DefenseConfig(
        mode=args.defense,
        prune_threshold=args.prune_threshold,
        top_k_conf=args.top_k_conf,
    )
    attack_ids = _attacker_ids(args)
    attack = AttackConfig(
        attack_type=args.attack,
        attacker_ids=attack_ids,
        seed=args.seed,
        temperature=args.attack_temperature,
        max_tokens=args.attack_max_tokens,
    )

    print(f"\nConfig:")
    print(f"  model={args.model}  dataset={args.dataset}")
    print(f"  attack={args.attack}  attackers={attack_ids}")
    print(f"  defense={args.defense}  threshold={args.prune_threshold}")
    print(f"  topology={args.topology}  density={args.topology_density}")
    print(f"  agents={args.n_agents}  rounds={args.n_rounds}")
    print(f"  output → {out_path}\n")

    sem = asyncio.Semaphore(args.eval_concurrency)
    write_lock = asyncio.Lock()
    total = correct = 0
    t0 = time.time()

    async def process_one(step: int, rec: dict) -> None:
        nonlocal total, correct

        mas = DebateMAS(
            client=client,
            model=args.model,
            n_agents=args.n_agents,
            n_rounds=args.n_rounds,
            dataset=args.dataset,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            adj=adj,
            defense=defense,
        )

        async with sem:
            seed_offset = _stable_seed(rec["question"], args.seed)
            res = await mas.run_one(
                question=rec["question"],
                gold=rec["gold"],
                attack=attack,
                seed_offset=seed_offset,
            )

        async with write_lock:
            total += 1
            if rec["gold"] and res["pred"] == rec["gold"]:
                correct += 1
            if total % 20 == 0 or total == len(records):
                print(f"  [{total}/{len(records)}] acc={correct/total:.3f}  "
                      f"elapsed={time.time()-t0:.0f}s")

            row = {
                "step":              step,
                "subject":           rec["subject"],
                "question":          rec["question"],
                "choices":           rec["choices"],
                "gold":              rec["gold"],
                "pred":              res["pred"],
                "vote_counts":       res["vote_counts"],
                "per_agent_answers": res["per_agent_answers"],
                "per_agent_conf":    res["per_agent_conf"],
                "num_agents":        args.n_agents,
                "rounds":            args.n_rounds,
                "model":             args.model,
                "dataset":           args.dataset,
                "attack_type":       res["attack_type"],
                "attacker_ids":      res["attacker_ids"],
                "target_wrong":      res["target_wrong"],
                "defense_mode":      res["defense_mode"],
                "topology":          args.topology,
                "adjacency":         adjacency_to_list(adj),
                "agent_histories":   res["agent_histories"],
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tasks = [asyncio.create_task(process_one(i + 1, rec))
             for i, rec in enumerate(records)]
    await asyncio.gather(*tasks)

    acc = correct / total if total else 0.0
    print(f"\nDONE  total={total}  acc={acc:.4f}  output={out_path}")


if __name__ == "__main__":
    asyncio.run(main())
