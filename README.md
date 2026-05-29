# When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems

ICML 2026 · Lingxi Zhang, Guangtao Zheng, Hanjie Chen

> We show that embedding-based MAS defenses are vulnerable to near-benign attacks that reduce
> embedding separability, and propose confidence-guided defenses using token-level uncertainty.

---

## Overview

This repository contains code for:

1. **Three near-benign attacks** that bypass embedding-based defenses:
   - **Slow Drift** — gradually shifts the attacker's embedding across rounds via "yes, but" pivots
   - **Benign Wrapper** — writes true correct reasoning (the "wrapper") and appends a short malicious directive
   - **Chaos Seeding** — induces benign agents to disagree, widening the benign embedding spread

2. **Confidence-guided defense** using token-level uncertainty:
   - **Confidence Pruning** — block messages with C(m) < δ before delivery to neighbors
   - **Confidence Down-weighting** — prepend `[confidence=C]` to messages; prompt agents to discount low-confidence views

---

## Installation

```bash
conda create -n rethinking-mas python=3.11
conda activate rethinking-mas
pip install -r requirements.txt
```

---

## Data Preparation

Datasets are loaded from **local JSONL files**. Each line must be a JSON object:

```json
{"question": "Which gas makes up the largest portion of Earth's atmosphere?\n\nA. CO2\nB. Nitrogen\nC. Oxygen\nD. Argon\n\nChoose exactly one: A, B, C, or D.", "choices": ["A. CO2", "B. Nitrogen", "C. Oxygen", "D. Argon"], "gold": "B", "subject": "earth_science"}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | Full question text. For MCQA, include the formatted choices. |
| `choices` | List[str] | `["A. ...", "B. ...", ...]`; empty list `[]` for GSM8K. |
| `gold` | str | `"A"`/`"B"`/`"C"`/`"D"` for MCQA; numeric string for GSM8K. |
| `subject` | str | Subject/category tag (used for logging only). |

Place your data files in `data/`, e.g.:
```
data/mmlu_test.jsonl
data/gsm8k_test.jsonl
data/bbh_test.jsonl
```

> **Note on prompts and reproducibility.**
> `src/attacks.py` provides simplified reference templates for the benign agent
> and attacker prompts. 

---

## Quick Start

### Step 1 — Start a vLLM server

```bash
# LLaMA-3.1-8B on GPUs 4-7 (port 8001)
bash scripts/serve_llama.sh

# Qwen3-4B on GPUs 0-3 (port 8002)
bash scripts/serve_qwen.sh
```

For GPT-4o-mini, set `OPENAI_API_KEY` and use `--base_url "" --api_key $OPENAI_API_KEY --model gpt-4o-mini`.

### Step 2 — Run evaluation

```bash
# Clean baseline (no attack, no defense)
python evaluate.py \
    --data_file data/mmlu_test.jsonl --dataset mmlu \
    --model llama3-8b --attack none --defense none

# Slow Drift + Confidence Pruning
python evaluate.py \
    --data_file data/mmlu_test.jsonl --dataset mmlu \
    --model llama3-8b \
    --attack slow_drift --defense pruning --prune_threshold 0.8 \
    --n_attackers 2 --n_agents 5 --n_rounds 3
```

Results are written to `results/` as JSONL files.

### Step 3 — Analyse results

```bash
# Single file stats
python analyze.py results/mmlu_test_llama3-8b_slow_drift_pruning_d0.4_sparse_random.jsonl

# Summary table
python analyze.py results/*.jsonl --summary

# Interactive per-sample viewer
python analyze.py results/mmlu_test_llama3-8b_slow_drift_pruning_d0.4_sparse_random.jsonl --view
```

---

## Citation

```bibtex
@inproceedings{zhang2026embedding,
  title     = {When Embedding-Based Defenses Fail: Rethinking Safety in LLM-Based Multi-Agent Systems},
  author    = {Zhang, Lingxi and Zheng, Guangtao and Chen, Hanjie},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```
