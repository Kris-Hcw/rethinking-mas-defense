"""
Multi-agent debate system with confidence-guided defense.

Core classes:
  Agent        — single LLM agent with logprob-based confidence tracking
  DebateMAS    — orchestrates N agents over R rounds with optional defense

Defense modes (Section 6.2):
  "none"        — no defense
  "pruning"     — block messages with C < prune_threshold before delivery
  "downweight"  — prepend [confidence=C] to each message; prompt agents to discount
                  low-confidence views
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from src.api import AsyncLLMClient, ChatOut
from src.attacks import (
    AttackType, DatasetType,
    get_system_prompt, get_first_prompt, get_regen_prompt, get_postprocessor,
    choose_wrong_target,
)
from src.confidence import compute_confidence
from src.topology import build_adjacency


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<ANSWER>\s*:\s*([A-Z0-9\-]+)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"<REASON>\s*:\s*(.*?)(?:\n\s*<ANSWER>|$)", re.IGNORECASE | re.DOTALL)


def _parse_answer(text: str) -> Optional[str]:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().upper()
    # Fallback: last standalone letter in the tail
    tail = text[-300:]
    letters = re.findall(r"\b([ABCD])\b", tail.upper())
    return letters[-1] if letters else None


def _parse_reason(text: str) -> str:
    m = _REASON_RE.search(text)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"(?i)<ANSWER>\s*:\s*\S+", "", text).strip()
    return re.sub(r"(?i)<REASON>\s*:\s*", "", cleaned).strip()


def _clean_for_views(text: Optional[str], max_chars: int = 300) -> str:
    if not text:
        return ""
    s = re.sub(r"(?i)<REASON>\s*:\s*", "", text)
    s = re.sub(r"(?i)<ANSWER>\s*:\s*\S+", "", s)
    return " ".join(s.split())[:max_chars]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    answer:     Optional[str] = None
    reason:     Optional[str] = None
    confidence: Optional[float] = None   # C(m) = exp(-U(m))
    history:    List[Dict[str, Any]] = field(default_factory=list)


class Agent:
    """Single LLM-backed agent."""

    def __init__(
        self,
        client: AsyncLLMClient,
        model: str,
        system_prompt: str,
        postprocess: Optional[Callable[[str], str]] = None,
        top_k_conf: int = 10,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.postprocess = postprocess
        self.top_k_conf = top_k_conf
        self.state = AgentState()

    def reset(self, system_prompt: str, postprocess: Optional[Callable[[str], str]] = None):
        self.system_prompt = system_prompt
        self.postprocess = postprocess
        self.state = AgentState()

    async def chat(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        out: ChatOut = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=20,
        )

        text = out.text
        if self.postprocess is not None:
            text = self.postprocess(text)

        self.state.answer     = _parse_answer(text)
        self.state.reason     = _parse_reason(text)
        self.state.confidence = compute_confidence(out.token_infos, top_k=self.top_k_conf)
        self.state.history.append({"prompt": prompt, "response": text})
        return text


# ---------------------------------------------------------------------------
# Defense config
# ---------------------------------------------------------------------------

@dataclass
class DefenseConfig:
    mode:            str   = "none"   # "none" | "pruning" | "downweight"
    prune_threshold: float = 0.4      # δ: block messages with C < threshold
    top_k_conf:      int   = 10       # k for confidence computation


# ---------------------------------------------------------------------------
# Attack config
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    attack_type:  AttackType = "none"
    attacker_ids: List[int]  = field(default_factory=list)
    seed:         int        = 0
    temperature:  float      = 0.9
    max_tokens:   int        = 512


# ---------------------------------------------------------------------------
# DebateMAS
# ---------------------------------------------------------------------------

class DebateMAS:
    """
    Multi-agent debate system.

    Each question is processed by calling run_one().  The MAS runs R rounds;
    in each round all agents generate in parallel.  The defense (if enabled)
    filters or annotates peer views based on confidence scores from the
    previous round before building the prompts for the current round.
    """

    def __init__(
        self,
        client:      AsyncLLMClient,
        model:       str,
        n_agents:    int,
        n_rounds:    int,
        dataset:     DatasetType,
        temperature: float = 0.3,
        max_tokens:  int   = 512,
        adj:         Optional[np.ndarray] = None,
        defense:     Optional[DefenseConfig] = None,
    ):
        self.client      = client
        self.model       = model
        self.n_agents    = n_agents
        self.n_rounds    = n_rounds
        self.dataset     = dataset
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.adj         = adj if adj is not None else np.ones((n_agents, n_agents), dtype=int) - np.eye(n_agents, dtype=int)
        self.defense     = defense or DefenseConfig(mode="none")

        top_k = self.defense.top_k_conf
        base_sys = get_system_prompt("none", dataset, None, is_attacker=False)
        self.agents = [
            Agent(client, model, base_sys, top_k_conf=top_k)
            for _ in range(n_agents)
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_one(
        self,
        question: str,
        gold:     Optional[str],
        attack:   Optional[AttackConfig] = None,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        """Run one debate question and return a result dict."""
        attack = attack or AttackConfig(attack_type="none")
        attacker_set = set(attack.attacker_ids)

        # Pick wrong target for attackers
        target: Optional[str] = None
        if attack.attack_type != "none" and attacker_set:
            target = choose_wrong_target(
                gold, self.dataset,
                seed=attack.seed + seed_offset,
            )

        # Reset agents
        for i, ag in enumerate(self.agents):
            is_atk = (i in attacker_set) and (target is not None)
            sys_p = get_system_prompt(attack.attack_type, self.dataset, target, is_atk)
            pp    = get_postprocessor(attack.attack_type, target) if is_atk else None
            ag.reset(sys_p, pp)

        # ---- Round 1 -------------------------------------------------------
        tasks = []
        for i in range(self.n_agents):
            is_atk = (i in attacker_set) and (target is not None)
            prompt = get_first_prompt(attack.attack_type, self.dataset, question, target, is_atk)
            temp   = attack.temperature if is_atk else self.temperature
            tok    = attack.max_tokens  if is_atk else self.max_tokens
            tasks.append(self.agents[i].chat(prompt, temperature=temp, max_tokens=tok))

        await asyncio.gather(*tasks)

        # ---- Rounds 2..R ---------------------------------------------------
        for rid in range(2, self.n_rounds + 1):
            tasks = []
            for i in range(self.n_agents):
                is_atk    = (i in attacker_set) and (target is not None)
                other_views = self._format_views(i, rid)
                prompt = get_regen_prompt(
                    attack.attack_type, self.dataset,
                    question, other_views, rid,
                    target, is_atk,
                    defense_mode=self.defense.mode,
                )
                temp = attack.temperature if is_atk else self.temperature
                tok  = attack.max_tokens  if is_atk else self.max_tokens
                tasks.append(self.agents[i].chat(prompt, temperature=temp, max_tokens=tok))

            await asyncio.gather(*tasks)

        # ---- Aggregate -----------------------------------------------------
        final_answers = [ag.state.answer for ag in self.agents]
        pred, vote_counts = _majority_vote(final_answers)

        return {
            "pred":               pred,
            "gold":               gold,
            "vote_counts":        vote_counts,
            "per_agent_answers":  final_answers,
            "per_agent_conf":     [ag.state.confidence for ag in self.agents],
            "attack_type":        attack.attack_type,
            "attacker_ids":       list(attacker_set),
            "target_wrong":       target,
            "defense_mode":       self.defense.mode,
            "agent_histories":    [ag.state.history for ag in self.agents],
        }

    # ------------------------------------------------------------------
    # View formatting with defense
    # ------------------------------------------------------------------

    def _format_views(self, agent_idx: int, round_id: int) -> str:
        """
        Build the 'other_views' string for agent_idx at round_id.

        Applies confidence-based defense:
          - pruning:    omit messages with C < threshold
          - downweight: prepend [confidence=C] to each message
        """
        in_peers = np.nonzero(self.adj[:, agent_idx])[0].tolist()
        if not in_peers:
            return "(no other views)"

        defense = self.defense
        lines: List[str] = []

        for j in in_peers:
            st  = self.agents[j].state
            ans = st.answer or "?"
            rea = _clean_for_views(st.reason)
            conf = st.confidence  # may be None if logprobs unavailable

            if defense.mode == "pruning":
                # Block message if confidence is below threshold
                if conf is not None and conf < defense.prune_threshold:
                    continue  # withheld

            if defense.mode == "downweight" and conf is not None:
                lines.append(f"- Peer {j} [confidence={conf:.2f}]: {{answer={ans}; reason={rea}}}")
            else:
                lines.append(f"- Peer {j}: {{answer={ans}; reason={rea}}}")

        # Also include the agent's own previous view
        self_st  = self.agents[agent_idx].state
        self_ans = self_st.answer or "?"
        self_rea = _clean_for_views(self_st.reason)
        self_conf = self_st.confidence

        if defense.mode == "downweight" and self_conf is not None:
            lines.append(f"- Peer {agent_idx} (self) [confidence={self_conf:.2f}]: {{answer={self_ans}; reason={self_rea}}}")
        else:
            lines.append(f"- Peer {agent_idx} (self): {{answer={self_ans}; reason={self_rea}}}")

        return "\n".join(lines) if lines else "(no other views — all messages pruned)"


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _majority_vote(
    answers: List[Optional[str]],
) -> tuple:
    counts: Dict[str, int] = {}
    for a in answers:
        if a:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None, {}
    best = max(counts, key=lambda k: counts[k])
    return best, counts
