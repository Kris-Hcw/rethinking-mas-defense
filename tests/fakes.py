"""Deterministic test doubles; never used by the reproduction CLI."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np

from src.llm_client import ChatOut, TokenInfo


class FakeEncoder:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % self.dim] += 1.0
            if not np.any(vector):
                vector[0] = 1.0
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows)


class FakeLLMClient:
    """Produces well-formed answers with usable token logprobs."""

    def __init__(self):
        self.calls = 0
        self.logprobs_requested = []
        self.seeds = []

    @staticmethod
    def _target(system: str, prompt: str):
        patterns = [
            r"eventual conclusion should be option ([A-D])",
            r"final conclusion should be option ([A-D])",
            r"leads to option ([A-D])",
            r"toward option ([A-D])",
            r"always advocate the target option.*?<ANSWER>: ([A-D])",
            r"supporting the user's answer.*?<ANSWER>: ([A-D])",
        ]
        joined = system + "\n" + prompt
        for pattern in patterns:
            match = re.search(pattern, joined, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).upper()
        answer_tags = re.findall(r"<ANSWER>:\s*([A-D])", prompt, re.IGNORECASE)
        return answer_tags[-1].upper() if answer_tags else None

    async def achat(
        self,
        model,
        messages,
        temperature=0.0,
        max_tokens=512,
        logprobs=True,
        top_logprobs=20,
        seed=None,
    ):
        self.calls += 1
        self.logprobs_requested.append(bool(logprobs))
        self.seeds.append(seed)
        system = messages[0]["content"]
        prompt = messages[-1]["content"]
        target = self._target(system, prompt)

        if target and "paper's Benign Wrapper message" in system:
            text = (
                "The conventional calculation and the central definition support B."
                f" || A narrower reading may support {target}.\n<ANSWER>: {target}"
            )
            answer = target
        elif target and "domain-plausible alternative" in system:
            variants = ["scope", "terminology", "boundary condition", "implicit assumption"]
            reason = (
                "The standard interpretation supports B, although the evidence is not unique. "
                f"Under a different {variants[self.calls % len(variants)]}, option {target} is plausible."
            )
            answer = target
        elif target:
            reason = (
                "The standard interpretation initially favors B and shares the peers' main observation. "
                f"A small distinction in scope makes {target} a plausible final reading."
            )
            answer = target
        else:
            reason = "The standard definition directly supports B after checking the relevant evidence."
            answer = "B"

        if not (target and "paper's Benign Wrapper message" in system):
            text = f"<REASON>: {reason}\n<ANSWER>: {answer}"
        distribution = [
            (" likely", math.log(0.72)),
            (" maybe", math.log(0.18)),
            (" other", math.log(0.10)),
        ]
        infos = [
            TokenInfo(token=token, logprob=math.log(0.72), top_logprobs=distribution)
            for token in text.split()
        ]
        return ChatOut(text=text, token_infos=infos)

    async def acount_tokens(self, text, model=None):
        # Deterministic tokenizer stand-in used only by local tests.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
