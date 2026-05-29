"""
Async LLM client compatible with vLLM (open-weight models) and OpenAI API (GPT).

Both backends expose the same OpenAI-compatible /v1/chat/completions endpoint,
so the same client works for both — only base_url and api_key differ.
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


@dataclass
class TokenInfo:
    token: str
    logprob: float                       # log P(token)
    top_logprobs: List[tuple]            # [(token_str, logprob), ...]


@dataclass
class ChatOut:
    text: str
    token_infos: List[TokenInfo] = field(default_factory=list)


class AsyncLLMClient:
    """
    Thin async wrapper around AsyncOpenAI.

    Works for:
      - Local vLLM server: base_url="http://localhost:8001/v1", api_key="EMPTY"
      - OpenAI API:        base_url=None (default), api_key=<real key>
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        max_concurrency: int = 64,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout = timeout
        self._max_retries = max_retries

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 20,
    ) -> ChatOut:
        """Single chat completion. Returns text + per-token logprob info."""
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._sem:
                    resp = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            logprobs=logprobs,
                            top_logprobs=top_logprobs if logprobs else None,
                        ),
                        timeout=self._timeout,
                    )
                ch = resp.choices[0]
                text = (ch.message.content or "").strip()
                infos = self._extract_token_infos(ch) if logprobs else []
                return ChatOut(text=text, token_infos=infos)

            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    await asyncio.sleep(0.3 * (2 ** attempt))
                else:
                    raise
        raise last_err  # unreachable, but satisfies type checkers

    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    @staticmethod
    def _extract_token_infos(choice: Any) -> List[TokenInfo]:
        lp_obj = getattr(choice, "logprobs", None)
        content = getattr(lp_obj, "content", None) if lp_obj else None
        if not content:
            return []

        infos: List[TokenInfo] = []
        for item in content:
            token = getattr(item, "token", "") or ""
            logprob = AsyncLLMClient._safe_float(getattr(item, "logprob", float("nan")))
            top: List[tuple] = []
            for t in (getattr(item, "top_logprobs", None) or []):
                t_tok = getattr(t, "token", "") or ""
                t_lp = AsyncLLMClient._safe_float(getattr(t, "logprob", float("nan")))
                if not math.isnan(t_lp):
                    top.append((t_tok, t_lp))
            infos.append(TokenInfo(token=token, logprob=logprob, top_logprobs=top))
        return infos
