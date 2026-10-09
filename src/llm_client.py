"""Unified direct/SSH client for OpenAI-compatible LLM inference.

All datasets, embeddings, metrics, and defenses remain local. In ``ssh`` mode
this client runs curl against the remote vLLM loopback API over an SSH channel;
it never uses SFTP/SCP and never uploads project files.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import AsyncOpenAI, OpenAI


class LogprobsUnavailableError(RuntimeError):
    """The backend cannot expose distributions required by confidence defense."""


class LLMTransportError(RuntimeError):
    """Direct HTTP or SSH transport failed."""


class BackendIdentityError(LLMTransportError):
    """The responding backend does not match the preregistered runtime identity."""


@dataclass
class TokenInfo:
    token: str
    logprob: float
    top_logprobs: List[tuple]
    # OpenAI-compatible APIs normally return only a truncated top-k list.  A
    # backend may opt in to the custom exhaustive flag when it returns the
    # complete vocabulary distribution for this token.
    top_logprobs_exhaustive: bool = False
    # A trusted loopback backend may reduce the complete vocabulary logits on
    # the GPU and return only the resulting Shannon entropy.  This avoids
    # serialising O(output_length * vocab_size) logprob entries while retaining
    # the paper's exact full-vocabulary statistic.
    full_vocab_entropy: Optional[float] = None
    full_vocab_size: Optional[int] = None
    full_vocab_entropy_source: Optional[str] = None


@dataclass
class ChatOut:
    text: str
    token_infos: List[TokenInfo] = field(default_factory=list)
    backend_metadata: Dict[str, object] = field(default_factory=dict)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


class LLMClient:
    """One business-facing client supporting direct HTTP and SSH transports."""

    def __init__(
        self,
        transport: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        max_concurrency: int = 64,
        expected_backend_metadata: Optional[Dict[str, object]] = None,
        require_backend_metadata: bool = False,
        disable_thinking: bool = False,
        payload_tokenizer_path: Optional[str] = None,
    ):
        self.transport = (transport or os.environ.get("GSAFEGUARD_LLM_TRANSPORT", "direct")).lower()
        if self.transport not in {"direct", "ssh"}:
            raise ValueError("GSAFEGUARD_LLM_TRANSPORT must be 'direct' or 'ssh'.")
        if base_url is not None:
            # An explicit empty string selects the OpenAI SDK default endpoint.
            self.base_url = base_url or None
        else:
            self.base_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("BASE_URL")
                or "http://localhost:8001/v1"
            )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.environ.get("MODEL_TYPE", "llama3-8b")
        self.timeout = timeout
        self.max_retries = max_retries
        self._max_concurrency = max_concurrency
        self.expected_backend_metadata = dict(expected_backend_metadata or {})
        self.disable_thinking = disable_thinking
        self.payload_tokenizer_path = payload_tokenizer_path
        self._payload_tokenizer = None
        self.require_backend_metadata = bool(require_backend_metadata)
        if payload_tokenizer_path and self.require_backend_metadata:
            raise ValueError("Local payload tokenizer is diagnostic-only; formal counts require the serving backend.")
        self._async_sem: Optional[asyncio.Semaphore] = None
        self._ssh_thread_local = threading.local()
        self._detected_ssh_alias: Optional[str] = None
        self._sync_direct: Optional[OpenAI] = None
        self._async_direct: Optional[AsyncOpenAI] = None
        if self.transport == "direct":
            self._sync_direct = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._async_direct = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=timeout,
                max_retries=max_retries,
            )

    def _effective_model(self, model: Optional[str]) -> str:
        return model or self.model

    def _semaphore(self) -> asyncio.Semaphore:
        if self._async_sem is None:
            self._async_sem = asyncio.Semaphore(self._max_concurrency)
        return self._async_sem

    @staticmethod
    def _qwen_extra(model: str, extra_body: Optional[dict]) -> dict:
        merged = dict(extra_body or {})
        if "qwen3" in model.lower():
            template = dict(merged.get("chat_template_kwargs") or {})
            template.setdefault("enable_thinking", False)
            merged["chat_template_kwargs"] = template
        return merged

    def _request_extra(self, model: str, extra_body: Optional[dict]) -> dict:
        merged = self._qwen_extra(model, extra_body)
        host = urlparse(self.base_url or "").hostname or ""
        is_dashscope = host == "dashscope.aliyuncs.com" or host.endswith(".dashscope.aliyuncs.com")
        if is_dashscope:
            template = dict(merged.get("chat_template_kwargs") or {})
            if "enable_thinking" in template:
                merged.setdefault("enable_thinking", template.pop("enable_thinking"))
            if template:
                merged["chat_template_kwargs"] = template
            else:
                merged.pop("chat_template_kwargs", None)
        if self.disable_thinking:
            if is_dashscope:
                merged["enable_thinking"] = False
            elif "chat_template_kwargs" in merged:
                merged["chat_template_kwargs"]["enable_thinking"] = False
            else:
                merged["enable_thinking"] = False
        return merged

    def _local_payload_token_count(self, text: str) -> Optional[int]:
        if not self.payload_tokenizer_path:
            return None
        if self._payload_tokenizer is None:
            from transformers import AutoTokenizer
            self._payload_tokenizer = AutoTokenizer.from_pretrained(
                self.payload_tokenizer_path, local_files_only=True, trust_remote_code=False,
            )
        return len(self._payload_tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _parse_chat_payload(payload: dict, require_logprobs: bool) -> ChatOut:
        try:
            choice = payload["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMTransportError(
                f"Invalid OpenAI-compatible chat response: {json.dumps(payload)[:1000]}"
            ) from exc

        token_infos: List[TokenInfo] = []
        for item in (choice.get("logprobs") or {}).get("content") or []:
            top = []
            for candidate in item.get("top_logprobs") or []:
                try:
                    value = float(candidate.get("logprob"))
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    top.append((candidate.get("token") or "", value))
            try:
                chosen_logprob = float(item.get("logprob"))
            except (TypeError, ValueError):
                chosen_logprob = float("nan")
            try:
                full_vocab_entropy = float(item.get("full_vocab_entropy"))
            except (TypeError, ValueError):
                full_vocab_entropy = None
            if full_vocab_entropy is not None and (
                math.isnan(full_vocab_entropy) or full_vocab_entropy < 0.0
            ):
                full_vocab_entropy = None
            try:
                full_vocab_size = int(item.get("full_vocab_size"))
            except (TypeError, ValueError):
                full_vocab_size = None
            if full_vocab_size is not None and full_vocab_size <= 0:
                full_vocab_size = None
            raw_entropy_source = item.get("full_vocab_entropy_source")
            full_vocab_entropy_source = (
                raw_entropy_source.strip()
                if isinstance(raw_entropy_source, str) and raw_entropy_source.strip()
                else None
            )
            token_infos.append(
                TokenInfo(
                    item.get("token") or "",
                    chosen_logprob,
                    top,
                    bool(item.get("top_logprobs_exhaustive", False)),
                    full_vocab_entropy,
                    full_vocab_size,
                    full_vocab_entropy_source,
                )
            )

        if require_logprobs and not token_infos:
            raise LogprobsUnavailableError(
                "The vLLM/OpenAI-compatible backend returned no token logprobs. "
                "Confidence-guided defense requires logprobs=True."
            )
        if require_logprobs and not any(
            info.top_logprobs or info.full_vocab_entropy is not None
            for info in token_infos
        ):
            raise LogprobsUnavailableError(
                "The backend returned neither top_logprobs distributions nor "
                "full-vocabulary entropy, so token entropy cannot be computed."
            )
        raw_backend_metadata = payload.get("backend_metadata")
        backend_metadata = (
            dict(raw_backend_metadata) if isinstance(raw_backend_metadata, dict) else {}
        )
        return ChatOut(
            text=text,
            token_infos=token_infos,
            backend_metadata=backend_metadata,
        )

    def _parse_and_validate(self, payload: dict, require_logprobs: bool) -> ChatOut:
        result = self._parse_chat_payload(payload, require_logprobs=require_logprobs)
        if not self.require_backend_metadata:
            return result
        if not result.backend_metadata:
            raise BackendIdentityError(
                "Formal backend response omitted required backend_metadata."
            )
        mismatches = {
            key: {"expected": expected, "actual": result.backend_metadata.get(key)}
            for key, expected in self.expected_backend_metadata.items()
            if result.backend_metadata.get(key) != expected
        }
        if mismatches:
            raise BackendIdentityError(
                "Formal backend identity mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        expected_size = self.expected_backend_metadata.get("full_vocab_size")
        expected_source = self.expected_backend_metadata.get(
            "full_vocab_entropy_source"
        )
        if require_logprobs and (expected_size is not None or expected_source is not None):
            token_mismatches = []
            for index, info in enumerate(result.token_infos):
                if (
                    info.full_vocab_entropy is None
                    or info.full_vocab_size != expected_size
                    or info.full_vocab_entropy_source != expected_source
                ):
                    token_mismatches.append(
                        {
                            "token_index": index,
                            "full_vocab_entropy_present": info.full_vocab_entropy is not None,
                            "full_vocab_size": info.full_vocab_size,
                            "full_vocab_entropy_source": info.full_vocab_entropy_source,
                        }
                    )
            if token_mismatches:
                raise BackendIdentityError(
                    "Formal exact-entropy token metadata mismatch: "
                    + json.dumps(token_mismatches[:5], ensure_ascii=False, sort_keys=True)
                )
        return result

    def _parse_and_validate_token_count(
        self, payload: dict, *, expected_model: str
    ) -> int:
        if not isinstance(payload, dict):
            raise LLMTransportError("Tokenizer endpoint returned a non-object response.")
        if payload.get("model") != expected_model:
            raise BackendIdentityError(
                "Tokenizer endpoint model identity mismatch: "
                f"expected {expected_model!r}, got {payload.get('model')!r}."
            )
        if payload.get("add_special_tokens") is not False:
            raise LLMTransportError(
                "Tokenizer endpoint must count payload tokens with add_special_tokens=false."
            )
        raw_count = payload.get("token_count")
        if isinstance(raw_count, bool):
            raise LLMTransportError("Tokenizer endpoint returned an invalid token_count.")
        try:
            token_count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise LLMTransportError(
                "Tokenizer endpoint returned an invalid token_count."
            ) from exc
        if token_count < 0:
            raise LLMTransportError("Tokenizer endpoint returned a negative token_count.")
        if self.require_backend_metadata:
            metadata = payload.get("backend_metadata")
            if not isinstance(metadata, dict):
                raise BackendIdentityError(
                    "Formal tokenizer response omitted required backend_metadata."
                )
            mismatches = {
                key: {"expected": expected, "actual": metadata.get(key)}
                for key, expected in self.expected_backend_metadata.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise BackendIdentityError(
                    "Formal tokenizer backend identity mismatch: "
                    + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
                )
        return token_count

    def _chat_request(
        self,
        model: Optional[str],
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        logprobs: bool,
        top_logprobs: int,
        extra_body: Optional[dict],
        seed: Optional[int],
    ) -> tuple[str, dict]:
        selected_model = self._effective_model(model)
        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        if logprobs:
            payload["top_logprobs"] = top_logprobs
        if seed is not None:
            payload["seed"] = int(seed)
        payload.update(self._request_extra(selected_model, extra_body))
        return selected_model, payload

    def chat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 20,
        extra_body: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> ChatOut:
        if messages is None:
            raise ValueError("messages is required.")
        selected_model, payload = self._chat_request(
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            logprobs,
            top_logprobs,
            extra_body,
            seed,
        )
        if self.transport == "ssh":
            response = self._ssh_json("POST", "/v1/chat/completions", payload)
        else:
            assert self._sync_direct is not None
            response_obj = self._sync_direct.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs if logprobs else None,
                seed=seed,
                extra_body=self._request_extra(selected_model, extra_body) or None,
            )
            response = response_obj.model_dump()
        return self._parse_and_validate(response, require_logprobs=logprobs)

    async def achat(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 20,
        extra_body: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> ChatOut:
        if messages is None:
            raise ValueError("messages is required.")
        if self.transport == "ssh":
            async with self._semaphore():
                return await asyncio.to_thread(
                    self.chat,
                    model,
                    messages,
                    temperature,
                    top_p,
                    max_tokens,
                    logprobs,
                    top_logprobs,
                    extra_body,
                    seed,
                )

        selected_model = self._effective_model(model)
        assert self._async_direct is not None
        async with self._semaphore():
            response_obj = await self._async_direct.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs if logprobs else None,
                seed=seed,
                extra_body=self._request_extra(selected_model, extra_body) or None,
            )
        return self._parse_and_validate(
            response_obj.model_dump(), require_logprobs=logprobs
        )

    def chat_content(self, *args, **kwargs) -> str:
        return self.chat(*args, **kwargs).text

    async def achat_content(self, *args, **kwargs) -> str:
        return (await self.achat(*args, **kwargs)).text

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count via the serving backend, or an explicitly selected diagnostic tokenizer."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        local_count = self._local_payload_token_count(text)
        if local_count is not None:
            return local_count
        selected_model = self._effective_model(model)
        request = {"model": selected_model, "text": text, "add_special_tokens": False}
        if self.transport == "ssh":
            payload = self._ssh_json("POST", "/tokenize", request)
        else:
            assert self._sync_direct is not None
            payload = self._sync_direct.post(
                "/tokenize", cast_to=dict, body=request
            )
        return self._parse_and_validate_token_count(
            payload, expected_model=selected_model
        )

    async def acount_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Async counterpart of :meth:`count_tokens`."""

        if self.payload_tokenizer_path:
            return self.count_tokens(text, model=model)
        if self.transport == "ssh":
            async with self._semaphore():
                return await asyncio.to_thread(self.count_tokens, text, model)
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        selected_model = self._effective_model(model)
        request = {"model": selected_model, "text": text, "add_special_tokens": False}
        assert self._async_direct is not None
        async with self._semaphore():
            payload = await self._async_direct.post(
                "/tokenize", cast_to=dict, body=request
            )
        return self._parse_and_validate_token_count(
            payload, expected_model=selected_model
        )

    def list_models(self) -> List[str]:
        if self.transport == "ssh":
            payload = self._ssh_json("GET", "/v1/models", None)
            return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
        assert self._sync_direct is not None
        return [item.id for item in self._sync_direct.models.list().data]

    async def alist_models(self) -> List[str]:
        if self.transport == "ssh":
            return await asyncio.to_thread(self.list_models)
        assert self._async_direct is not None
        response = await self._async_direct.models.list()
        return [item.id for item in response.data]

    def _ssh_client(self):
        try:
            import paramiko
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SSH transport requires paramiko. Install project requirements first."
            ) from exc

        client = getattr(self._ssh_thread_local, "client", None)
        transport = client.get_transport() if client is not None else None
        if transport is not None and transport.is_active():
            return client

        host_alias = (
            os.environ.get("GSAFEGUARD_SSH_HOST")
            or os.environ.get("GSAFEGUARD_SSH_ALIAS")
            or self._detect_vllm_ssh_alias()
        )
        ssh_config = self._ssh_config_lookup(paramiko, host_alias)
        host = ssh_config.get("hostname") or host_alias
        username = os.environ.get("GSAFEGUARD_SSH_USERNAME") or ssh_config.get("user")
        if not host or not username:
            raise LLMTransportError(
                "ssh transport requires GSAFEGUARD_SSH_HOST/GSAFEGUARD_SSH_ALIAS and "
                "GSAFEGUARD_SSH_USERNAME, or a matching Host entry in ~/.ssh/config."
            )
        port = int(os.environ.get("GSAFEGUARD_SSH_PORT") or ssh_config.get("port") or "22")
        identity_files = ssh_config.get("identityfile") or []
        if isinstance(identity_files, str):
            identity_files = [identity_files]
        key_path = os.path.expanduser(
            os.environ.get("GSAFEGUARD_SSH_PRIVATE_KEY")
            or (identity_files[0] if identity_files else "~/.ssh/id_ed25519")
        )
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        if _env_bool("GSAFEGUARD_SSH_AUTO_ADD_HOST_KEY", False):
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            port=port,
            username=username,
            key_filename=key_path,
            timeout=min(self.timeout, 30.0),
            banner_timeout=min(self.timeout, 30.0),
            auth_timeout=min(self.timeout, 30.0),
        )
        self._ssh_thread_local.client = client
        return client

    @staticmethod
    def _ssh_config_lookup(paramiko_module, host_alias: str) -> dict:
        path = Path.home() / ".ssh" / "config"
        if not path.exists():
            return {}
        config = paramiko_module.SSHConfig()
        with path.open(encoding="utf-8", errors="replace") as stream:
            config.parse(stream)
        return config.lookup(host_alias) or {}

    @staticmethod
    def _ssh_config_hosts() -> List[str]:
        path = Path.home() / ".ssh" / "config"
        if not path.exists():
            return []
        hosts: List[str] = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == "host":
                for alias in parts[1:]:
                    if "*" not in alias and "?" not in alias:
                        hosts.append(alias)
        return hosts

    def _detect_vllm_ssh_alias(self) -> str:
        if self._detected_ssh_alias:
            return self._detected_ssh_alias

        remote_host = os.environ.get("GSAFEGUARD_REMOTE_LLM_HOST", "127.0.0.1")
        remote_port = int(os.environ.get("GSAFEGUARD_REMOTE_LLM_PORT", "8002"))
        endpoint = f"http://{remote_host}:{remote_port}/v1/models"
        model = self._effective_model(None)
        for alias in self._ssh_config_hosts():
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                alias,
                f"curl -sS --max-time 8 {shlex.quote(endpoint)}",
            ]
            try:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode != 0 or not result.stdout.strip():
                continue
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
            ids = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
            if model in ids or ids:
                self._detected_ssh_alias = alias
                return alias
        raise LLMTransportError(
            "ssh transport could not auto-detect a reachable vLLM Host from ~/.ssh/config. "
            "Set GSAFEGUARD_SSH_ALIAS or GSAFEGUARD_SSH_HOST explicitly."
        )

    def _ssh_json(self, method: str, endpoint: str, payload: Optional[dict]) -> dict:
        remote_host = os.environ.get("GSAFEGUARD_REMOTE_LLM_HOST", "127.0.0.1")
        remote_port = int(os.environ.get("GSAFEGUARD_REMOTE_LLM_PORT", "8002"))
        url = f"http://{remote_host}:{remote_port}{endpoint}"
        command_parts = [
            "curl", "-sS", "--fail-with-body", "--max-time", str(int(self.timeout)),
            "-X", method, url, "-H", "Content-Type: application/json",
        ]
        if self.api_key and self.api_key != "EMPTY":
            command_parts.extend(["-H", f"Authorization: Bearer {self.api_key}"])
        if payload is not None:
            command_parts.extend(["--data-binary", "@-"])
        command = " ".join(shlex.quote(part) for part in command_parts)

        stdin, stdout, stderr = self._ssh_client().exec_command(command, timeout=self.timeout)
        if payload is not None:
            stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            stdin.flush()
        stdin.channel.shutdown_write()
        raw = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            raise LLMTransportError(
                f"Remote vLLM curl failed with exit code {exit_code}: {error or raw[:1000]}"
            )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMTransportError(f"Remote vLLM returned invalid JSON: {raw[:1000]}") from exc


_DEFAULT_CLIENT: Optional[LLMClient] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_client() -> LLMClient:
    global _DEFAULT_CLIENT
    with _DEFAULT_LOCK:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = LLMClient()
        return _DEFAULT_CLIENT


def chat_content(*args, **kwargs) -> str:
    return get_default_client().chat_content(*args, **kwargs)


async def achat_content(*args, **kwargs) -> str:
    return await get_default_client().achat_content(*args, **kwargs)


AsyncLLMClient = LLMClient
