import math
import json
import unittest
from unittest.mock import patch

from src import llm_client
from src.llm_client import LLMClient


class LLMClientTests(unittest.TestCase):
    def test_qwen3_disables_thinking_without_overriding_explicit_value(self):
        self.assertEqual(
            LLMClient._qwen_extra("qwen3-30b-a3b", {}),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        explicit = {"chat_template_kwargs": {"enable_thinking": True}}
        self.assertTrue(
            LLMClient._qwen_extra("qwen3-30b-a3b", explicit)["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    def test_openai_payload_parses_content_and_top_logprobs(self):
        payload = {
            "choices": [
                {
                    "message": {"content": "OK"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "OK",
                                "logprob": math.log(0.9),
                                "top_logprobs": [
                                    {"token": "OK", "logprob": math.log(0.9)},
                                    {"token": "NO", "logprob": math.log(0.1)},
                                ],
                            }
                        ]
                    },
                }
            ]
        }
        result = LLMClient._parse_chat_payload(payload, require_logprobs=True)
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.token_infos[0].token, "OK")
        self.assertEqual(len(result.token_infos[0].top_logprobs), 2)

    def test_openai_payload_parses_backend_reduced_full_vocab_entropy(self):
        payload = {
            "choices": [
                {
                    "message": {"content": "OK"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "OK",
                                "logprob": math.log(0.9),
                                "top_logprobs": [],
                                "full_vocab_entropy": 1.2345,
                                "full_vocab_size": 151936,
                                "full_vocab_entropy_source": "raw_generation_logits",
                            }
                        ]
                    },
                }
            ]
        }
        result = LLMClient._parse_chat_payload(payload, require_logprobs=True)
        self.assertAlmostEqual(result.token_infos[0].full_vocab_entropy, 1.2345)
        self.assertEqual(result.token_infos[0].full_vocab_size, 151936)
        self.assertEqual(
            result.token_infos[0].full_vocab_entropy_source,
            "raw_generation_logits",
        )

    def test_formal_exact_entropy_rejects_token_contract_mismatch(self):
        expected = {
            "backend_revision": "backend-source-sha",
            "backend_manifest_sha256": "manifest-sha",
            "model_revision": "model-sha",
            "sampling_top_k": 20,
            "full_vocab_size": 151936,
            "full_vocab_entropy_source": "raw_generation_logits",
        }
        client = LLMClient(
            transport="ssh",
            expected_backend_metadata=expected,
            require_backend_metadata=True,
        )
        payload = {
            "choices": [{
                "message": {"content": "OK"},
                "logprobs": {"content": [{
                    "token": "OK",
                    "logprob": math.log(0.9),
                    "top_logprobs": [],
                    "full_vocab_entropy": 1.2345,
                    "full_vocab_size": 32000,
                    "full_vocab_entropy_source": "raw_generation_logits",
                }]},
            }],
            "backend_metadata": dict(expected),
        }
        with self.assertRaises(llm_client.BackendIdentityError):
            client._parse_and_validate(payload, require_logprobs=True)

    def test_openai_payload_preserves_backend_metadata(self):
        payload = {
            "choices": [{"message": {"content": "OK"}}],
            "backend_metadata": {
                "backend_revision": "backend-source-sha",
                "backend_manifest_sha256": "manifest-sha",
                "model_revision": "model-sha",
                "sampling_top_k": 20,
            },
        }
        result = LLMClient._parse_chat_payload(payload, require_logprobs=False)
        self.assertEqual(result.backend_metadata, payload["backend_metadata"])

    def test_expected_backend_metadata_rejects_mismatch(self):
        expected = {
            "backend_revision": "backend-source-sha",
            "backend_manifest_sha256": "manifest-sha",
            "model_revision": "model-sha",
            "sampling_top_k": 20,
        }
        client = LLMClient(
            transport="ssh",
            expected_backend_metadata=expected,
            require_backend_metadata=True,
        )
        payload = {
            "choices": [{"message": {"content": "OK"}}],
            "backend_metadata": {**expected, "sampling_top_k": 40},
        }
        with self.assertRaises(llm_client.BackendIdentityError):
            client._parse_and_validate(payload, require_logprobs=False)

    def test_expected_backend_metadata_accepts_exact_match(self):
        expected = {
            "backend_revision": "backend-source-sha",
            "backend_manifest_sha256": "manifest-sha",
            "model_revision": "model-sha",
            "sampling_top_k": 20,
        }
        client = LLMClient(
            transport="ssh",
            expected_backend_metadata=expected,
            require_backend_metadata=True,
        )
        payload = {
            "choices": [{"message": {"content": "OK"}}],
            "backend_metadata": dict(expected),
        }
        result = client._parse_and_validate(payload, require_logprobs=False)
        self.assertEqual(result.backend_metadata, expected)

    def test_ssh_transport_sends_json_on_stdin_and_parses_content(self):
        captured = {}

        class Channel:
            def shutdown_write(self):
                pass

            def recv_exit_status(self):
                return 0

        class Input:
            def __init__(self):
                self.channel = Channel()

            def write(self, value):
                captured["body"] = value

            def flush(self):
                pass

        class Output:
            def __init__(self, value):
                self.value = value
                self.channel = Channel()

            def read(self):
                return self.value

        class SSH:
            def exec_command(self, command, timeout):
                captured["command"] = command
                return (
                    Input(),
                    Output(b'{"choices":[{"message":{"content":"SSH_OK"}}]}'),
                    Output(b""),
                )

        with patch.dict(
            "os.environ",
            {
                "GSAFEGUARD_REMOTE_LLM_HOST": "127.0.0.1",
                "GSAFEGUARD_REMOTE_LLM_PORT": "8002",
            },
            clear=False,
        ):
            client = LLMClient(transport="ssh", model="qwen3-30b-a3b")
            client._ssh_client = lambda: SSH()
            output = client.chat(
                messages=[{"role": "user", "content": "local prompt"}],
                logprobs=False,
                seed=12345,
            )

        self.assertEqual(output.text, "SSH_OK")
        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(body["messages"][0]["content"], "local prompt")
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["seed"], 12345)
        self.assertIn("127.0.0.1:8002/v1/chat/completions", captured["command"])
        self.assertIn("--data-binary", captured["command"])

    def test_token_count_uses_serving_model_tokenizer_contract(self):
        expected = {
            "backend_revision": "backend-source-sha",
            "model_revision": "model-sha",
        }
        client = LLMClient(
            transport="ssh",
            model="qwen3-4b",
            expected_backend_metadata=expected,
            require_backend_metadata=True,
        )
        client._ssh_json = lambda method, endpoint, payload: {
            "object": "tokenization",
            "model": "qwen3-4b",
            "token_count": 7,
            "add_special_tokens": False,
            "backend_metadata": dict(expected),
        }

        self.assertEqual(client.count_tokens("payload B"), 7)

    def test_token_count_rejects_wrong_model_or_special_token_policy(self):
        client = LLMClient(transport="ssh", model="qwen3-4b")
        with self.assertRaises(llm_client.BackendIdentityError):
            client._parse_and_validate_token_count(
                {
                    "model": "other-model",
                    "token_count": 2,
                    "add_special_tokens": False,
                },
                expected_model="qwen3-4b",
            )
        with self.assertRaises(llm_client.LLMTransportError):
            client._parse_and_validate_token_count(
                {
                    "model": "qwen3-4b",
                    "token_count": 2,
                    "add_special_tokens": True,
                },
                expected_model="qwen3-4b",
            )


if __name__ == "__main__":
    unittest.main()
