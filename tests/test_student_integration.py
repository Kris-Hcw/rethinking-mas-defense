"""Offline compatibility checks for the main-to-student port; no model downloads."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import analyze
import evaluate
from src.llm_client import LLMClient
from src.message_classifier import extract_message_samples
from tests.fakes import FakeEncoder, FakeLLMClient


class StudentClientCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_http_request_body_has_top_level_thinking_flag(self):
        import httpx
        import openai

        bodies = []

        def respond(request):
            self.assertEqual(str(request.url), "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion",
                "model": "qwen3.5-35b-a3b", "created": 0,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "B"},
                             "logprobs": {"content": [{"token": "B", "logprob": -0.1,
                                          "top_logprobs": [{"token": "B", "logprob": -0.1}]}]}}]})

        base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        sync_sdk = openai.OpenAI(api_key="offline-test-key", base_url=base,
                                 http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        async_sdk = openai.AsyncOpenAI(api_key="offline-test-key", base_url=base,
                                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        try:
            with patch("src.llm_client.OpenAI", return_value=sync_sdk), patch("src.llm_client.AsyncOpenAI", return_value=async_sdk):
                client = LLMClient(transport="direct", base_url=base, model="qwen3.5-35b-a3b", disable_thinking=True)
                kwargs = {"messages": [{"role": "user", "content": "question"}], "top_logprobs": 5, "seed": 42}
                self.assertEqual(client.chat(**kwargs).text, "B")
                self.assertEqual((await client.achat(**kwargs)).text, "B")
        finally:
            sync_sdk.close()
            await async_sdk.close()
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0], bodies[1])
        for body in bodies:
            self.assertIs(body["enable_thinking"], False)
            self.assertNotIn("chat_template_kwargs", body)
            self.assertEqual(body["top_logprobs"], 5)
            self.assertEqual(body["seed"], 42)

    async def test_actual_direct_sync_and_async_requests_use_same_backend_parameters(self):
        response = Mock()
        response.model_dump.return_value = {"choices": [{"message": {"content": "<ANSWER>: B"}}]}
        for base, expected in (
            ("https://dashscope.aliyuncs.com/compatible-mode/v1", {"enable_thinking": False}),
            ("http://localhost:8001/v1", {"chat_template_kwargs": {"enable_thinking": False}}),
        ):
            sync_sdk, async_sdk = Mock(), Mock()
            sync_sdk.chat.completions.create.return_value = response
            async_sdk.chat.completions.create = AsyncMock(return_value=response)
            with patch("src.llm_client.OpenAI", return_value=sync_sdk), patch("src.llm_client.AsyncOpenAI", return_value=async_sdk):
                client = LLMClient(transport="direct", base_url=base, model="qwen3.5-35b-a3b", disable_thinking=True)
                kwargs = {"messages": [{"role": "user", "content": "question"}], "logprobs": False, "seed": 42}
                client.chat(**kwargs)
                await client.achat(**kwargs)
            self.assertEqual(sync_sdk.chat.completions.create.call_args.kwargs["extra_body"], expected)
            self.assertEqual(async_sdk.chat.completions.create.await_args.kwargs["extra_body"], expected)
            self.assertEqual(sync_sdk.chat.completions.create.call_args.kwargs,
                             async_sdk.chat.completions.create.await_args.kwargs)

    def test_backend_specific_thinking_and_uncapped_logprobs(self):
        for base, expected in (
            ("http://localhost:8001/v1", {"chat_template_kwargs": {"enable_thinking": False}}),
            ("https://dashscope.aliyuncs.com/compatible-mode/v1", {"enable_thinking": False}),
        ):
            client = LLMClient(transport="ssh", base_url=base, model="qwen3.5-35b-a3b")
            _, request = client._chat_request(None, [], 0.3, 1.0, 512, True, 20, None, 42)
            self.assertEqual(request["top_logprobs"], 20)
            self.assertEqual(request["seed"], 42)
            for key, value in expected.items():
                self.assertEqual(request[key], value)

    def test_explicit_thinking_override_and_other_template_fields(self):
        client = LLMClient(transport="ssh", base_url="https://dashscope.aliyuncs.com/v1",
                           model="qwen3", disable_thinking=True)
        extra = {"chat_template_kwargs": {"enable_thinking": True, "other": "keep"}}
        request = client._request_extra("qwen3", extra)
        self.assertEqual(request, {"enable_thinking": False, "chat_template_kwargs": {"other": "keep"}})
        self.assertTrue(extra["chat_template_kwargs"]["enable_thinking"])

    async def test_local_count_is_explicit_lazy_and_offline(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [11, 22, 33]
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        client = LLMClient(transport="ssh", payload_tokenizer_path="/local/tokenizer")
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}):
            self.assertEqual(client.count_tokens("payload"), 3)
            self.assertEqual(await client.acount_tokens("payload"), 3)
        auto.from_pretrained.assert_called_once_with("/local/tokenizer", local_files_only=True, trust_remote_code=False)
        tokenizer.encode.assert_called_with("payload", add_special_tokens=False)

    def test_formal_client_cannot_use_unverified_local_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            LLMClient(transport="ssh", payload_tokenizer_path="/local", require_backend_metadata=True)


class StudentEvaluatorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_args(self, root, extra=()):
        return evaluate.parse_args([
            "--data_file", str(root / "data.jsonl"), "--dataset", "mmlu",
            "--out_file", str(root / "result.jsonl"), "--checkpoint_dir", str(root / "checkpoints"),
            "--n_agents", "3", "--n_attackers", "1", "--n_rounds", "2",
            "--topology", "full", *extra,
        ])

    async def test_existing_outputs_rejected_before_encoder_or_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root)
            Path(args.out_file).write_text("historical result\n")
            with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "SentenceTransformerEncoder") as encoder, patch.object(evaluate, "LLMClient") as client:
                with self.assertRaises(FileExistsError):
                    await evaluate.main()
            encoder.assert_not_called()
            client.assert_not_called()
            self.assertEqual(Path(args.out_file).read_text(), "historical result\n")

    async def test_dataset_collision_is_rejected_even_with_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, ["--out_file", str(root / "data.jsonl"), "--resume"])
            Path(args.data_file).write_text("source dataset\n")
            with patch.object(evaluate, "parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "input dataset"):
                    await evaluate.main()
            self.assertEqual(Path(args.data_file).read_text(), "source dataset\n")

    async def test_duplicate_report_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, ["--classify_messages", "--classification_output", str(root / "result.jsonl")])
            with patch.object(evaluate, "parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    await evaluate.main()

    async def test_tokenizer_input_file_cannot_be_used_as_an_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_dir = root / "tokenizer"
            token_dir.mkdir()
            token_file = token_dir / "tokenizer.json"
            token_file.write_text('{"model": "keep"}')
            args = self.make_args(root, ["--payload_tokenizer_path", str(token_dir),
                                         "--summary_file", str(token_file), "--resume"])
            with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "load_local", return_value=[]) as loader:
                with self.assertRaisesRegex(ValueError, "tokenizer"):
                    await evaluate.main()
            loader.assert_not_called()
            self.assertEqual(token_file.read_text(), '{"model": "keep"}')

    async def test_unsupported_bbh_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root, ["--dataset", "bbh"])
            record = {"question": "Produce a word", "gold": "New York", "subject": "free_form"}
            with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "load_local", return_value=[record]), patch.object(evaluate, "SentenceTransformerEncoder") as encoder:
                with self.assertRaisesRegex(ValueError, "BBH MCQA profile"):
                    await evaluate.main()
            encoder.assert_not_called()

    def test_tokenizer_content_changes_run_identity_and_formal_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text("{}\n")
            tok = root / "tokenizer"
            tok.mkdir()
            token_file = tok / "tokenizer.json"
            token_file.write_text("{}")
            args = self.make_args(root, ["--payload_tokenizer_path", str(tok)])
            original = evaluate._run_identity(args)
            token_file.write_text('{"changed":true}')
            changed = evaluate._run_identity(args)
            self.assertNotEqual(original["config_hash"], changed["config_hash"])
            self.assertEqual(changed["payload_tokenizer_identity"]["mode"], "local_diagnostic_unverified_serving_identity")
            args.require_resolved_revisions = True
            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                evaluate._validate_args(args)

    async def test_complete_evaluation_and_resume_keep_classifier_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"question": "Pick B: A. no B. yes C. no D. no", "choices": ["A. no", "B. yes", "C. no", "D. no"], "gold": "B", "subject": "test"}
            (root / "data.jsonl").write_text(json.dumps(record) + "\n")
            args = self.make_args(root, ["--classify_messages", "--n_samples", "1"])
            encoder = FakeEncoder()
            encoder.identity = None
            client = FakeLLMClient()
            with patch.object(evaluate, "parse_args", return_value=args), patch.object(evaluate, "SentenceTransformerEncoder", return_value=encoder), patch.object(evaluate, "LLMClient", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                await evaluate.main()
                original = Path(args.out_file).read_bytes()
                first_calls = client.calls
                args.resume = True
                await evaluate.main()
            self.assertEqual(Path(args.out_file).read_bytes(), original)
            self.assertEqual(client.calls, first_calls)
            self.assertEqual(client.calls, 6)
            rows = analyze.load_jsonl(args.out_file)
            self.assertEqual(rows[0]["pred"], "B")
            self.assertIn("config_hash", rows[0])
            self.assertIn("payload_tokenizer_identity", rows[0])
            self.assertEqual(analyze.run_status(args.out_file, rows), "complete 1/1")
            report = json.loads(Path(args.out_file + ".classification.json").read_text())
            self.assertEqual(report["status"], "insufficient_classes")
            self.assertEqual(report["n"], 6)
            self.assertTrue(Path(args.out_file + ".separability.md").is_file())

    def test_new_default_outputs_have_separate_namespace(self):
        args = evaluate.parse_args(["--data_file", "sample.jsonl", "--dataset", "mmlu"])
        self.assertEqual(Path(evaluate._auto_out_file(args)).parts[:2], ("results", "repaired"))
        self.assertEqual(args.checkpoint_dir, "results/repaired/checkpoints")


class StudentAnalysisTests(unittest.TestCase):
    def test_completion_requires_provenance_and_consistent_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            sidecar = path.with_suffix(".summary.json")
            rows = [{"config_hash": "new", "gold": "B", "pred": "B"}]
            cases = [
                ({"total": 1, "requested_samples": 1, "failed_samples": 0, "complete": True},
                 [{"gold": "B", "pred": "B"}], "unverified"),
                ({"config_hash": "new", "total": 1, "requested_samples": 20, "failed_samples": 19, "complete": True}, rows, "INCOMPLETE"),
                ({"config_hash": "new", "total": 1, "requested_samples": 1, "failed_samples": 0, "complete": True},
                 [{**rows[0], "attack_type": "slow_drift", "attack_valid": False}], "INCOMPLETE"),
            ]
            for summary, result_rows, expected in cases:
                sidecar.write_text(json.dumps(summary))
                self.assertIn(expected, analyze.run_status(str(path), result_rows))

    def test_missing_predictions_count_wrong_and_invalid_attacks_are_visible(self):
        stats = analyze.compute_accuracy([
            {"gold": "B", "pred": "B"}, {"gold": "B", "pred": None},
            {"gold": "B", "pred": "B", "attack_type": "slow_drift", "attack_valid": False},
        ])
        self.assertEqual(stats["accuracy"], 0.5)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["missing_predictions"], 1)
        self.assertEqual(stats["invalid_attacks"], 1)

    def test_malformed_log_cannot_silently_raise_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"pred":"B"}\n{broken\n')
            with self.assertRaisesRegex(ValueError, ":2"):
                analyze.load_jsonl(str(path))

    def test_incomplete_and_mismatched_summary_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            rows = [{"config_hash": "new"}]
            summary = path.with_suffix(".summary.json")
            summary.write_text(json.dumps({"config_hash": "new", "complete": False, "total": 1, "requested_samples": 20}))
            self.assertEqual(analyze.run_status(str(path), rows), "INCOMPLETE 1/20")
            rows[0]["config_hash"] = "old"
            self.assertIn("identity mismatch", analyze.run_status(str(path), rows))

    def test_classifier_uses_explicit_round_id(self):
        samples = extract_message_samples({"attack_type": "slow_drift", "attacker_ids": [0], "agent_histories": [[{"round": 3, "response": "<REASON>: scope\n<ANSWER>: C"}]]})
        self.assertEqual(samples[0].round_id, 3)
        self.assertEqual(samples[0].label, 1)


if __name__ == "__main__":
    unittest.main()
