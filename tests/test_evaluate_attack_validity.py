import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import evaluate
from src.mas import NoFeasibleAttackCandidate


class EvaluateAttackValidityTests(unittest.IsolatedAsyncioTestCase):
    def test_active_attack_requires_explicit_valid_result(self):
        with self.assertRaises(evaluate.InvalidAttackResultError):
            evaluate._require_fresh_attack_valid(
                {"attack_valid": False}, "slow_drift"
            )
        with self.assertRaises(evaluate.InvalidAttackResultError):
            evaluate._require_fresh_attack_valid({}, "chaos_seeding")

    def test_clean_result_is_not_subject_to_attack_validity_gate(self):
        evaluate._require_fresh_attack_valid({"attack_valid": False}, "none")

    async def test_invalid_fresh_attack_is_not_persisted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            out_path = root / "result.jsonl"
            embedding_path = root / "embedding.jsonl"
            summary_path = root / "summary.json"
            signal_path = root / "signal.json"
            checkpoint_dir = root / "checkpoints"
            data_path.write_text("{}\n", encoding="utf-8")
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--n_samples",
                    "1",
                    "--attack",
                    "slow_drift",
                    "--n_agents",
                    "3",
                    "--n_attackers",
                    "1",
                    "--attacker_ids",
                    "0",
                    "--topology",
                    "full",
                    "--out_file",
                    str(out_path),
                    "--embedding_out_file",
                    str(embedding_path),
                    "--summary_file",
                    str(summary_path),
                    "--signal_decay_file",
                    str(signal_path),
                    "--checkpoint_dir",
                    str(checkpoint_dir),
                ]
            )
            record = {
                "id": "sample-1",
                "subject": "test",
                "question": "Question?",
                "choices": ["A", "B", "C", "D"],
                "answer_options": ["A", "B", "C", "D"],
                "gold": "B",
            }
            invalid_result = {
                "pred": "C",
                "target_wrong": "C",
                "attack_valid": False,
                "vote_counts": {"C": 2, "B": 1},
                "per_agent_answers": ["C", "C", "B"],
                "per_agent_conf": [0.8, 0.7, 0.6],
                "attack_type": "slow_drift",
                "attacker_ids": [0],
                "sample_seed": 123,
                "defense_mode": "none",
                "agent_histories": [[], [], []],
                "round_results": [],
                "embedding_separation": {},
            }
            fake_mas = unittest.mock.Mock()
            fake_mas.run_one = AsyncMock(return_value=invalid_result)

            with (
                patch.object(evaluate, "parse_args", return_value=args),
                patch.object(evaluate, "load_local", return_value=[record]),
                patch.object(evaluate, "LLMClient"),
                patch.object(evaluate, "SentenceTransformerEncoder"),
                patch.object(evaluate, "DebateMAS", return_value=fake_mas),
            ):
                with self.assertRaises(evaluate.EvaluationIncompleteError):
                    await evaluate.main()

            self.assertEqual(
                fake_mas.run_one.await_args.kwargs["answer_options"],
                ["A", "B", "C", "D"],
            )

            checkpoint_path = Path(evaluate._checkpoint_path(args, str(out_path)))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(out_path.read_text(encoding="utf-8"), "")
            self.assertEqual(embedding_path.read_text(encoding="utf-8"), "")
            self.assertEqual(checkpoint["completed_samples"], [])
            self.assertIsNone(checkpoint["last_result"])
            self.assertEqual(
                checkpoint["errors"][-1]["error_type"],
                "InvalidAttackResultError",
            )

    async def test_infeasible_candidate_diagnostics_are_persisted_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            out_path = root / "result.jsonl"
            embedding_path = root / "embedding.jsonl"
            summary_path = root / "summary.json"
            signal_path = root / "signal.json"
            checkpoint_dir = root / "checkpoints"
            data_path.write_text("{}\n", encoding="utf-8")
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--n_samples",
                    "1",
                    "--attack",
                    "benign_wrapper",
                    "--attack_candidates",
                    "4",
                    "--n_agents",
                    "3",
                    "--n_attackers",
                    "1",
                    "--attacker_ids",
                    "0",
                    "--topology",
                    "full",
                    "--out_file",
                    str(out_path),
                    "--embedding_out_file",
                    str(embedding_path),
                    "--summary_file",
                    str(summary_path),
                    "--signal_decay_file",
                    str(signal_path),
                    "--checkpoint_dir",
                    str(checkpoint_dir),
                ]
            )
            record = {
                "id": "sample-1",
                "subject": "test",
                "question": "Question?",
                "choices": ["A", "B", "C", "D"],
                "gold": "B",
            }
            diagnostics = [
                {
                    "candidate_index": index,
                    "target_answered": index != 0,
                    "obvious_marker_heuristic_passed": index != 1,
                    "constraint_satisfied": False,
                    "formally_valid": False,
                    "rejection_reasons": [
                        ["target_answer_mismatch"],
                        ["final_cosine_distance_exceeds_tau"],
                        ["wrapper_cosine_distance_exceeds_tau"],
                        ["payload_token_budget_exceeded"],
                    ][index],
                }
                for index in range(4)
            ]
            failure = NoFeasibleAttackCandidate(
                "benign_wrapper",
                diagnostics,
                candidate_budget=4,
                constraint_thresholds={
                    "tau": 0.25,
                    "drift_epsilon": 0.5,
                    "wrapper_tau": 0.25,
                    "payload_token_budget": 16,
                },
            )
            fake_mas = unittest.mock.Mock()
            fake_mas.run_one = AsyncMock(side_effect=failure)

            with (
                patch.object(evaluate, "parse_args", return_value=args),
                patch.object(evaluate, "load_local", return_value=[record]),
                patch.object(evaluate, "LLMClient"),
                patch.object(evaluate, "SentenceTransformerEncoder"),
                patch.object(evaluate, "DebateMAS", return_value=fake_mas),
            ):
                with self.assertRaises(evaluate.EvaluationIncompleteError):
                    await evaluate.main()

            checkpoint_path = Path(evaluate._checkpoint_path(args, str(out_path)))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            error = checkpoint["errors"][-1]
            self.assertEqual(out_path.read_text(encoding="utf-8"), "")
            self.assertEqual(embedding_path.read_text(encoding="utf-8"), "")
            self.assertEqual(checkpoint["completed_samples"], [])
            self.assertIsNone(checkpoint["last_result"])
            self.assertEqual(error["error_type"], "NoFeasibleAttackCandidate")
            self.assertEqual(error["candidate_diagnostics_schema_version"], 1)
            self.assertEqual(error["attack_type"], "benign_wrapper")
            self.assertEqual(error["candidate_budget"], 4)
            self.assertEqual(error["generated_candidate_count"], 4)
            self.assertEqual(error["constraint_thresholds"]["wrapper_tau"], 0.25)
            self.assertEqual(error["candidate_diagnostics"], diagnostics)


if __name__ == "__main__":
    unittest.main()
