import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate


class EvaluateIdentityTests(unittest.TestCase):
    EXPECTED_IMPLEMENTATION_FILES = (
        "evaluate.py",
        "src/attacks.py",
        "src/confidence.py",
        "src/data.py",
        "src/embeddings.py",
        "src/entropy_contract.py",
        "src/gsm8k_eval.py",
        "src/llm_client.py",
        "src/mas.py",
        "src/result_validity.py",
        "src/topology.py",
    )

    def test_code_fingerprint_covers_formal_semantic_modules(self):
        self.assertEqual(
            evaluate._IMPLEMENTATION_CODE_FILES,
            self.EXPECTED_IMPLEMENTATION_FILES,
        )
        manifest = evaluate._code_fingerprint_manifest()
        self.assertEqual(tuple(manifest), self.EXPECTED_IMPLEMENTATION_FILES)
        self.assertTrue(
            all(len(value) == 64 for value in manifest.values()),
            manifest,
        )

    def test_each_semantic_file_mutation_changes_code_fingerprint(self):
        source_root = Path(evaluate.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in self.EXPECTED_IMPLEMENTATION_FILES:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((source_root / relative).read_bytes())

            baseline = evaluate._code_fingerprint(root)
            for relative in self.EXPECTED_IMPLEMENTATION_FILES:
                with self.subTest(path=relative):
                    path = root / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# fingerprint mutation\n")
                    self.assertNotEqual(baseline, evaluate._code_fingerprint(root))
                    path.write_bytes(original)

    def test_run_identity_persists_auditable_code_fingerprint_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.jsonl"
            data_path.write_text("{}\n", encoding="utf-8")
            args = evaluate.parse_args(
                ["--data_file", str(data_path), "--dataset", "mmlu"]
            )
            identity = evaluate._run_identity(args)

        self.assertEqual(identity["schema_version"], 9)
        self.assertEqual(
            identity["code_fingerprint_schema"],
            "implementation_semantic_files_v2",
        )
        self.assertEqual(
            tuple(identity["code_fingerprint_files"]),
            self.EXPECTED_IMPLEMENTATION_FILES,
        )
        self.assertEqual(
            identity["code_fingerprint_sha256"],
            evaluate._code_fingerprint(),
        )

    def test_exact_query_selector_preserves_declared_order(self):
        records = [
            {"question": "alpha"},
            {"question": "beta"},
            {"question": "gamma"},
        ]
        alpha = hashlib.md5(b"alpha").hexdigest()
        gamma = hashlib.md5(b"gamma").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": [gamma, alpha],
                    }
                ),
                encoding="utf-8",
            )
            selector = evaluate._load_query_selector(path)
            selected = evaluate._select_records_by_query_ids(records, selector)

        self.assertEqual([row["question"] for row in selected], ["gamma", "alpha"])

    def test_exact_query_selector_rejects_duplicate_and_missing_ids(self):
        alpha = hashlib.md5(b"alpha").hexdigest()
        missing = hashlib.md5(b"missing").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": [alpha, alpha],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate._load_query_selector(path)

            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": [missing],
                    }
                ),
                encoding="utf-8",
            )
            selector = evaluate._load_query_selector(path)
            with self.assertRaisesRegex(ValueError, "not found"):
                evaluate._select_records_by_query_ids(
                    [{"question": "alpha"}], selector
                )

    def test_exact_query_selector_cannot_be_combined_with_n_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            selector_path = root / "selector.json"
            data_path.write_text("{}\n", encoding="utf-8")
            selector_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": ["0" * 32],
                    }
                ),
                encoding="utf-8",
            )
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--query_ids_file",
                    str(selector_path),
                    "--n_samples",
                    "1",
                ]
            )
            with self.assertRaisesRegex(ValueError, "n_samples"):
                evaluate._validate_args(args)

    def test_query_selector_content_changes_config_hash(self):
        alpha = hashlib.md5(b"alpha").hexdigest()
        beta = hashlib.md5(b"beta").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            selector_path = root / "selector.json"
            data_path.write_text("{}\n", encoding="utf-8")
            selector_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": [alpha],
                    }
                ),
                encoding="utf-8",
            )
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--query_ids_file",
                    str(selector_path),
                ]
            )
            first = evaluate._run_identity(args)
            selector_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "query_id_scheme": "md5_utf8_question_v1",
                        "query_ids": [beta],
                    }
                ),
                encoding="utf-8",
            )
            second = evaluate._run_identity(args)

        self.assertNotEqual(first["config_hash"], second["config_hash"])
        self.assertNotEqual(
            first["query_selector"]["file_sha256"],
            second["query_selector"]["file_sha256"],
        )

    def test_resume_rejects_config_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps({"schema_version": 2, "config_hash": "old"}),
                encoding="utf-8",
            )
            with self.assertRaises(evaluate.CheckpointIdentityError):
                evaluate._load_checkpoint(
                    str(path), {"config_hash": "new"}, strict_identity=True
                )

    def test_resume_rebuilds_missing_embedding_sidecar_from_primary_rows(self):
        round_result = {
            "round": 1,
            "agents": [
                {
                    "agent_id": 0,
                    "role": "attacker",
                    "message": "candidate",
                    "embedding": [1.0, 0.0],
                    "confidence": 0.4,
                },
                {
                    "agent_id": 1,
                    "role": "benign",
                    "message": "support",
                    "embedding": [0.0, 1.0],
                    "confidence": 0.8,
                },
            ],
            "attacker_benign_cosine_distance": 1.0,
            "benign_benign_same_cosine_distance": None,
            "benign_benign_diff_cosine_distance": None,
            "benign_embedding_variance": 0.0,
            "benign_disagreement_rate": 0.0,
            "embedding_score_definition": "label_free_mean_2_nearest_peer_cosine_distance",
            "embedding_auc": 1.0,
            "confidence_auc": 1.0,
        }
        result_row = {
            "step": 1,
            "query_id": "query-one",
            "config_hash": "config-one",
            "confidence_entropy_mode": "exact_full_vocab",
            "backend_revision": "backend-one",
            "backend_manifest_sha256": "manifest-one",
            "sampling_top_k": 20,
            "full_vocab_size": 151936,
            "full_vocab_entropy_source": "raw_generation_logits",
            "runtime_python_version": "3.10.18",
            "runtime_torch_version": "2.8.0+cu128",
            "runtime_torch_distribution_version": "2.8.0",
            "runtime_transformers_version": "5.15.1",
            "runtime_fastapi_version": "0.136.1",
            "runtime_uvicorn_version": "0.46.0",
            "round_results": [round_result],
        }
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "condition.jsonl"
            sidecar = Path(directory) / "condition.embedding_analysis.jsonl"
            primary.write_text(
                json.dumps(result_row) + "\n", encoding="utf-8"
            )
            sidecar.write_text('{"stale": true}\n', encoding="utf-8")
            resume = evaluate._load_resume_artifacts(
                str(primary),
                str(sidecar),
                config_hash="config-one",
                expected_query_ids={1: "query-one"},
            )
            rows = [
                json.loads(line)
                for line in sidecar.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(resume["completed_steps"], {1})
            self.assertEqual(resume["embedding_rows_rebuilt"], 2)
            self.assertEqual({row["agent_id"] for row in rows}, {0, 1})
            self.assertTrue(all(row["step"] == 1 for row in rows))
            self.assertTrue(all(row["query_id"] == "query-one" for row in rows))
            self.assertTrue(all(row["config_hash"] == "config-one" for row in rows))
            self.assertTrue(
                all(row["runtime_torch_version"] == "2.8.0+cu128" for row in rows)
            )
            self.assertFalse(Path(str(sidecar) + ".resume.tmp").exists())

    def test_resume_rejects_duplicate_primary_steps(self):
        result_row = {
            "step": 1,
            "query_id": "query-one",
            "config_hash": "config-one",
            "confidence_entropy_mode": "exact_full_vocab",
            "round_results": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "condition.jsonl"
            sidecar = Path(directory) / "condition.embedding_analysis.jsonl"
            primary.write_text(
                json.dumps(result_row) + "\n" + json.dumps(result_row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                evaluate.CheckpointIdentityError, "duplicate step=1"
            ):
                evaluate._load_resume_artifacts(
                    str(primary),
                    str(sidecar),
                    config_hash="config-one",
                    expected_query_ids={1: "query-one"},
                )

    def test_random_attacker_placement_is_seeded_and_not_fixed(self):
        args = argparse.Namespace(
            attack="slow_drift",
            attacker_ids=None,
            attacker_placement="random_per_sample",
            n_attackers=2,
            n_agents=5,
        )
        adjacency = np.ones((5, 5), dtype=int) - np.eye(5, dtype=int)
        first = evaluate._attacker_ids(args, sample_seed=100, adjacency=adjacency)
        self.assertEqual(
            first, evaluate._attacker_ids(args, sample_seed=100, adjacency=adjacency)
        )
        observed = {
            tuple(evaluate._attacker_ids(args, sample_seed=seed, adjacency=adjacency))
            for seed in range(20)
        }
        self.assertGreater(len(observed), 1)

    def test_random_placement_excludes_non_propagating_nodes(self):
        args = argparse.Namespace(
            attack="slow_drift",
            attacker_ids=None,
            attacker_placement="random_per_sample",
            n_attackers=2,
            n_agents=4,
        )
        adjacency = np.asarray(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
            dtype=int,
        )
        for seed in range(10):
            selected = evaluate._attacker_ids(
                args, sample_seed=seed, adjacency=adjacency
            )
            self.assertNotIn(2, selected)

    def test_topology_diagnostics_reports_effective_attackers(self):
        adjacency = np.asarray(
            [[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=int
        )
        diagnostics = evaluate._topology_diagnostics(adjacency, [0, 1])
        self.assertEqual(diagnostics["effective_propagating_attackers"], 1)
        self.assertEqual(diagnostics["attacker_out_degree"], {"0": 1, "1": 0})

    def test_explicit_attacker_with_zero_out_degree_is_rejected(self):
        args = argparse.Namespace(
            attack="slow_drift",
            attacker_ids="0,1",
            attacker_placement="random_per_sample",
            n_attackers=2,
            n_agents=3,
        )
        adjacency = np.asarray(
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=int
        )

        with self.assertRaisesRegex(ValueError, "propagat"):
            evaluate._attacker_ids(
                args, sample_seed=100, adjacency=adjacency
            )

    def test_formal_backend_identity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.jsonl"
            data_path.write_text("{}\n", encoding="utf-8")
            args = evaluate.parse_args(
                ["--data_file", str(data_path), "--dataset", "mmlu"]
            )
            args.require_resolved_backend_identity = True
            args.backend_revision = "unresolved"
            args.backend_manifest = None
            args.sampling_top_k = -1
            with self.assertRaisesRegex(ValueError, "backend"):
                evaluate._validate_args(args)

    def test_formal_backend_identity_rejects_approximate_entropy_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            manifest_path = root / "backend.json"
            data_path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text("{}\n", encoding="utf-8")
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--backend_revision",
                    "resolved-backend",
                    "--backend_manifest",
                    str(manifest_path),
                    "--sampling_top_k",
                    "20",
                    "--require_resolved_backend_identity",
                ]
            )

            self.assertEqual(
                args.confidence_entropy_mode, "top_logprobs_tail_bucket"
            )
            with self.assertRaisesRegex(ValueError, "exact_full_vocab"):
                evaluate._validate_args(args)

    def test_formal_embedding_identity_requires_a_content_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            model_path = root / "embedding"
            manifest_path = root / "backend.json"
            data_path.write_text("{}\n", encoding="utf-8")
            model_path.mkdir()
            manifest_path.write_text(
                json.dumps(
                    {
                        "embedding": {
                            "repository_id": "sentence-transformers/all-MiniLM-L6-v2",
                            "revision": "1" * 40,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = evaluate.parse_args(
                [
                    "--data_file",
                    str(data_path),
                    "--dataset",
                    "mmlu",
                    "--embedding_model",
                    str(model_path),
                    "--model_revision",
                    "model-revision",
                    "--embedding_revision",
                    "1" * 40,
                    "--backend_manifest",
                    str(manifest_path),
                    "--require_resolved_revisions",
                ]
            )
            with self.assertRaisesRegex(ValueError, "embedding.*manifest"):
                evaluate._validate_args(args)

    def test_backend_manifest_content_changes_config_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            manifest_path = root / "backend.json"
            data_path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text('{"runtime":"first"}\n', encoding="utf-8")
            args = evaluate.parse_args(
                ["--data_file", str(data_path), "--dataset", "mmlu"]
            )
            args.backend_revision = "backend-source-sha"
            args.backend_manifest = str(manifest_path)
            args.sampling_top_k = 20
            first = evaluate._run_identity(args)
            manifest_path.write_text('{"runtime":"second"}\n', encoding="utf-8")
            second = evaluate._run_identity(args)
        self.assertNotEqual(first["config_hash"], second["config_hash"])
        self.assertNotEqual(
            first["backend_identity"]["backend_manifest_sha256"],
            second["backend_identity"]["backend_manifest_sha256"],
        )

    def test_backend_identity_fields_are_ready_for_result_rows(self):
        identity = {
            "backend_identity": {
                "backend_revision": "backend-source-sha",
                "backend_manifest_sha256": "manifest-sha",
                "sampling_top_k": 20,
                "full_vocab_size": 151936,
                "full_vocab_entropy_source": "raw_generation_logits",
                "runtime_python_version": "3.10.18",
                "runtime_torch_version": "2.8.0+cu128",
                "runtime_torch_distribution_version": "2.8.0",
                "runtime_transformers_version": "5.15.1",
                "runtime_fastapi_version": "0.136.1",
                "runtime_uvicorn_version": "0.46.0",
            }
        }
        self.assertEqual(
            evaluate._backend_result_fields(identity),
            {
                "backend_revision": "backend-source-sha",
                "backend_manifest_sha256": "manifest-sha",
                "sampling_top_k": 20,
                "full_vocab_size": 151936,
                "full_vocab_entropy_source": "raw_generation_logits",
                "runtime_python_version": "3.10.18",
                "runtime_torch_version": "2.8.0+cu128",
                "runtime_torch_distribution_version": "2.8.0",
                "runtime_transformers_version": "5.15.1",
                "runtime_fastapi_version": "0.136.1",
                "runtime_uvicorn_version": "0.46.0",
            },
        )

    def test_formal_exact_entropy_manifest_requires_vocab_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            manifest_path = root / "backend.json"
            data_path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps({
                    "model": {},
                    "confidence": {
                        "mode": "exact_full_vocab",
                        "source": "raw_generation_logits",
                    },
                }),
                encoding="utf-8",
            )
            args = evaluate.parse_args([
                "--data_file", str(data_path),
                "--dataset", "mmlu",
                "--confidence_entropy_mode", "exact_full_vocab",
                "--backend_revision", "backend-source-sha",
                "--backend_manifest", str(manifest_path),
                "--sampling_top_k", "20",
                "--require_resolved_backend_identity",
            ])
            with self.assertRaisesRegex(ValueError, "vocab"):
                evaluate._validate_args(args)


if __name__ == "__main__":
    unittest.main()
