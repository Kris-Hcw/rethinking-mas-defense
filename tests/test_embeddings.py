import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.embeddings import (
    EmbeddingIdentityError,
    SentenceTransformerEncoder,
    analyze_round_embeddings,
    benign_disagreement_rate,
    binary_auc,
    verify_embedding_identity,
)


class EmbeddingTests(unittest.TestCase):
    REVISION = "1" * 40

    def _identity_fixture(self, root: Path) -> dict:
        payloads = {
            "modules.json": b"{}\n",
            "model.safetensors": b"fixed embedding weights",
            "1_Pooling/config.json": b'{"pooling_mode_mean_tokens": true}\n',
        }
        files = []
        for name, content in payloads.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            metadata = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(f"{self.REVISION}\n", encoding="utf-8")
            files.append(
                {
                    "name": name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        combined = hashlib.sha256(
            "".join(f"{entry['sha256']}\n" for entry in files).encode("ascii")
        ).hexdigest()
        return {
            "repository_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": self.REVISION,
            "identity_kind": "runtime_content_manifest_v1",
            "identity": f"sha256-manifest-v1:{combined}",
            "files": files,
            "revision_metadata": "huggingface_download_metadata_first_line",
        }

    def test_embedding_identity_accepts_exact_runtime_files_and_revision_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._identity_fixture(root)
            observed = verify_embedding_identity(
                root,
                expected_revision=self.REVISION,
                contract=contract,
            )
        self.assertEqual(observed["embedding_revision"], self.REVISION)
        self.assertEqual(observed["embedding_identity"], contract["identity"])
        self.assertEqual(observed["verified_file_count"], 3)

    def test_embedding_identity_rejects_weight_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._identity_fixture(root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(EmbeddingIdentityError, "model.safetensors"):
                verify_embedding_identity(
                    root,
                    expected_revision=self.REVISION,
                    contract=contract,
                )

    def test_embedding_identity_rejects_declared_or_metadata_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._identity_fixture(root)
            with self.assertRaisesRegex(EmbeddingIdentityError, "declared revision"):
                verify_embedding_identity(
                    root,
                    expected_revision="2" * 40,
                    contract=contract,
                )
            metadata = (
                root
                / ".cache"
                / "huggingface"
                / "download"
                / "modules.json.metadata"
            )
            metadata.write_text("3" * 40 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingIdentityError, "metadata revision"):
                verify_embedding_identity(
                    root,
                    expected_revision=self.REVISION,
                    contract=contract,
                )

    def test_embedding_identity_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._identity_fixture(root)
            contract = json.loads(json.dumps(contract))
            contract["files"][0]["name"] = "../outside"
            with self.assertRaisesRegex(EmbeddingIdentityError, "relative"):
                verify_embedding_identity(
                    root,
                    expected_revision=self.REVISION,
                    contract=contract,
                )

    def test_formal_encoder_pins_revision_local_files_and_safetensors(self):
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name, *, device=None, **kwargs):
                calls.append((model_name, device, kwargs))

        fake_module = types.SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self._identity_fixture(root)
            with patch.dict(
                sys.modules, {"sentence_transformers": fake_module}
            ):
                encoder = SentenceTransformerEncoder(
                    str(root),
                    device="cpu",
                    revision=self.REVISION,
                    identity_contract=contract,
                    require_identity=True,
                )

        self.assertEqual(len(calls), 1)
        model_name, device, kwargs = calls[0]
        self.assertEqual(model_name, str(root))
        self.assertEqual(device, "cpu")
        self.assertEqual(kwargs["revision"], self.REVISION)
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["model_kwargs"], {"use_safetensors": True})
        self.assertEqual(encoder.identity["embedding_revision"], self.REVISION)

    def test_table2_metrics_and_disagreement(self):
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
        ]
        analysis = analyze_round_embeddings(
            embeddings, ["A", "A", "B"], [2], confidences=[0.9, 0.8, 0.2]
        )
        self.assertIsNotNone(analysis.attacker_benign_cosine_distance)
        self.assertIsNotNone(analysis.benign_benign_same_cosine_distance)
        self.assertIsNone(analysis.benign_benign_diff_cosine_distance)
        self.assertEqual(analysis.benign_disagreement_rate, 0.0)
        self.assertIsNotNone(analysis.embedding_auc)
        self.assertEqual(
            analysis.embedding_score_definition,
            "label_free_mean_2_nearest_peer_cosine_distance",
        )
        self.assertTrue(
            all(item["embedding_outlier_score"] is not None for item in analysis.per_agent)
        )
        self.assertEqual(analysis.confidence_auc, 1.0)
        self.assertEqual(benign_disagreement_rate(["A", "B", "B"]), 2 / 3)
        with self.assertRaisesRegex(ValueError, "unparsed positions"):
            benign_disagreement_rate(["A", None, "B"])

    def test_embedding_outlier_scores_do_not_use_attacker_labels(self):
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]),
        ]
        first = analyze_round_embeddings(embeddings, ["A", "A", "B", "C"], [0])
        second = analyze_round_embeddings(embeddings, ["A", "A", "B", "C"], [3])
        self.assertEqual(
            [item["embedding_outlier_score"] for item in first.per_agent],
            [item["embedding_outlier_score"] for item in second.per_agent],
        )

    def test_binary_auc_is_tie_aware(self):
        self.assertEqual(binary_auc([2.0], [1.0]), 1.0)
        self.assertEqual(binary_auc([1.0], [1.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
