"""Sentence-BERT embeddings and round-wise separation metrics.

This module implements the embedding interface used by the near-benign attacks
in Sections 4--5 of the paper.  Embeddings are L2-normalized so cosine distance
is ``1 - dot(a, b)`` and Euclidean drift remains directly measurable.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_EMBEDDING_IDENTITY_KIND = "runtime_content_manifest_v1"
_IDENTITY_PREFIX = "sha256-manifest-v1:"
_REVISION_METADATA_KIND = "huggingface_download_metadata_first_line"
LABEL_FREE_EMBEDDING_SCORE_DEFINITION = (
    "label_free_mean_2_nearest_peer_cosine_distance"
)


class EmbeddingIdentityError(RuntimeError):
    """The loaded embedding model does not match its declared identity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise EmbeddingIdentityError(
            f"Embedding manifest file name must be a non-empty POSIX relative path: {name!r}"
        )
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise EmbeddingIdentityError(
            f"Embedding manifest file name must stay within the model root as a relative path: {name!r}"
        )
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root):
        raise EmbeddingIdentityError(
            f"Embedding manifest file name escapes the model root: {name!r}"
        )
    return candidate


def validate_embedding_contract(
    contract: Mapping[str, Any], *, expected_revision: str
) -> dict[str, Any]:
    """Validate the declared, runtime-relevant Sentence-BERT identity."""

    if not _REVISION_RE.fullmatch(str(expected_revision).lower()):
        raise EmbeddingIdentityError(
            "Formal embedding revision must be a resolved 40-character lowercase commit SHA"
        )
    repository_id = contract.get("repository_id")
    if not isinstance(repository_id, str) or not repository_id:
        raise EmbeddingIdentityError("Embedding content manifest requires repository_id")
    revision = str(contract.get("revision", "")).lower()
    if revision != str(expected_revision).lower():
        raise EmbeddingIdentityError(
            f"Embedding manifest revision {revision!r} does not match declared revision "
            f"{expected_revision!r}"
        )
    if contract.get("identity_kind") != _EMBEDDING_IDENTITY_KIND:
        raise EmbeddingIdentityError(
            f"Embedding content manifest identity_kind must be {_EMBEDDING_IDENTITY_KIND}"
        )
    if contract.get("revision_metadata") != _REVISION_METADATA_KIND:
        raise EmbeddingIdentityError(
            f"Embedding content manifest revision_metadata must be {_REVISION_METADATA_KIND}"
        )
    files = contract.get("files")
    if not isinstance(files, list) or not files:
        raise EmbeddingIdentityError(
            "Embedding content manifest files must be a non-empty ordered list"
        )
    names: set[str] = set()
    hashes: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise EmbeddingIdentityError(
                f"Embedding content manifest files[{index}] must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise EmbeddingIdentityError(
                f"Embedding content manifest contains an invalid or duplicate file name: {name!r}"
            )
        relative = PurePosixPath(name)
        if (
            "\\" in name
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise EmbeddingIdentityError(
                f"Embedding manifest file name must be a POSIX relative path: {name!r}"
            )
        names.add(name)
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise EmbeddingIdentityError(f"Invalid embedding byte count for {name}")
        expected_sha256 = str(entry.get("sha256", "")).lower()
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise EmbeddingIdentityError(f"Invalid embedding SHA-256 for {name}")
        hashes.append(expected_sha256)
    combined = hashlib.sha256(
        "".join(f"{value}\n" for value in hashes).encode("ascii")
    ).hexdigest()
    expected_identity = _IDENTITY_PREFIX + combined
    if contract.get("identity") != expected_identity:
        raise EmbeddingIdentityError(
            "Embedding content manifest identity does not match its ordered file hashes"
        )
    return {
        "repository_id": repository_id,
        "revision": revision,
        "identity": expected_identity,
        "files": files,
    }


def load_embedding_contract(manifest_path: str | Path) -> dict[str, Any]:
    """Load the embedding object from a runtime content manifest."""

    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingIdentityError(
            f"Could not read embedding content manifest: {manifest_path}"
        ) from exc
    contract = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(contract, dict):
        raise EmbeddingIdentityError(
            "Embedding content manifest must contain an embedding object"
        )
    return contract


def verify_embedding_identity(
    model_name: str | Path,
    *,
    expected_revision: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless runtime files and Hugging Face commit metadata match."""

    declared = validate_embedding_contract(
        contract, expected_revision=expected_revision
    )
    root = Path(model_name).resolve()
    if not root.is_dir():
        raise EmbeddingIdentityError(
            f"Formal embedding model must be an existing local directory: {root}"
        )
    observed_bytes = 0
    observed_hashes: list[str] = []
    for entry in declared["files"]:
        name = entry["name"]
        path = _safe_relative_path(root, name)
        if not path.is_file():
            raise EmbeddingIdentityError(f"Embedding runtime file is missing: {name}")
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["bytes"]:
            raise EmbeddingIdentityError(
                f"Embedding size mismatch for {name}: expected {entry['bytes']}, got {actual_bytes}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            raise EmbeddingIdentityError(
                f"Embedding SHA-256 mismatch for {name}: expected {entry['sha256']}, "
                f"got {actual_sha256}"
            )
        relative = PurePosixPath(name)
        metadata = (
            root
            / ".cache"
            / "huggingface"
            / "download"
            / Path(*relative.parts[:-1])
            / f"{relative.name}.metadata"
        )
        if not metadata.is_file():
            raise EmbeddingIdentityError(
                f"Embedding Hugging Face revision metadata is missing for {name}"
            )
        lines = metadata.read_text(encoding="utf-8", errors="strict").splitlines()
        metadata_revision = lines[0].strip().lower() if lines else ""
        if metadata_revision != declared["revision"]:
            raise EmbeddingIdentityError(
                f"Embedding metadata revision mismatch for {name}: expected "
                f"{declared['revision']}, got {metadata_revision or '<empty>'}"
            )
        observed_bytes += actual_bytes
        observed_hashes.append(actual_sha256)
    combined = hashlib.sha256(
        "".join(f"{value}\n" for value in observed_hashes).encode("ascii")
    ).hexdigest()
    observed_identity = _IDENTITY_PREFIX + combined
    if observed_identity != declared["identity"]:
        raise EmbeddingIdentityError(
            "Observed embedding runtime identity does not match the content manifest"
        )
    return {
        "embedding_repository_id": declared["repository_id"],
        "embedding_revision": declared["revision"],
        "embedding_identity": observed_identity,
        "embedding_identity_kind": _EMBEDDING_IDENTITY_KIND,
        "verified_file_count": len(declared["files"]),
        "verified_total_bytes": observed_bytes,
    }


class TextEncoder(Protocol):
    """Minimal encoder protocol, allowing deterministic encoders in tests."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array with shape ``(len(texts), embedding_dim)``."""


class SentenceTransformerEncoder:
    """Lazy Sentence-BERT wrapper with normalized numpy output."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        revision: Optional[str] = None,
        identity_contract: Optional[Mapping[str, Any]] = None,
        require_identity: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "Embedding analysis requires sentence-transformers. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        if require_identity and identity_contract is None:
            raise EmbeddingIdentityError(
                "Formal embedding loading requires a runtime content manifest"
            )
        self.identity = (
            verify_embedding_identity(
                model_name,
                expected_revision=str(revision or ""),
                contract=identity_contract,
            )
            if identity_contract is not None
            else None
        )
        loader_kwargs: dict[str, Any] = {}
        if revision and revision != "unresolved":
            loader_kwargs["revision"] = revision
        if Path(model_name).is_dir():
            loader_kwargs["local_files_only"] = True
        if identity_contract is not None:
            loader_kwargs["model_kwargs"] = {"use_safetensors": True}
        self._model = SentenceTransformer(model_name, device=device, **loader_kwargs)
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        # A shared encoder is used by concurrent evaluation tasks.  Serializing
        # encode calls avoids concurrent access to one torch model/GPU stream.
        with self._lock:
            values = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return _normalize_rows(np.asarray(values, dtype=np.float32))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(values, dtype=np.float32))
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding encoder returned a zero-length vector.")
    return values / norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _normalize_rows(np.asarray(a))[0]
    b_n = _normalize_rows(np.asarray(b))[0]
    return float(np.clip(1.0 - np.dot(a_n, b_n), 0.0, 2.0))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def distances_to_support(vector: np.ndarray, support: np.ndarray) -> np.ndarray:
    support_n = _normalize_rows(support)
    vector_n = _normalize_rows(np.asarray(vector))[0]
    return np.clip(1.0 - support_n @ vector_n, 0.0, 2.0)


def distance_to_support(vector: np.ndarray, support: np.ndarray) -> float:
    if len(support) == 0:
        raise ValueError("Benign embedding support is empty.")
    return float(np.min(distances_to_support(vector, support)))


def label_free_peer_outlier_scores(
    embeddings: Sequence[np.ndarray], *, nearest_neighbors: int = 2
) -> List[float]:
    """Score every agent without using role labels or a known-benign support."""

    if nearest_neighbors < 1:
        raise ValueError("nearest_neighbors must be positive.")
    if len(embeddings) < 2:
        raise ValueError("Label-free embedding score requires at least two agents.")
    scores: List[float] = []
    for index, embedding in enumerate(embeddings):
        peers = np.stack(
            [peer for peer_index, peer in enumerate(embeddings) if peer_index != index]
        )
        distances = sorted(float(value) for value in distances_to_support(embedding, peers))
        k = min(nearest_neighbors, len(distances))
        scores.append(float(np.mean(distances[:k])))
    return scores


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def binary_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> Optional[float]:
    """Compute a tie-aware Mann--Whitney AUC without an sklearn dependency."""
    pos = [float(value) for value in positive_scores if value is not None]
    neg = [float(value) for value in negative_scores if value is not None]
    if not pos or not neg:
        return None
    wins = 0.0
    for left in pos:
        for right in neg:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def benign_disagreement_rate(answers: Sequence[Optional[str]]) -> float:
    """Ordered-pair disagreement from Eq. (4) of the paper."""
    invalid = [
        index
        for index, answer in enumerate(answers)
        if not isinstance(answer, str) or not answer.strip()
    ]
    if invalid:
        raise ValueError(
            "Benign disagreement requires one parsed answer per benign agent; "
            f"unparsed positions={invalid}."
        )
    valid = [str(answer) for answer in answers]
    n = len(valid)
    if n < 2:
        return 0.0
    disagree = sum(valid[i] != valid[j] for i in range(n) for j in range(n) if i != j)
    return float(disagree / (n * (n - 1)))


@dataclass
class RoundEmbeddingAnalysis:
    per_agent: List[Dict[str, Optional[float]]]
    attacker_benign_cosine_distance: Optional[float]
    benign_benign_same_cosine_distance: Optional[float]
    benign_benign_diff_cosine_distance: Optional[float]
    benign_embedding_variance: float
    benign_disagreement_rate: float
    embedding_score_definition: str
    embedding_auc: Optional[float]
    confidence_auc: Optional[float]

    def to_dict(self) -> dict:
        return {
            "per_agent": self.per_agent,
            "attacker_benign_cosine_distance": self.attacker_benign_cosine_distance,
            "benign_benign_same_cosine_distance": self.benign_benign_same_cosine_distance,
            "benign_benign_diff_cosine_distance": self.benign_benign_diff_cosine_distance,
            "benign_embedding_variance": self.benign_embedding_variance,
            "benign_disagreement_rate": self.benign_disagreement_rate,
            "embedding_score_definition": self.embedding_score_definition,
            "embedding_auc": self.embedding_auc,
            "confidence_auc": self.confidence_auc,
        }


def analyze_round_embeddings(
    embeddings: Sequence[np.ndarray],
    answers: Sequence[Optional[str]],
    attacker_ids: Sequence[int],
    confidences: Optional[Sequence[Optional[float]]] = None,
) -> RoundEmbeddingAnalysis:
    """Compute the B--M and B--B quantities reported in paper Table 2."""
    attacker_set = set(attacker_ids)
    benign_ids = [i for i in range(len(embeddings)) if i not in attacker_set]
    attack_ids = [i for i in range(len(embeddings)) if i in attacker_set]
    if not benign_ids:
        raise ValueError("Embedding analysis requires at least one benign agent.")

    benign_matrix = _normalize_rows(np.stack([embeddings[i] for i in benign_ids]))
    attack_benign: List[float] = []
    same: List[float] = []
    diff: List[float] = []

    for attack_id in attack_ids:
        attack_benign.extend(distances_to_support(embeddings[attack_id], benign_matrix).tolist())

    for pos, left in enumerate(benign_ids):
        for right in benign_ids[pos + 1 :]:
            value = cosine_distance(embeddings[left], embeddings[right])
            if answers[left] is not None and answers[left] == answers[right]:
                same.append(value)
            else:
                diff.append(value)

    centroid = np.mean(benign_matrix, axis=0)
    variance = float(np.mean(np.sum((benign_matrix - centroid) ** 2, axis=1)))

    outlier_scores = label_free_peer_outlier_scores(embeddings)
    per_agent: List[Dict[str, Optional[float]]] = []
    for agent_id, embedding in enumerate(embeddings):
        if agent_id in attacker_set:
            nearest = distance_to_support(embedding, benign_matrix)
        else:
            other = [i for i in benign_ids if i != agent_id]
            nearest = (
                distance_to_support(embedding, np.stack([embeddings[i] for i in other]))
                if other
                else None
            )
        per_agent.append(
            {
                "cosine_distance_to_benign": nearest,
                "embedding_outlier_score": outlier_scores[agent_id],
            }
        )

    distance_scores = [outlier_scores[agent_id] for agent_id in attack_ids]
    benign_distance_scores = [outlier_scores[agent_id] for agent_id in benign_ids]
    embedding_auc = binary_auc(distance_scores, benign_distance_scores)
    confidence_auc = None
    if confidences is not None and len(confidences) == len(embeddings):
        # Low confidence is treated as the positive attack signal, hence -C.
        confidence_auc = binary_auc(
            [-float(confidences[agent_id]) for agent_id in attack_ids if confidences[agent_id] is not None],
            [-float(confidences[agent_id]) for agent_id in benign_ids if confidences[agent_id] is not None],
        )

    return RoundEmbeddingAnalysis(
        per_agent=per_agent,
        attacker_benign_cosine_distance=_mean_or_none(attack_benign),
        benign_benign_same_cosine_distance=_mean_or_none(same),
        benign_benign_diff_cosine_distance=_mean_or_none(diff),
        benign_embedding_variance=variance,
        benign_disagreement_rate=benign_disagreement_rate([answers[i] for i in benign_ids]),
        embedding_score_definition=LABEL_FREE_EMBEDDING_SCORE_DEFINITION,
        embedding_auc=embedding_auc,
        confidence_auc=confidence_auc,
    )
