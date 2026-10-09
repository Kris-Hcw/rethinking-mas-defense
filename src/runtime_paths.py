"""Cross-platform resolution of local runtime assets.

Remote Linux model paths are part of frozen experiment configuration and do
not belong here. This module resolves only assets on the launching machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).absolute().parents[1]
EMBEDDING_MODEL_ENV = "GSAFEGUARD_EMBEDDING_MODEL"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _local_path(value: str | os.PathLike[str]) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def embedding_model_candidates(explicit: str | os.PathLike[str] | None = None) -> Iterable[Path]:
    """Yield platform-neutral candidates in priority order."""

    if explicit is not None:
        yield _local_path(explicit)
        return

    configured = os.environ.get(EMBEDDING_MODEL_ENV)
    if configured:
        yield _local_path(configured)

    # Shared layout on macOS and Windows:
    # SEU-Project-3/models/all-MiniLM-L6-v2/
    # SEU-Project-3/Confidence-Guided Defense/<project>/
    yield (PROJECT_ROOT.parent.parent / "models" / EMBEDDING_MODEL_NAME).resolve()
    yield (PROJECT_ROOT.parent / "models" / EMBEDDING_MODEL_NAME).resolve()


def resolve_embedding_model(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return an existing local SentenceTransformer directory or fail clearly."""

    tried: list[Path] = []
    for candidate in embedding_model_candidates(explicit):
        if candidate in tried:
            continue
        tried.append(candidate)
        if candidate.is_dir():
            return candidate

    rendered = "; ".join(str(path) for path in tried) or "<none>"
    raise FileNotFoundError(
        f"Local embedding model not found. Set {EMBEDDING_MODEL_ENV} to the "
        f"all-MiniLM-L6-v2 directory. Tried: {rendered}"
    )
