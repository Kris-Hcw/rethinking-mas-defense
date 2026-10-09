from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import runtime_paths


class RuntimePathTests(unittest.TestCase):
    def test_explicit_embedding_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(runtime_paths.resolve_embedding_model(tmp), Path(tmp).resolve())

    def test_environment_embedding_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {runtime_paths.EMBEDDING_MODEL_ENV: tmp}
        ):
            self.assertEqual(runtime_paths.resolve_embedding_model(), Path(tmp).resolve())

    def test_project_relative_windows_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SEU-Project-3" / "Confidence-Guided Defense" / "repo"
            model = Path(tmp) / "SEU-Project-3" / "models" / runtime_paths.EMBEDDING_MODEL_NAME
            root.mkdir(parents=True)
            model.mkdir(parents=True)
            with patch.object(runtime_paths, "PROJECT_ROOT", root):
                self.assertEqual(runtime_paths.resolve_embedding_model(), model.resolve())

    def test_missing_path_reports_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            runtime_paths, "PROJECT_ROOT", Path(tmp) / "missing" / "repo"
        ), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, runtime_paths.EMBEDDING_MODEL_ENV):
                runtime_paths.resolve_embedding_model()


if __name__ == "__main__":
    unittest.main()
