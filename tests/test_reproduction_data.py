import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_reproduction_data import prepare_bbh, prepare_gsm8k


class ReproductionDataTests(unittest.TestCase):
    def test_gsm8k_extracts_final_number(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.jsonl"
            source.write_text(json.dumps({"question": "How many?", "answer": "work\n#### 1,234"}) + "\n", encoding="utf-8")
            rows = prepare_gsm8k(source, 1, 7)
            self.assertEqual(rows[0]["gold"], "1234")

    def test_bbh_keeps_non_abc_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, target in (("one", "A"), ("two", "F"), ("three", "True")):
                (root / f"{name}.json").write_text(json.dumps({"examples": [{"input": "Q", "target": target}]}), encoding="utf-8")
            rows = prepare_bbh(root, 3, 7)
            self.assertEqual({row["gold"] for row in rows}, {"A", "F", "True"})


if __name__ == "__main__":
    unittest.main()
