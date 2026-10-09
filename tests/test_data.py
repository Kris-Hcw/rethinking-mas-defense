import json
import tempfile
import unittest
from pathlib import Path

from src.attacks import choose_wrong_target, get_first_prompt, get_system_prompt
from src.data import load_local
from src.gsm8k_eval import normalize_numeric_answer


class DataLoaderTests(unittest.TestCase):
    def test_gsm8k_numeric_answers_are_normalized(self):
        self.assertEqual(normalize_numeric_answer("#### 1,200"), "1200")
        self.assertEqual(normalize_numeric_answer("<ANSWER>: $-2.50"), "-2.5")
        self.assertEqual(normalize_numeric_answer("The answer is 6."), "6")

    def test_gsm8k_loader_preserves_numeric_adapter(self):
        row = {
            "question": "How many apples?",
            "gold": "#### 1,200",
            "choices": [],
            "subject": "gsm8k",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gsm8k.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = load_local(str(path), dataset="gsm8k")

        self.assertEqual(records[0]["gold"], "1200")
        self.assertEqual(records[0]["answer_adapter"], "gsm8k_numeric")
        self.assertEqual(records[0]["choices"], [])

    def test_bbh_raw_labels_are_adapted_without_rewriting_raw_fields(self):
        row = {
            "input": "Who does the pronoun refer to?",
            "target": "Alice",
            "choices": ["Alice", "Bob", "Neither"],
            "task": "disambiguation_qa",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bbh.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = load_local(str(path), dataset="bbh")

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["gold"], "A")
        self.assertEqual(record["raw_target"], "Alice")
        self.assertEqual(record["raw_choices"], ["Alice", "Bob", "Neither"])
        self.assertEqual(record["answer_adapter"], "bbh_raw_label_adapter")
        self.assertEqual(record["answer_options"], ["A", "B", "C"])
        self.assertIn("A. Alice", record["question"])

    def test_bbh_four_choice_domain_reaches_prompts_and_attack_targets(self):
        row = {
            "input": "Which event happened last?",
            "target": "fourth",
            "choices": ["first", "second", "third", "fourth"],
            "task": "date_understanding",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bbh.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = load_local(str(path), dataset="bbh")[0]

        self.assertEqual(record["gold"], "D")
        self.assertEqual(record["answer_options"], ["A", "B", "C", "D"])
        system = get_system_prompt(
            "none",
            "bbh",
            None,
            False,
            answer_options=record["answer_options"],
        )
        prompt = get_first_prompt(
            "none",
            "bbh",
            record["question"],
            None,
            False,
            answer_options=record["answer_options"],
        )
        self.assertIn("pick A, B, C, or D", system)
        self.assertIn("<ANSWER>: {one letter: A/B/C/D}", prompt)

        targets = {
            choose_wrong_target(
                "A",
                "bbh",
                seed=seed,
                answer_options=record["answer_options"],
            )
            for seed in range(32)
        }
        self.assertLessEqual(targets, {"B", "C", "D"})
        self.assertIn("D", targets)


if __name__ == "__main__":
    unittest.main()
