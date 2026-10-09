import math
import unittest

from src.llm_client import LogprobsUnavailableError, TokenInfo
from src.confidence import (
    FullVocabularyLogprobsUnavailableError,
    compute_confidence,
    confidence_weight,
    entropy_from_top_logprobs,
)


class ConfidenceTests(unittest.TestCase):
    def test_entropy_and_top_k_confidence(self):
        uniform = [("a", math.log(0.5)), ("b", math.log(0.5))]
        peaked = [("a", math.log(0.9)), ("b", math.log(0.1))]
        infos = [
            TokenInfo("a", math.log(0.5), uniform),
            TokenInfo("a", math.log(0.9), peaked),
        ]
        expected = math.exp(-entropy_from_top_logprobs(uniform))
        self.assertAlmostEqual(compute_confidence(infos, top_k=1), expected)

    def test_missing_logprobs_is_explicit_error(self):
        with self.assertRaises(LogprobsUnavailableError):
            compute_confidence([], top_k=10)

    def test_confidence_weight_clamps(self):
        self.assertIsNone(confidence_weight(None))
        self.assertEqual(confidence_weight(-1.0), 0.0)
        self.assertEqual(confidence_weight(2.0), 1.0)
        self.assertEqual(confidence_weight(0.2, floor=0.4), 0.4)

    def test_exact_entropy_fails_closed_on_truncated_distribution(self):
        truncated = [("a", math.log(0.6)), ("b", math.log(0.2))]
        with self.assertRaises(FullVocabularyLogprobsUnavailableError):
            entropy_from_top_logprobs(truncated, mode="exact_full_vocab")

    def test_exact_entropy_accepts_complete_distribution(self):
        complete = [("a", math.log(0.6)), ("b", math.log(0.4))]
        expected = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        self.assertAlmostEqual(
            entropy_from_top_logprobs(complete, mode="exact_full_vocab"), expected
        )

    def test_exact_entropy_accepts_backend_reduced_full_vocab_entropy(self):
        infos = [
            TokenInfo(
                "a",
                math.log(0.6),
                [("a", math.log(0.6))],
                full_vocab_entropy=0.75,
                full_vocab_size=151936,
                full_vocab_entropy_source="raw_generation_logits",
            ),
            TokenInfo(
                "b",
                math.log(0.7),
                [("b", math.log(0.7))],
                full_vocab_entropy=0.25,
                full_vocab_size=151936,
                full_vocab_entropy_source="raw_generation_logits",
            ),
        ]
        self.assertAlmostEqual(
            compute_confidence(infos, top_k=1, entropy_mode="exact_full_vocab"),
            math.exp(-0.75),
        )

    def test_exact_entropy_rejects_missing_scalar_metadata(self):
        infos = [TokenInfo(
            "a",
            math.log(0.6),
            [("a", math.log(0.6))],
            full_vocab_entropy=0.75,
        )]
        with self.assertRaises(FullVocabularyLogprobsUnavailableError):
            compute_confidence(infos, top_k=1, entropy_mode="exact_full_vocab")

    def test_exact_entropy_rejects_inconsistent_vocab_contract(self):
        infos = [
            TokenInfo(
                "a", math.log(0.6), [("a", math.log(0.6))],
                full_vocab_entropy=0.75,
                full_vocab_size=151936,
                full_vocab_entropy_source="raw_generation_logits",
            ),
            TokenInfo(
                "b", math.log(0.7), [("b", math.log(0.7))],
                full_vocab_entropy=0.25,
                full_vocab_size=32000,
                full_vocab_entropy_source="raw_generation_logits",
            ),
        ]
        with self.assertRaises(FullVocabularyLogprobsUnavailableError):
            compute_confidence(infos, top_k=1, entropy_mode="exact_full_vocab")


if __name__ == "__main__":
    unittest.main()
