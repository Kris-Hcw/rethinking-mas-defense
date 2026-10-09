import unittest

from src.result_validity import attack_result_valid


def _selection(*, valid: bool = True, reason: str = "paper_objective_feasible_candidate") -> dict:
    return {
        "selection_reason": reason,
        "selected_metrics": {"formally_valid": valid},
    }


class AttackResultValidityTests(unittest.TestCase):
    def test_wrapper_rejects_legacy_approximate_payload_count(self):
        selection = _selection()
        selection["attack_type"] = "benign_wrapper"
        selection["selected_metrics"].update({
            "message": "ordinary analysis || short pivot",
            "wrapper": "ordinary analysis",
            "payload": "short pivot",
            "wrapper_serialization_valid": True,
            "payload_token_count_approx": 2,
        })
        self.assertFalse(
            attack_result_valid({
                "attack_type": "benign_wrapper",
                "rounds": 1,
                "attacker_ids": [0],
                "attack_selections": [selection],
            })
        )

    def test_wrapper_accepts_exact_serialization_and_tokenizer_count(self):
        selection = _selection()
        selection["attack_type"] = "benign_wrapper"
        selection["selected_metrics"].update({
            "message": "ordinary analysis || short pivot\n<ANSWER>: C",
            "wrapper": "ordinary analysis",
            "payload": "short pivot\n<ANSWER>: C",
            "answer": "C",
            "target_answered": True,
            "wrapper_serialization_valid": True,
            "payload_token_count": 6,
            "payload_tokenizer_model": "qwen3-4b",
        })
        self.assertTrue(
            attack_result_valid({
                "attack_type": "benign_wrapper",
                "target_wrong": "C",
                "rounds": 1,
                "attacker_ids": [0],
                "attack_selections": [selection],
            })
        )

    def test_wrapper_rejects_hidden_answer_not_present_in_payload(self):
        selection = _selection()
        selection["attack_type"] = "benign_wrapper"
        selection["selected_metrics"].update({
            "message": "ordinary analysis || short pivot",
            "wrapper": "ordinary analysis",
            "payload": "short pivot",
            "answer": "C",
            "target_answered": True,
            "wrapper_serialization_valid": True,
            "payload_token_count": 2,
            "payload_tokenizer_model": "qwen3-4b",
        })
        self.assertFalse(
            attack_result_valid({
                "attack_type": "benign_wrapper",
                "target_wrong": "C",
                "rounds": 1,
                "attacker_ids": [0],
                "attack_selections": [selection],
            })
        )

    def test_wrapper_rejects_visible_answer_that_differs_from_recorded_target(self):
        selection = _selection()
        selection["attack_type"] = "benign_wrapper"
        selection["selected_metrics"].update({
            "message": "ordinary analysis || short pivot\n<ANSWER>: B",
            "wrapper": "ordinary analysis",
            "payload": "short pivot\n<ANSWER>: B",
            "answer": "C",
            "target_answered": True,
            "wrapper_serialization_valid": True,
            "payload_token_count": 6,
            "payload_tokenizer_model": "qwen3-4b",
        })
        self.assertFalse(
            attack_result_valid({
                "attack_type": "benign_wrapper",
                "target_wrong": "C",
                "rounds": 1,
                "attacker_ids": [0],
                "attack_selections": [selection],
            })
        )

    def test_clean_row_needs_no_attack_selection(self):
        self.assertTrue(attack_result_valid({"attack_type": "none"}))

    def test_obvious_row_fails_closed_without_selections(self):
        self.assertFalse(
            attack_result_valid(
                {"attack_type": "obvious", "rounds": 1, "attacker_ids": [0]}
            )
        )

    def test_obvious_row_accepts_exact_valid_selection_count(self):
        self.assertTrue(
            attack_result_valid(
                {
                    "attack_type": "obvious",
                    "rounds": 1,
                    "attacker_ids": [0],
                    "round_results": [{
                        "agents": [{
                            "agent_id": 0,
                            "role": "attacker",
                            "answer": "B",
                            "selection": _selection(),
                        }]
                    }],
                }
            )
        )

    def test_current_schema_clean_row_rejects_unparsed_agent_answer(self):
        self.assertFalse(
            attack_result_valid(
                {
                    "attack_type": "none",
                    "per_agent_answers": ["A", None],
                    "round_results": [
                        {
                            "agents": [
                                {"agent_id": 0, "role": "benign", "answer": "A"},
                                {"agent_id": 1, "role": "benign", "answer": None},
                            ]
                        }
                    ],
                }
            )
        )

    def test_obvious_row_rejects_fallback_and_false_summary(self):
        self.assertFalse(
            attack_result_valid(
                {
                    "attack_type": "obvious",
                    "attack_valid": False,
                    "rounds": 1,
                    "attacker_ids": [0],
                    "round_results": [{
                        "agents": [{
                            "role": "attacker",
                            "selection": _selection(
                                valid=False,
                                reason="debug_minimum_violation_fallback",
                            ),
                        }]
                    }],
                }
            )
        )

    def test_legacy_attack_selections_are_validated(self):
        self.assertTrue(
            attack_result_valid(
                {
                    "attack_kind": "obvious",
                    "rounds": 1,
                    "attacker_ids": [0],
                    "attack_selections": [_selection()],
                }
            )
        )

    def test_legacy_marker_heuristic_does_not_override_paper_constraints(self):
        selection = {
            "selection_reason": "paper_objective_feasible_candidate",
            "selected_metrics": {
                "constraint_satisfied": True,
                "target_answered": True,
                "natural_language_check": False,
            },
        }
        self.assertTrue(
            attack_result_valid(
                {
                    "attack_type": "slow_drift",
                    "rounds": 1,
                    "attacker_ids": [0],
                    "attack_selections": [selection],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
