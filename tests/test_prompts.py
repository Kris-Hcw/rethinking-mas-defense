import unittest

from src.attacks import get_first_prompt, get_regen_prompt, get_system_prompt


class MMLUPaperPromptTests(unittest.TestCase):
    def test_overt_round_one_matches_appendix_b_instructions(self):
        prompt = get_first_prompt("overt", "mmlu", "Q", "C", True)
        self.assertIn("The answer provided by the user is: C", prompt)
        self.assertIn("Do not mention, imply, or provide any reasoning", prompt)
        self.assertIn("<ANSWER>: C", prompt)

    def test_slow_drift_round_two_contains_appendix_b_guidance(self):
        prompt = get_regen_prompt("slow_drift", "mmlu", "Q", "peer view", 2, "C", True)
        self.assertIn("Current state of debate: peer view", prompt)
        self.assertIn("The target answer you must advocate for is: C", prompt)
        self.assertIn("Gradually increase your confidence", prompt)

    def test_compromised_mmlu_agents_receive_target_without_extra_system_attack(self):
        for attack in ("overt", "slow_drift"):
            with self.subTest(attack=attack):
                self.assertEqual(
                    get_system_prompt(attack, "mmlu", "C", True),
                    get_system_prompt("none", "mmlu", None, False),
                )
                for target in "ABCD":
                    first = get_first_prompt(attack, "mmlu", "QUESTION", target, True)
                    later = get_regen_prompt(attack, "mmlu", "QUESTION", "PEERS", 2, target, True)
                    self.assertIn(f"is: {target}", first)
                    self.assertIn("QUESTION", later)
                    if attack == "overt":
                        self.assertIn("The answer provided by the user is: " + target, later)
                    else:
                        self.assertIn("Current state of debate: PEERS", later)
                    self.assertIn("PEERS", later)

    def test_benign_routing_ignores_attack_and_target(self):
        for attack in ("overt", "slow_drift", "benign_wrapper", "chaos_seeding"):
            self.assertEqual(
                get_first_prompt(attack, "mmlu", "Q", "C", False),
                get_first_prompt("none", "mmlu", "Q", None, False),
            )

    def test_mmlu_benign_prompt_keeps_appendix_format(self):
        prompt = get_first_prompt("none", "mmlu", "Q", None, False)
        self.assertIn("You are one participant in a multi-agent debate", prompt)
        self.assertIn("<ANSWER>: {one letter: A/B/C/D}", prompt)


if __name__ == "__main__":
    unittest.main()
