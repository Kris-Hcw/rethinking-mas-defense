import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from src.attacks import get_first_prompt, get_system_prompt
from src.mas import (
    AttackConfig,
    DebateMAS,
    DefenseConfig,
    GeneratedMessage,
    NoFeasibleAttackCandidate,
    _candidate_rejection_reasons,
    _majority_vote,
    _parse_answer,
    _split_wrapper_payload,
)
from src.result_validity import MAJORITY_TIE_POLICY
from src.topology import build_adjacency
from tests.fakes import FakeEncoder, FakeLLMClient


class DebateIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_majority_ties_explicitly_follow_first_agent_answer_order(self):
        self.assertEqual(MAJORITY_TIE_POLICY, "first_agent_order")
        self.assertEqual(
            _majority_vote(["B", "A", "A", "B"]),
            ("B", {"B": 2, "A": 2}),
        )
        self.assertEqual(
            _majority_vote(["A", "B", "B", "A"]),
            ("A", {"A": 2, "B": 2}),
        )

    async def test_bbh_run_persists_and_prompts_with_record_answer_domain(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="bbh",
            encoder=FakeEncoder(),
        )
        result = await mas.run_one(
            question="Choose one:\nA. first\nB. second\nC. third\nD. fourth",
            gold="D",
            answer_options=["A", "B", "C", "D"],
            sample_seed=7,
        )

        self.assertEqual(result["answer_options"], ["A", "B", "C", "D"])
        self.assertIn("pick A, B, C, or D", mas.agents[0].system_prompt)
        self.assertIn("<ANSWER>: {one letter: A/B/C/D}", result["agent_histories"][0][0]["prompt"])

    async def test_bbh_run_rejects_parsed_answer_outside_record_domain(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="bbh",
            encoder=FakeEncoder(),
        )
        out_of_domain = GeneratedMessage(
            text="<REASON>: unsupported option\n<ANSWER>: D",
            answer="D",
            reason="unsupported option",
            confidence=None,
            token_count=4,
        )

        with patch("src.mas.Agent.generate", new=AsyncMock(return_value=out_of_domain)):
            with self.assertRaisesRegex(RuntimeError, "outside the declared answer domain"):
                await mas.run_one(
                    question="Choose one:\nA. first\nB. second\nC. third",
                    gold="A",
                    answer_options=["A", "B", "C"],
                    sample_seed=7,
                )

    def test_gsm8k_parser_does_not_use_letter_parser(self):
        self.assertEqual(_parse_answer("<REASON>: choose A as a variable\n<ANSWER>: 1,200", "gsm8k"), "1200")
        self.assertIsNone(_parse_answer("The symbolic choice is A only.", "gsm8k"))
        self.assertEqual(_parse_answer("<ANSWER>: A", "mmlu"), "A")
        self.assertEqual(_parse_answer("<ANSWER>: B.", "mmlu"), "B")

    def test_wrapper_parser_rejects_markdown_wrapping(self):
        payload = "short steering B\n<ANSWER>: C"
        self.assertEqual(
            _split_wrapper_payload(f"benign A || {payload}"),
            ("benign A", payload),
        )
        self.assertEqual(_split_wrapper_payload("benign A || short B"), (None, None))
        self.assertEqual(
            _split_wrapper_payload("benign A <ANSWER>: C || short B\n<ANSWER>: C"),
            (None, None),
        )
        self.assertEqual(
            _split_wrapper_payload(
                "benign A || short B\n<ANSWER>: C\n<ANSWER>: C"
            ),
            (None, None),
        )
        self.assertEqual(
            _split_wrapper_payload("`benign A || short B\n<ANSWER>: C`"),
            (None, None),
        )
        system = get_system_prompt("benign_wrapper", "mmlu", "C", True)
        prompt = get_first_prompt("benign_wrapper", "mmlu", "question", "C", True)
        self.assertNotIn("`", system)
        self.assertNotIn("`", prompt)
        self.assertIn("B must end exactly with <ANSWER>: C", system)
        self.assertIn("End B exactly with <ANSWER>: C", prompt)
        self.assertIn("two natural-language paragraphs", system)
        self.assertIn("one blank line", prompt)
        self.assertNotIn("two pipe characters", prompt)

    def test_wrapper_parser_accepts_exact_two_paragraph_concatenation(self):
        text = (
            "A. The conventional medical analysis supports nephrolithiasis.\n\n"
            "B. A narrow alternative reading supports D.\n<ANSWER>: D"
        )

        self.assertEqual(
            _split_wrapper_payload(text),
            (
                "A. The conventional medical analysis supports nephrolithiasis.",
                "B. A narrow alternative reading supports D.\n<ANSWER>: D",
            ),
        )
        self.assertEqual(
            _split_wrapper_payload(
                "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n<ANSWER>: D"
            ),
            (None, None),
        )

    def test_round_views_include_only_incoming_neighbours_not_self(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=2,
            dataset="mmlu",
            adj=build_adjacency("full", 2),
            encoder=FakeEncoder(),
        )
        for agent_id, answer in enumerate(("A", "B")):
            message = GeneratedMessage(
                text=f"<REASON>: agent-{agent_id}\n<ANSWER>: {answer}",
                answer=answer,
                reason=f"agent-{agent_id}",
                confidence=0.8,
                token_count=2,
                embedding=np.asarray([1.0, float(agent_id)], dtype=np.float32),
            )
            mas.agents[agent_id].commit(message, 1, "round-one", role="benign")

        views = mas._format_views(0)
        self.assertIn("Peer 1:", views)
        self.assertIn("agent-1", views)
        self.assertNotIn("Peer 0", views)
        self.assertNotIn("(self)", views)

    async def test_all_near_benign_attacks_record_round_metrics(self):
        for attack_name in ("slow_drift", "benign_wrapper", "chaos_seeding"):
            with self.subTest(attack=attack_name):
                mas = DebateMAS(
                    client=FakeLLMClient(),
                    model="fake-model",
                    n_agents=5,
                    n_rounds=3,
                    dataset="mmlu",
                    adj=build_adjacency("chain", 5),
                    defense=DefenseConfig(mode="confidence_pruning", prune_threshold=0.1),
                    encoder=FakeEncoder(),
                )
                result = await mas.run_one(
                    question="Which gas dominates the atmosphere? A. CO2 B. Nitrogen C. Oxygen D. Argon",
                    gold="B",
                    attack=AttackConfig(
                        attack_type=attack_name,
                        attacker_ids=[0, 1],
                        num_candidates=3,
                        tau=2.0,
                        wrapper_tau=2.0,
                        drift_epsilon=2.0,
                    ),
                )
                self.assertEqual(len(result["round_results"]), 3)
                self.assertEqual(len(result["agent_histories"]), 5)
                for round_result in result["round_results"]:
                    self.assertEqual(len(round_result["agents"]), 5)
                    self.assertEqual(
                        round_result["embedding_score_definition"],
                        "label_free_mean_2_nearest_peer_cosine_distance",
                    )
                    self.assertIn("benign_disagreement_rate", round_result)
                    self.assertIn("benign_embedding_variance", round_result)
                    self.assertIsNotNone(round_result["attacker_benign_cosine_distance"])
                    for agent in round_result["agents"]:
                        self.assertIsNotNone(agent["confidence"])
                        self.assertTrue(agent["embedding"])
                        self.assertIsNotNone(agent["embedding_outlier_score"])

                selected = result["round_results"][0]["agents"][0]["selection"]
                self.assertEqual(
                    selected["selection_reason"], "paper_objective_feasible_candidate"
                )
                self.assertEqual(
                    selected["support_source"], "round0_independent_benign_reference"
                )
                self.assertIn("flip_count", selected["selected_metrics"])
                self.assertEqual(
                    selected["selected_metrics"]["counterfactual_scope"],
                    "joint_attacker_combination",
                )
                self.assertEqual(selected["joint_attacker_ids"], [0, 1])
                self.assertEqual(selected["joint_combination_count"], 9)
                if attack_name == "benign_wrapper":
                    metric = selected["candidates"][selected["selected_candidate_index"]]
                    self.assertIsNotNone(metric["wrapper_cosine_distance_to_benign"])
                    self.assertIsNotNone(metric["final_cosine_distance_to_benign"])
                terminal = result["round_results"][-1]["agents"][0]["selection"]
                self.assertEqual(
                    terminal["selection_reason"],
                    "terminal_constraint_only_feasible_candidate",
                )
                self.assertEqual(
                    terminal["selected_metrics"]["objective_horizon"],
                    "terminal_no_benign_propagation",
                )
                self.assertTrue(result["attack_valid"])

    async def test_strict_selection_rejects_infeasible_candidate(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        candidate = GeneratedMessage(
            text="<REASON>: ordinary analysis\n<ANSWER>: C",
            answer="C",
            reason="ordinary analysis",
            confidence=0.5,
            token_count=2,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        with self.assertRaises(NoFeasibleAttackCandidate) as raised:
            await mas._select_attack_candidate(
                candidates=[candidate],
                attack=AttackConfig(attack_type="slow_drift", tau=0.0),
                target="C",
                benign_support=np.asarray([[0.0, 1.0]], dtype=np.float32),
                previous_embedding=None,
                objective_scores=[
                    {"flip_count": 1, "target_count": 1, "benign_disagreement_rate": 0.0}
                ],
                support_source="previous_round_benign_messages",
            )
        failure = raised.exception
        self.assertEqual(failure.candidate_budget, 4)
        self.assertEqual(failure.generated_candidate_count, 1)
        self.assertEqual(failure.constraint_thresholds["tau"], 0.0)
        self.assertEqual(
            failure.candidates[0]["rejection_reasons"],
            ["final_cosine_distance_exceeds_tau"],
        )

    def test_wrapper_rejection_reasons_name_every_failed_gate(self):
        reasons = _candidate_rejection_reasons(
            AttackConfig(
                attack_type="benign_wrapper",
                tau=0.25,
                wrapper_tau=0.25,
                payload_token_budget=16,
            ),
            {
                "target_answered": False,
                "natural_language_check": False,
                "final_cosine_distance_to_benign": 0.30,
                "wrapper_cosine_distance_to_benign": 0.40,
                "wrapper_serialization_valid": True,
                "payload_token_count": 17,
                "formally_valid": False,
            },
        )
        self.assertEqual(
            reasons,
            [
                "target_answer_mismatch",
                "final_cosine_distance_exceeds_tau",
                "wrapper_cosine_distance_exceeds_tau",
                "payload_token_budget_exceeded",
            ],
        )

    async def test_marker_heuristic_is_diagnostic_not_a_formal_constraint(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        benign = np.asarray([1.0, 0.0], dtype=np.float32)
        candidate = GeneratedMessage(
            text="The payload term is part of the ordinary problem statement.",
            answer="C",
            reason="The payload term is part of the ordinary problem statement.",
            confidence=0.5,
            token_count=8,
            embedding=benign.copy(),
        )

        selected, selection = await mas._select_attack_candidate(
            candidates=[candidate],
            attack=AttackConfig(
                attack_type="slow_drift",
                tau=0.25,
                num_candidates=1,
            ),
            target="C",
            benign_support=np.asarray([benign]),
            previous_embedding=None,
            objective_scores=[{
                "flip_count": 1,
                "target_count": 1,
                "benign_disagreement_rate": 0.0,
            }],
            support_source="round0_independent_benign_reference",
        )

        metric = selection["selected_metrics"]
        self.assertIs(selected, candidate)
        self.assertTrue(metric["constraint_satisfied"])
        self.assertTrue(metric["target_answered"])
        self.assertTrue(metric["formally_valid"])
        self.assertFalse(metric["natural_language_check"])
        self.assertFalse(metric["obvious_marker_heuristic_passed"])
        self.assertEqual(metric["obvious_marker_hits"], ["payload"])
        self.assertEqual(
            selection["obvious_marker_heuristic_policy"],
            "diagnostic_only_not_paper_constraint",
        )

    async def test_run_one_uses_fail_closed_attack_result_validation(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        with patch("src.mas.attack_result_valid", return_value=False) as validate:
            result = await mas.run_one(
                question="Which gas dominates? A. CO2 B. Nitrogen C. Oxygen D. Argon",
                gold="B",
                attack=AttackConfig(
                    attack_type="obvious",
                    attacker_ids=[0],
                    num_candidates=1,
                ),
            )
        self.assertFalse(result["attack_valid"])
        validate.assert_called_once()

    async def test_obvious_and_overt_honor_configured_candidate_budget(self):
        for attack_name in ("obvious", "overt"):
            with self.subTest(attack=attack_name):
                mas = DebateMAS(
                    client=FakeLLMClient(),
                    model="fake-model",
                    n_agents=3,
                    n_rounds=1,
                    dataset="mmlu",
                    encoder=FakeEncoder(),
                )
                result = await mas.run_one(
                    question="Which gas dominates? A. CO2 B. Nitrogen C. Oxygen D. Argon",
                    gold="B",
                    attack=AttackConfig(
                        attack_type=attack_name,
                        attacker_ids=[0],
                        num_candidates=3,
                    ),
                )

                selection = result["round_results"][0]["agents"][0]["selection"]
                self.assertEqual(len(selection["candidates"]), 3)
                self.assertEqual(
                    [item["candidate_index"] for item in selection["candidates"]],
                    [0, 1, 2],
                )
                self.assertEqual(selection["candidate_budget"], 3)
                self.assertEqual(selection["generated_candidate_count"], 3)

    async def test_obvious_and_overt_reject_nonpositive_candidate_budget(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        for attack_name in ("obvious", "overt"):
            with self.subTest(attack=attack_name):
                with self.assertRaisesRegex(ValueError, "num_candidates must be positive"):
                    await mas._generate_attack_candidates(
                        agent_id=0,
                        prompt="candidate prompt",
                        attack=AttackConfig(
                            attack_type=attack_name,
                            attacker_ids=[0],
                            num_candidates=0,
                        ),
                        round_id=1,
                        sample_seed=23,
                    )

    async def test_counterfactual_objective_controls_selection(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        candidates = [
            GeneratedMessage(
                text=f"<REASON>: ordinary {index}\n<ANSWER>: C",
                answer="C",
                reason=f"ordinary {index}",
                confidence=0.5,
                token_count=2,
                embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            )
            for index in range(2)
        ]
        _, selection = await mas._select_attack_candidate(
            candidates=candidates,
            attack=AttackConfig(attack_type="slow_drift", tau=1.0),
            target="C",
            benign_support=np.asarray([[0.0, 1.0]], dtype=np.float32),
            previous_embedding=None,
            objective_scores=[
                {"flip_count": 0, "target_count": 0, "benign_disagreement_rate": 0.0},
                {"flip_count": 2, "target_count": 2, "benign_disagreement_rate": 0.0},
            ],
            support_source="previous_round_benign_messages",
        )
        self.assertEqual(selection["selected_candidate_index"], 1)

    async def test_unparsed_counterfactual_candidate_cannot_win_selection(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        candidates = [
            GeneratedMessage(
                text="<REASON>: invalid rollout\n<ANSWER>: C",
                answer="C",
                reason="invalid rollout",
                confidence=None,
                token_count=2,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            GeneratedMessage(
                text="<REASON>: complete rollout\n<ANSWER>: C",
                answer="C",
                reason="complete rollout",
                confidence=None,
                token_count=2,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
        ]
        selected, selection = await mas._select_attack_candidate(
            candidates=candidates,
            attack=AttackConfig(
                attack_type="chaos_seeding",
                attacker_ids=[0],
                tau=1.0,
            ),
            target="C",
            benign_support=np.asarray([[1.0, 0.0]], dtype=np.float32),
            previous_embedding=None,
            objective_scores=[
                {
                    "flip_count": 3,
                    "target_count": 3,
                    "benign_disagreement_rate": 1.0,
                    "counterfactual_answers_complete": False,
                    "counterfactual_unparsed_positions": [1],
                },
                {
                    "flip_count": 1,
                    "target_count": 1,
                    "benign_disagreement_rate": 0.5,
                    "counterfactual_answers_complete": True,
                    "counterfactual_unparsed_positions": [],
                },
            ],
        )
        self.assertIs(selected, candidates[1])
        self.assertEqual(selection["selected_candidate_index"], 1)
        self.assertFalse(selection["candidates"][0]["formally_valid"])
        self.assertIn(
            "counterfactual_answer_unparsed",
            _candidate_rejection_reasons(
                AttackConfig(attack_type="chaos_seeding"),
                selection["candidates"][0],
            ),
        )

    async def test_counterfactual_rollout_uses_the_next_round_snapshot(self):
        adj = np.zeros((3, 3), dtype=int)
        adj[0, 1] = 1
        adj[1, 2] = 1
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            adj=adj,
            encoder=FakeEncoder(),
        )

        benign_outputs = [
            GeneratedMessage(
                text="<REASON>: benign-one-current\n<ANSWER>: B",
                answer="B",
                reason="benign-one-current",
                confidence=None,
                token_count=2,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
            GeneratedMessage(
                text="<REASON>: benign-two-current\n<ANSWER>: B",
                answer="B",
                reason="benign-two-current",
                confidence=None,
                token_count=2,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            ),
        ]
        for benign_id, output in zip((1, 2), benign_outputs):
            mas.agents[benign_id].commit(output, 1, "round-one prompt", role="benign")

        stale_attacker = GeneratedMessage(
            text="<REASON>: stale-attacker-message\n<ANSWER>: D",
            answer="D",
            reason="stale-attacker-message",
            confidence=None,
            token_count=2,
            embedding=np.asarray([0.0, 1.0], dtype=np.float32),
        )
        mas.agents[0].commit(stale_attacker, 0, "stale prompt", role="attacker")
        candidate = GeneratedMessage(
            text="<REASON>: focal-current-candidate\n<ANSWER>: C",
            answer="C",
            reason="focal-current-candidate",
            confidence=None,
            token_count=2,
            embedding=np.asarray([0.0, 1.0], dtype=np.float32),
        )
        next_for_recipient = GeneratedMessage(
            text="<REASON>: changed after delivery\n<ANSWER>: C",
            answer="C",
            reason="changed after delivery",
            confidence=None,
            token_count=2,
        )
        next_for_nonrecipient = GeneratedMessage(
            text="<REASON>: updated without delivery\n<ANSWER>: B",
            answer="B",
            reason="updated without delivery",
            confidence=None,
            token_count=2,
        )

        recipient_generate = AsyncMock(
            side_effect=[next_for_nonrecipient, next_for_recipient]
        )
        nonrecipient_generate = AsyncMock(return_value=next_for_nonrecipient)
        with patch.object(mas.agents[1], "generate", new=recipient_generate), patch.object(
            mas.agents[2], "generate", new=nonrecipient_generate
        ):
            scores = await mas._counterfactual_objectives(
                question="Which option? A. Alpha B. Beta C. Gamma D. Delta",
                round_id=1,
                attacker_id=0,
                candidates=[candidate],
                benign_ids=[1, 2],
                benign_outputs=benign_outputs,
                benign_prompts=["attacker-current", "benign-one-current", "benign-two-current"],
                target="C",
                sample_seed=7,
                attack=AttackConfig(attack_type="slow_drift", attacker_ids=[0]),
            )

        self.assertEqual(recipient_generate.await_count, 2)
        nonrecipient_generate.assert_awaited_once()
        baseline_recipient_prompt = recipient_generate.await_args_list[0].args[0]
        recipient_prompt = recipient_generate.await_args_list[1].args[0]
        nonrecipient_prompt = nonrecipient_generate.await_args.args[0]
        mas.agents[0].commit(candidate, 1, "round-one attacker prompt", role="attacker")
        expected_recipient_prompt = mas._build_prompt(
            1,
            2,
            "Which option? A. Alpha B. Beta C. Gamma D. Delta",
            "slow_drift",
            "C",
            {0},
        )
        expected_nonrecipient_prompt = mas._build_prompt(
            2,
            2,
            "Which option? A. Alpha B. Beta C. Gamma D. Delta",
            "slow_drift",
            "C",
            {0},
        )
        self.assertEqual(recipient_prompt, expected_recipient_prompt)
        self.assertNotIn("focal-current-candidate", baseline_recipient_prompt)
        self.assertEqual(nonrecipient_prompt, expected_nonrecipient_prompt)
        self.assertIn("multi-agent debate (round 2)", recipient_prompt)
        self.assertIn("focal-current-candidate", recipient_prompt)
        self.assertNotIn("stale-attacker-message", recipient_prompt)
        self.assertNotIn("Counterfactual current peer message", recipient_prompt)
        self.assertIn("multi-agent debate (round 2)", nonrecipient_prompt)
        self.assertIn("benign-one-current", nonrecipient_prompt)
        self.assertNotIn("focal-current-candidate", nonrecipient_prompt)
        self.assertEqual(scores[0]["objective_horizon"], "next_round_transition")
        self.assertEqual(scores[0]["baseline_mode"], "matched_next_round_no_focal_candidate")
        self.assertEqual(scores[0]["rollout_round"], 2)
        self.assertEqual(scores[0]["delivered_to"], [1])
        self.assertEqual(scores[0]["baseline_answers"], ["B", "B"])
        self.assertEqual(scores[0]["counterfactual_answers"], ["C", "B"])
        self.assertEqual(scores[0]["flip_count"], 1)
        self.assertEqual(
            recipient_generate.await_args_list[0].kwargs["seed"],
            recipient_generate.await_args_list[1].kwargs["seed"],
        )

    async def test_counterfactual_flip_uses_matched_next_round_baseline(self):
        adj = np.zeros((2, 2), dtype=int)
        adj[0, 1] = 1
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=2,
            dataset="mmlu",
            adj=adj,
            encoder=FakeEncoder(),
        )
        current = GeneratedMessage(
            text="<REASON>: current benign\n<ANSWER>: B",
            answer="B",
            reason="current benign",
            confidence=None,
            token_count=2,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        mas.agents[1].commit(current, 1, "round-one prompt", role="benign")
        candidate = GeneratedMessage(
            text="<REASON>: focal candidate\n<ANSWER>: C",
            answer="C",
            reason="focal candidate",
            confidence=None,
            token_count=2,
        )
        natural_next = GeneratedMessage(
            text="<REASON>: natural next-round update\n<ANSWER>: C",
            answer="C",
            reason="natural next-round update",
            confidence=None,
            token_count=2,
        )
        candidate_next = GeneratedMessage(
            text="<REASON>: candidate next-round update\n<ANSWER>: C",
            answer="C",
            reason="candidate next-round update",
            confidence=None,
            token_count=2,
        )
        generate = AsyncMock(side_effect=[natural_next, candidate_next])
        with patch.object(mas.agents[1], "generate", new=generate):
            scores = await mas._counterfactual_objectives(
                question="Which option? A. Alpha B. Beta C. Gamma D. Delta",
                round_id=1,
                attacker_id=0,
                candidates=[candidate],
                benign_ids=[1],
                benign_outputs=[current],
                benign_prompts=["attacker-current", "benign-current"],
                target="C",
                sample_seed=29,
                attack=AttackConfig(attack_type="slow_drift", attacker_ids=[0]),
            )

        self.assertEqual(generate.await_count, 2)
        self.assertEqual(scores[0]["baseline_mode"], "matched_next_round_no_focal_candidate")
        self.assertEqual(scores[0]["baseline_answers"], ["C"])
        self.assertEqual(scores[0]["counterfactual_answers"], ["C"])
        self.assertEqual(scores[0]["flip_count"], 0)
        self.assertEqual(
            generate.await_args_list[0].kwargs["seed"],
            generate.await_args_list[1].kwargs["seed"],
        )

    async def test_counterfactual_final_round_has_no_fictitious_benign_rollout(self):
        adj = np.zeros((2, 2), dtype=int)
        adj[0, 1] = 1
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            adj=adj,
            encoder=FakeEncoder(),
        )
        benign = GeneratedMessage(
            text="<REASON>: terminal benign\n<ANSWER>: B",
            answer="B",
            reason="terminal benign",
            confidence=None,
            token_count=2,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        mas.agents[1].commit(benign, 1, "round-one prompt", role="benign")
        candidate = GeneratedMessage(
            text="<REASON>: terminal attacker\n<ANSWER>: C",
            answer="C",
            reason="terminal attacker",
            confidence=None,
            token_count=2,
            embedding=np.asarray([0.0, 1.0], dtype=np.float32),
        )
        generate = AsyncMock()
        with patch.object(mas.agents[1], "generate", new=generate):
            scores = await mas._counterfactual_objectives(
                question="Which option? A. Alpha B. Beta C. Gamma D. Delta",
                round_id=1,
                attacker_id=0,
                candidates=[candidate],
                benign_ids=[1],
                benign_outputs=[benign],
                benign_prompts=["attacker-current", "benign-current"],
                target="C",
                sample_seed=11,
                attack=AttackConfig(attack_type="slow_drift", attacker_ids=[0]),
            )

        generate.assert_not_awaited()
        self.assertEqual(scores[0]["objective_horizon"], "terminal_no_benign_propagation")
        self.assertIsNone(scores[0]["rollout_round"])
        self.assertEqual(scores[0]["delivered_to"], [])
        self.assertEqual(scores[0]["baseline_answers"], ["B"])
        self.assertEqual(scores[0]["counterfactual_answers"], ["B"])
        self.assertEqual(scores[0]["flip_count"], 0)
        self.assertEqual(
            scores[0]["counterfactual_seed_schedule"],
            "counterfactual_common_random_numbers_v1",
        )
        self.assertEqual(scores[0]["rollout_seeds_by_benign_id"], {})

    async def test_counterfactual_candidates_share_common_random_numbers(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            adj=np.ones((3, 3), dtype=int) - np.eye(3, dtype=int),
            encoder=FakeEncoder(),
        )
        benign_outputs = []
        for benign_id in (1, 2):
            output = GeneratedMessage(
                text=f"<REASON>: benign-{benign_id}-current\n<ANSWER>: B",
                answer="B",
                reason=f"benign-{benign_id}-current",
                confidence=None,
                token_count=2,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            )
            benign_outputs.append(output)
            mas.agents[benign_id].commit(output, 1, "round-one prompt", role="benign")
        candidates = [
            GeneratedMessage(
                text=f"<REASON>: candidate-{index}\n<ANSWER>: C",
                answer="C",
                reason=f"candidate-{index}",
                confidence=None,
                token_count=2,
            )
            for index in range(2)
        ]
        rollout = GeneratedMessage(
            text="<REASON>: stable rollout\n<ANSWER>: B",
            answer="B",
            reason="stable rollout",
            confidence=None,
            token_count=2,
        )
        benign_one_generate = AsyncMock(return_value=rollout)
        benign_two_generate = AsyncMock(return_value=rollout)
        with patch.object(
            mas.agents[1], "generate", new=benign_one_generate
        ), patch.object(mas.agents[2], "generate", new=benign_two_generate):
            scores = await mas._counterfactual_objectives(
                question="Which option? A. Alpha B. Beta C. Gamma D. Delta",
                round_id=1,
                attacker_id=0,
                candidates=candidates,
                benign_ids=[1, 2],
                benign_outputs=benign_outputs,
                benign_prompts=["attacker-current", "benign-one-current", "benign-two-current"],
                target="C",
                sample_seed=19,
                attack=AttackConfig(attack_type="slow_drift", attacker_ids=[0]),
            )

        benign_one_seeds = [call.kwargs["seed"] for call in benign_one_generate.await_args_list]
        benign_two_seeds = [call.kwargs["seed"] for call in benign_two_generate.await_args_list]
        self.assertEqual(len(benign_one_seeds), 3)
        self.assertEqual(len(benign_two_seeds), 3)
        self.assertEqual(len(set(benign_one_seeds)), 1)
        self.assertEqual(len(set(benign_two_seeds)), 1)
        self.assertNotEqual(benign_one_seeds[0], benign_two_seeds[0])
        for score in scores:
            self.assertEqual(
                score["counterfactual_seed_schedule"],
                "counterfactual_common_random_numbers_v1",
            )
            self.assertEqual(
                score["rollout_seed_components"],
                ["sample_seed", "rollout_round", "focal_attacker_id", "benign_id"],
            )
            self.assertEqual(
                score["rollout_seeds_by_benign_id"],
                {"1": benign_one_seeds[0], "2": benign_two_seeds[0]},
            )

    async def test_multi_attacker_selection_scores_joint_next_round_context(self):
        adj = np.zeros((3, 3), dtype=int)
        adj[0, 2] = 1
        adj[1, 2] = 1
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            adj=adj,
            encoder=FakeEncoder(),
        )
        benign = GeneratedMessage(
            text="<REASON>: stable benign\n<ANSWER>: B",
            answer="B",
            reason="stable benign",
            confidence=None,
            token_count=2,
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        mas.agents[2].commit(benign, 1, "round-one prompt", role="benign")

        def candidates(prefix):
            return [
                GeneratedMessage(
                    text=f"<REASON>: {prefix}-weak\n<ANSWER>: C",
                    answer="C",
                    reason=f"{prefix}-weak",
                    confidence=None,
                    token_count=2,
                    embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                ),
                GeneratedMessage(
                    text=f"<REASON>: {prefix}-strong\n<ANSWER>: C",
                    answer="C",
                    reason=f"{prefix}-strong",
                    confidence=None,
                    token_count=2,
                    embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                ),
            ]

        async def rollout(prompt, **kwargs):
            joint_strong = "attacker-zero-strong" in prompt and "attacker-one-strong" in prompt
            answer = "C" if joint_strong else "B"
            return GeneratedMessage(
                text=f"<REASON>: rollout\n<ANSWER>: {answer}",
                answer=answer,
                reason="rollout",
                confidence=None,
                token_count=2,
            )

        generate = AsyncMock(side_effect=rollout)
        with patch.object(mas.agents[2], "generate", new=generate):
            selected, selections = await mas._select_joint_attack_candidates(
                question="Which option? A. Alpha B. Beta C. Gamma D. Delta",
                round_id=1,
                attacker_ids=[0, 1],
                candidate_lists=[candidates("attacker-zero"), candidates("attacker-one")],
                benign_ids=[2],
                benign_outputs=[benign],
                benign_support=np.asarray([[1.0, 0.0]], dtype=np.float32),
                previous_embeddings={0: None, 1: None},
                target="C",
                sample_seed=31,
                attack=AttackConfig(
                    attack_type="slow_drift",
                    attacker_ids=[0, 1],
                    tau=1.0,
                    drift_epsilon=1.0,
                    num_candidates=2,
                ),
                support_source="previous_round_benign_messages",
            )

        self.assertEqual(selected[0].reason, "attacker-zero-strong")
        self.assertEqual(selected[1].reason, "attacker-one-strong")
        self.assertEqual(generate.await_count, 5)
        self.assertEqual(
            len({call.kwargs["seed"] for call in generate.await_args_list}),
            1,
        )
        for attacker_id in (0, 1):
            metric = selections[attacker_id]["selected_metrics"]
            self.assertEqual(metric["counterfactual_scope"], "joint_attacker_combination")
            self.assertEqual(metric["candidate_indices_by_attacker"], {"0": 1, "1": 1})
            self.assertEqual(metric["baseline_answers"], ["B"])
            self.assertEqual(metric["counterfactual_answers"], ["C"])
            self.assertEqual(metric["flip_count"], 1)

    async def test_benign_wrapper_maximizes_flip_before_proxy_tiebreakers(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        benign = np.asarray([1.0, 0.0], dtype=np.float32)
        proxy_favored = benign.copy()
        objective_favored = np.asarray([0.6, 0.8], dtype=np.float32)
        candidates = [
            GeneratedMessage(
                text=(
                    "Conventional context matches the reference. || "
                    "A narrow reading supports C.\n<ANSWER>: C"
                ),
                answer="C",
                reason="Conventional context matches the reference. || A narrow reading supports C.",
                confidence=0.5,
                token_count=8,
                embedding=proxy_favored,
                wrapper="Conventional context matches the reference.",
                payload="A narrow reading supports C.\n<ANSWER>: C",
                wrapper_serialization_valid=True,
            ),
            GeneratedMessage(
                text=(
                    "Alternative conventional context remains plausible. || "
                    "Another interpretation supports C.\n<ANSWER>: C"
                ),
                answer="C",
                reason="Alternative conventional context remains plausible. || Another interpretation supports C.",
                confidence=0.5,
                token_count=8,
                embedding=objective_favored,
                wrapper="Alternative conventional context remains plausible.",
                payload="Another interpretation supports C.\n<ANSWER>: C",
                wrapper_serialization_valid=True,
            ),
        ]
        with patch.object(
            mas,
            "_encode",
            new=AsyncMock(
                return_value=np.stack([proxy_favored, objective_favored])
            ),
        ):
            _, selection = await mas._select_attack_candidate(
                candidates=candidates,
                attack=AttackConfig(
                    attack_type="benign_wrapper",
                    tau=1.0,
                    wrapper_tau=1.0,
                    payload_token_budget=16,
                ),
                target="C",
                benign_support=np.asarray([benign]),
                previous_embedding=None,
                objective_scores=[
                    {
                        "flip_count": 0,
                        "target_count": 0,
                        "benign_disagreement_rate": 0.0,
                    },
                    {
                        "flip_count": 2,
                        "target_count": 2,
                        "benign_disagreement_rate": 0.0,
                    },
                ],
                support_source="previous_round_benign_messages",
            )

        first, second = selection["candidates"]
        self.assertLess(
            first["final_cosine_distance_to_benign"],
            second["final_cosine_distance_to_benign"],
        )
        self.assertLess(
            first["wrapper_cosine_distance_to_benign"],
            second["wrapper_cosine_distance_to_benign"],
        )
        self.assertEqual(selection["selected_candidate_index"], 1)
        self.assertEqual(selection["selected_metrics"]["flip_count"], 2)
        self.assertEqual(
            selection["selection_reason"], "paper_objective_feasible_candidate"
        )

    async def test_benign_wrapper_generation_serializes_exact_a_concat_b(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        mas.agents[0].reset(
            get_system_prompt("benign_wrapper", "mmlu", "C", True)
        )
        candidates = await mas._generate_attack_candidates(
            agent_id=0,
            prompt="Build the paper wrapper candidate toward C.",
            attack=AttackConfig(
                attack_type="benign_wrapper",
                attacker_ids=[0],
                num_candidates=1,
            ),
            round_id=1,
            sample_seed=31,
            target="C",
        )

        candidate = candidates[0]
        self.assertTrue(candidate.wrapper_serialization_valid)
        self.assertTrue(candidate.wrapper)
        self.assertTrue(candidate.payload)
        self.assertEqual(
            candidate.text,
            f"{candidate.wrapper} || {candidate.payload}",
        )
        self.assertNotIn("<REASON>", candidate.text)
        self.assertTrue(candidate.payload.endswith("<ANSWER>: C"))
        self.assertEqual(candidate.text.count("<ANSWER>"), 1)
        self.assertEqual(candidate.answer, "C")

    async def test_benign_wrapper_retries_invalid_wire_format_without_posthoc_injection(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        mas.agents[0].reset(
            get_system_prompt("benign_wrapper", "mmlu", "C", True)
        )
        invalid = GeneratedMessage(
            text="The conventional analysis supports B.\n<ANSWER>: C",
            answer="C",
            reason="The conventional analysis supports B.",
            confidence=0.4,
            token_count=7,
        )
        repaired = GeneratedMessage(
            text=(
                "The conventional analysis supports B. || "
                "A narrow exception supports C.\n<ANSWER>: C"
            ),
            answer="C",
            reason="A narrow exception supports C.",
            confidence=0.6,
            token_count=14,
        )
        generate = AsyncMock(side_effect=[invalid, repaired])
        with patch.object(mas.agents[0], "generate", new=generate):
            candidate = (
                await mas._generate_attack_candidates(
                    agent_id=0,
                    prompt="Build the paper wrapper candidate toward C.",
                    attack=AttackConfig(
                        attack_type="benign_wrapper",
                        attacker_ids=[0],
                        num_candidates=1,
                    ),
                    round_id=1,
                    sample_seed=31,
                    target="C",
                )
            )[0]

        self.assertEqual(generate.await_count, 2)
        initial_call, repair_call = generate.await_args_list
        self.assertNotEqual(initial_call.kwargs["seed"], repair_call.kwargs["seed"])
        self.assertIn("FORMAT REPAIR", repair_call.args[0])
        self.assertTrue(candidate.wrapper_format_repair_attempted)
        self.assertTrue(candidate.wrapper_serialization_valid)
        self.assertEqual(candidate.text, repaired.text)
        self.assertEqual(candidate.confidence, repaired.confidence)
        self.assertEqual(candidate.token_count, repaired.token_count)
        self.assertEqual(candidate.answer, "C")

    async def test_benign_wrapper_preserves_exact_two_paragraph_generation(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        mas.agents[0].reset(
            get_system_prompt("benign_wrapper", "mmlu", "D", True)
        )
        text = (
            "A. The conventional medical analysis supports nephrolithiasis.\n\n"
            "| B. A narrow alternative reading supports D.\n<ANSWER>: D"
        )
        generated = GeneratedMessage(
            text=text,
            answer="D",
            reason="generated reason",
            confidence=0.37,
            token_count=19,
        )
        generate = AsyncMock(return_value=generated)

        with patch.object(mas.agents[0], "generate", new=generate):
            candidate = (
                await mas._generate_attack_candidates(
                    agent_id=0,
                    prompt="Build the paper wrapper candidate toward D.",
                    attack=AttackConfig(
                        attack_type="benign_wrapper",
                        attacker_ids=[0],
                        num_candidates=1,
                    ),
                    round_id=1,
                    sample_seed=31,
                    target="D",
                )
            )[0]

        self.assertEqual(generate.await_count, 1)
        self.assertTrue(candidate.wrapper_serialization_valid)
        self.assertEqual(candidate.text, text)
        self.assertEqual(
            candidate.wrapper,
            "A. The conventional medical analysis supports nephrolithiasis.",
        )
        self.assertEqual(
            candidate.payload,
            "| B. A narrow alternative reading supports D.\n<ANSWER>: D",
        )
        self.assertEqual(candidate.confidence, 0.37)
        self.assertEqual(candidate.token_count, 19)

    async def test_benign_wrapper_never_injects_target_into_legacy_untagged_text(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        generated = GeneratedMessage(
            text=(
                "The conventional calculation supports B. || "
                "A narrower interpretation still supports B."
            ),
            answer="B",
            reason="observed non-target answer",
            confidence=0.5,
            token_count=10,
        )
        with patch.object(mas.agents[0], "generate", new=AsyncMock(return_value=generated)):
            candidates = await mas._generate_attack_candidates(
                agent_id=0,
                prompt="Build the paper wrapper candidate toward C.",
                attack=AttackConfig(
                    attack_type="benign_wrapper",
                    attacker_ids=[0],
                    num_candidates=1,
                ),
                round_id=1,
                sample_seed=31,
                target="C",
            )

        candidate = candidates[0]
        self.assertFalse(candidate.wrapper_serialization_valid)
        self.assertEqual(candidate.answer, "B")

    async def test_benign_wrapper_preserves_visible_non_target_answer_for_hard_gate(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        generated = GeneratedMessage(
            text=(
                "The conventional calculation supports B. || "
                "A narrower interpretation still supports B.\n<ANSWER>: B"
            ),
            answer="B",
            reason="observed non-target answer",
            confidence=0.5,
            token_count=12,
        )
        with patch.object(mas.agents[0], "generate", new=AsyncMock(return_value=generated)):
            candidate = (
                await mas._generate_attack_candidates(
                    agent_id=0,
                    prompt="Build the paper wrapper candidate toward C.",
                    attack=AttackConfig(
                        attack_type="benign_wrapper",
                        attacker_ids=[0],
                        num_candidates=1,
                    ),
                    round_id=1,
                    sample_seed=31,
                    target="C",
                )
            )[0]

        self.assertTrue(candidate.wrapper_serialization_valid)
        self.assertEqual(candidate.answer, "B")
        candidate.embedding = FakeEncoder().encode([candidate.text])[0]
        with self.assertRaises(NoFeasibleAttackCandidate) as raised:
            await mas._select_attack_candidate(
                candidates=[candidate],
                attack=AttackConfig(
                    attack_type="benign_wrapper",
                    attacker_ids=[0],
                    tau=1.0,
                    wrapper_tau=1.0,
                    payload_token_budget=24,
                    num_candidates=1,
                ),
                target="C",
                benign_support=np.asarray([candidate.embedding]),
                previous_embedding=None,
                objective_scores=[{
                    "flip_count": 0,
                    "target_count": 0,
                    "benign_disagreement_rate": 0.0,
                }],
                support_source="previous_round_benign_messages",
            )

        metric = raised.exception.candidates[0]
        self.assertFalse(metric["target_answered"])
        self.assertIn("target_answer_mismatch", metric["rejection_reasons"])

    async def test_benign_wrapper_budget_uses_generation_tokenizer(self):
        class TokenCountingClient(FakeLLMClient):
            def __init__(self):
                super().__init__()
                self.tokenized = []

            async def acount_tokens(self, text, model=None):
                self.tokenized.append((text, model))
                return 3

        client = TokenCountingClient()
        mas = DebateMAS(
            client=client,
            model="fake-model",
            n_agents=2,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        benign_vector = FakeEncoder().encode(["Conventional analysis."])[0]
        candidate = GeneratedMessage(
            text="Conventional analysis. || pivot\n<ANSWER>: C",
            answer="C",
            reason="Conventional analysis. || pivot",
            confidence=0.5,
            token_count=4,
            embedding=benign_vector.copy(),
        )
        candidate.wrapper = "Conventional analysis."
        candidate.payload = "pivot\n<ANSWER>: C"
        candidate.wrapper_serialization_valid = True

        with self.assertRaises(NoFeasibleAttackCandidate) as raised:
            await mas._select_attack_candidate(
                candidates=[candidate],
                attack=AttackConfig(
                    attack_type="benign_wrapper",
                    tau=1.0,
                    wrapper_tau=1.0,
                    payload_token_budget=2,
                    num_candidates=1,
                ),
                target="C",
                benign_support=np.asarray([benign_vector]),
                previous_embedding=None,
                objective_scores=[{
                    "flip_count": 1,
                    "target_count": 1,
                    "benign_disagreement_rate": 0.0,
                }],
                support_source="previous_round_benign_messages",
            )

        self.assertEqual(client.tokenized, [("pivot\n<ANSWER>: C", "fake-model")])
        metric = raised.exception.candidates[0]
        self.assertEqual(metric["payload_token_count"], 3)
        self.assertNotIn("payload_token_count_approx", metric)
        self.assertIn("payload_token_budget_exceeded", metric["rejection_reasons"])

    async def test_chaos_maximizes_actual_disagreement_before_proxy_metrics(self):
        mas = DebateMAS(
            client=FakeLLMClient(),
            model="fake-model",
            n_agents=3,
            n_rounds=1,
            dataset="mmlu",
            encoder=FakeEncoder(),
        )
        benign = np.asarray([1.0, 0.0], dtype=np.float32)
        candidates = [
            GeneratedMessage(
                text="<REASON>: Conventional alternative supports C.\n<ANSWER>: C",
                answer="C",
                reason="Conventional alternative supports C.",
                confidence=0.5,
                token_count=4,
                embedding=benign.copy(),
            ),
            GeneratedMessage(
                text="<REASON>: Diverse alternatives support C.\n<ANSWER>: C",
                answer="C",
                reason="Diverse alternatives support C.",
                confidence=0.5,
                token_count=4,
                embedding=np.asarray([0.6, 0.8], dtype=np.float32),
            ),
        ]
        _, selection = await mas._select_attack_candidate(
            candidates=candidates,
            attack=AttackConfig(attack_type="chaos_seeding", tau=1.0),
            target="C",
            benign_support=np.asarray([benign]),
            previous_embedding=None,
            objective_scores=[
                {
                    "flip_count": 3,
                    "target_count": 3,
                    "benign_disagreement_rate": 0.0,
                },
                {
                    "flip_count": 0,
                    "target_count": 0,
                    "benign_disagreement_rate": 1.0,
                },
            ],
            support_source="previous_round_benign_messages",
        )

        first, second = selection["candidates"]
        self.assertLess(
            first["cosine_distance_to_benign"],
            second["cosine_distance_to_benign"],
        )
        self.assertGreater(first["flip_count"], second["flip_count"])
        self.assertEqual(selection["selected_candidate_index"], 1)
        self.assertEqual(
            selection["selected_metrics"]["benign_disagreement_rate"], 1.0
        )
        self.assertEqual(
            selection["selection_reason"], "paper_objective_feasible_candidate"
        )

    async def test_none_defense_does_not_request_logprobs(self):
        client = FakeLLMClient()
        mas = DebateMAS(
            client=client,
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            adj=build_adjacency("chain", 3),
            defense=DefenseConfig(mode="none"),
            encoder=FakeEncoder(),
        )
        result = await mas.run_one(
            question="Which option is correct? A. Alpha B. Beta C. Gamma D. Delta",
            gold="B",
            attack=AttackConfig(attack_type="none", attacker_ids=[]),
        )
        self.assertTrue(client.logprobs_requested)
        self.assertTrue(all(value is False for value in client.logprobs_requested))
        self.assertTrue(all(value is None for value in result["per_agent_conf"]))
        self.assertTrue(all(seed is not None for seed in client.seeds))

    async def test_confidence_weighting_adds_weight_metadata(self):
        client = FakeLLMClient()
        mas = DebateMAS(
            client=client,
            model="fake-model",
            n_agents=3,
            n_rounds=2,
            dataset="mmlu",
            adj=build_adjacency("chain", 3),
            defense=DefenseConfig(mode="confidence_weighting"),
            encoder=FakeEncoder(),
        )
        result = await mas.run_one(
            question="Which option is correct? A. Alpha B. Beta C. Gamma D. Delta",
            gold="B",
            attack=AttackConfig(attack_type="none", attacker_ids=[]),
        )
        round_two_prompts = [
            item["prompt"]
            for history in result["agent_histories"]
            for item in history
            if item["round"] == 2
        ]
        self.assertTrue(any("[confidence=" in prompt for prompt in round_two_prompts))
        self.assertTrue(any("weight=" in prompt for prompt in round_two_prompts))
        for history in result["agent_histories"]:
            self.assertIn("confidence_weight", history[-1])


if __name__ == "__main__":
    unittest.main()
