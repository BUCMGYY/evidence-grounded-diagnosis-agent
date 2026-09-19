"""Synthetic contract/branch tests only; no study cases or private resources."""
from dataclasses import dataclass, replace
import unittest

from evidence_agent import (
    Clause, ContractError, Decision, E2Result, KnowledgeEntry, MemoryItem,
    Neighbour, OperationalFailure, Proposal, efficient_trigger, parent_context,
    retrieve_five, route_evidence, run_agent,
)


PATTERNS = ("toy_a", "toy_b", "toy_c", "toy_d", "toy_e")
DISEASES = ("toy_disease_a", "toy_disease_b")


def neighbours(labels=("toy_a", "toy_a", "toy_a", "toy_b", "toy_b")):
    return tuple(Neighbour(i + 1, 0.9 - i * 0.1, label,
                           (("chief_complaint", "toy memory text"),
                            ("physical_examination", "toy examination text")))
                 for i, label in enumerate(labels))


def parent(uncertainty="high"):
    return E2Result(DISEASES, DISEASES, PATTERNS, PATTERNS, uncertainty)


def knowledge(patterns):
    return tuple(KnowledgeEntry(p, ("knowledge_" + p,), ("toy support only",))
                 for p in patterns)


def changed():
    return Proposal(("toy_b", "toy_a", "toy_c", "toy_d", "toy_e"), "low",
                    ("C1",), ("knowledge_toy_b",))


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.evidence = route_evidence((
            Clause("C1", "chief_complaint", "toy shared evidence"),
            Clause("A1", "auxiliary_examination", "toy disease-only evidence"),
            Clause("T1", "tongue_pulse", "toy pattern-only evidence"),
        ))

    def run_case(self, *, e2=None, propose=None, adjudicate=None,
                 memory=None, knowledge_for=knowledge):
        return run_agent(self.evidence, memory or neighbours(), DISEASES, PATTERNS,
                         knowledge_for, e2 or (lambda x: parent()),
                         propose or (lambda x: changed()),
                         adjudicate or (lambda x: Decision("accept_k_proposal", ("C1",),
                                                          ("knowledge_toy_b",))))

    def test_routing(self):
        self.assertEqual(self.evidence.disease_ids, ("C1", "A1"))
        self.assertEqual(self.evidence.pattern_ids, ("C1", "T1"))

    def test_duplicate_clause_rejected(self):
        with self.assertRaises(ContractError):
            route_evidence((Clause("x", "chief_complaint", "toy"),) * 2)

    def test_unsupported_field_rejected(self):
        with self.assertRaises(ContractError):
            route_evidence((Clause("x", "reference_label", "toy"),))

    def test_label_attachment_happens_only_after_selection(self):
        items = [MemoryItem(str(i), (1.0, 0.0)) for i in reversed(range(6))]
        attached = []
        def attach(identifier):
            attached.append(identifier)
            return "toy_a", {"chief_complaint": "x" * 100,
                             "physical_examination": "y" * 100}
        selected = retrieve_five("query", (2.0, 0.0), items, attach)
        self.assertEqual(attached, ["0", "1", "2", "3", "4"])
        self.assertEqual(len(selected), 5)
        self.assertFalse(hasattr(selected[0], "opaque_id"))
        self.assertEqual(len(dict(selected[0].snippets)["chief_complaint"]), 80)
        self.assertEqual(len(parent_context(selected)), 3)
        self.assertNotIn("physical_examination", dict(parent_context(selected)[0].snippets))

    def test_label_changes_cannot_change_similarity_selection(self):
        items = [MemoryItem(str(i), (1.0, float(i))) for i in range(6)]
        first, second = [], []
        retrieve_five("q", (1.0, 0.0), items,
                      lambda key: (first.append(key) or "toy_a", {}))
        retrieve_five("q", (1.0, 0.0), items,
                      lambda key: (second.append(key) or "toy_e", {}))
        self.assertEqual(first, second)

    def test_self_neighbour_rejected(self):
        with self.assertRaises(ContractError):
            retrieve_five("0", (1, 0), [MemoryItem(str(i), (1, 0)) for i in range(5)],
                          lambda x: ("toy_a", {}))

    def test_bad_embedding_rejected(self):
        for vector in ((0, 0), (float("nan"), 0), (float("inf"), 0)):
            with self.assertRaises(ContractError):
                retrieve_five("q", vector, [MemoryItem(str(i), (1, 0)) for i in range(5)],
                              lambda x: ("toy_a", {}))

    def test_moderate_alone_does_not_trigger(self):
        self.assertFalse(efficient_trigger(parent("moderate"), neighbours()).active)

    def test_high_uncertainty_triggers(self):
        self.assertTrue(efficient_trigger(parent("high"), neighbours()).high_uncertainty)

    def test_no_majority_triggers(self):
        self.assertTrue(efficient_trigger(parent("low"), neighbours(PATTERNS)).no_majority)

    def test_majority_compares_final_not_initial_primary(self):
        p = replace(parent("low"), final_pattern=("toy_b", "toy_a", "toy_c", "toy_d", "toy_e"),
                    pattern_evidence=("C1",))
        self.assertTrue(efficient_trigger(p, neighbours()).majority_disagrees)

    def test_no_trigger_calls_only_e2(self):
        def forbidden(*args):
            self.fail("Unneeded callback invoked")
        out = self.run_case(e2=lambda x: parent("moderate"), propose=forbidden,
                            adjudicate=forbidden, knowledge_for=forbidden)
        self.assertEqual((out.status, out.pattern, out.called_stages), ("ok", "toy_a", ("E2",)))

    def test_unchanged_k_skips_v(self):
        out = self.run_case(propose=lambda x: Proposal(PATTERNS, "low"),
                            adjudicate=lambda x: self.fail("V should be skipped"))
        self.assertEqual((out.pattern, out.called_stages), ("toy_a", ("E2", "K")))

    def test_accepted_proposal_keeps_disease(self):
        out = self.run_case()
        self.assertEqual((out.status, out.disease, out.pattern, out.called_stages),
                         ("ok", "toy_disease_a", "toy_b", ("E2", "K", "V")))

    def test_rejected_proposal_keeps_e2(self):
        out = self.run_case(adjudicate=lambda x: Decision("retain_e2", ("T1",)))
        self.assertEqual((out.status, out.pattern), ("ok", "toy_a"))

    def test_k_v_inputs_exclude_disease(self):
        def e2(request):
            self.assertEqual(len(request.analogues), 3)
            self.assertNotIn("physical_examination", dict(request.analogues[0].snippets))
            return parent()
        def k(request):
            self.assertEqual(len(request.analogues), 5)
            self.assertEqual({c.identifier for c in request.clauses}, {"C1", "T1"})
            self.assertFalse(hasattr(request.anchor, "disease"))
            self.assertEqual({x.pattern for x in request.knowledge}, set(PATTERNS[:3]))
            return changed()
        def v(request):
            self.assertEqual({c.identifier for c in request.clauses}, {"C1", "T1"})
            self.assertEqual([x.pattern for x in request.cited_knowledge], ["toy_b"])
            return Decision("accept_k_proposal", ("C1",), ("knowledge_toy_b",))
        self.assertEqual(self.run_case(e2=e2, propose=k, adjudicate=v).status, "ok")

    def test_k_uses_initial_top_three_not_final_top_three(self):
        p = replace(parent(), final_pattern=("toy_c", "toy_d", "toy_e", "toy_a", "toy_b"),
                    pattern_evidence=("C1",))
        bad = Proposal(("toy_d", "toy_a", "toy_b", "toy_c", "toy_e"), "low",
                       ("C1",), ("knowledge_toy_a",))
        out = self.run_case(e2=lambda x: p, propose=lambda x: bad)
        self.assertEqual((out.status, out.failure_stage, out.disease, out.pattern),
                         ("invalid", "K", "toy_disease_a", None))

    def test_missing_k_evidence_invalid_no_fallback(self):
        out = self.run_case(propose=lambda x: replace(changed(), evidence=()))
        self.assertEqual((out.status, out.pattern, out.called_stages),
                         ("invalid", None, ("E2", "K")))

    def test_missing_k_knowledge_invalid(self):
        self.assertEqual(self.run_case(propose=lambda x: replace(changed(), knowledge_ids=())).status,
                         "invalid")

    def test_disease_only_evidence_invalid_for_k(self):
        self.assertEqual(self.run_case(propose=lambda x: replace(changed(), evidence=("A1",))).status,
                         "invalid")

    def test_v_cannot_verify_knowledge_k_did_not_cite(self):
        out = self.run_case(adjudicate=lambda x: Decision("accept_k_proposal", ("C1",),
                                                         ("knowledge_toy_a",)))
        self.assertEqual((out.status, out.failure_stage, out.pattern), ("invalid", "V", None))

    def test_v_cannot_offer_third_choice(self):
        self.assertEqual(self.run_case(adjudicate=lambda x: Decision("toy_c", ("C1",))).status,
                         "invalid")

    def test_v_requires_evidence_even_for_retain(self):
        self.assertEqual(self.run_case(adjudicate=lambda x: Decision("retain_e2", ())).status,
                         "invalid")

    def test_acceptance_requires_verified_knowledge(self):
        self.assertEqual(self.run_case(adjudicate=lambda x: Decision("accept_k_proposal", ("C1",))).status,
                         "invalid")

    def test_e2_cannot_change_disease_ranking(self):
        out = self.run_case(e2=lambda x: replace(parent(), final_disease=DISEASES[::-1]))
        self.assertEqual((out.status, out.disease, out.pattern, out.failure_stage),
                         ("invalid", None, None, "E2"))

    def test_operational_failure_distinct_from_invalid_output(self):
        def fail(request):
            raise OperationalFailure("toy transport failure")
        out = self.run_case(propose=fail)
        self.assertEqual((out.status, out.pattern, out.failure_stage), ("operational_error", None, "K"))

    def test_programming_error_is_not_silently_scored(self):
        def fail(request):
            raise KeyError("toy coding bug")
        with self.assertRaises(KeyError):
            self.run_case(propose=fail)

    def test_proposal_subclass_cannot_send_disease_to_v(self):
        @dataclass(frozen=True)
        class ExtraProposal(Proposal):
            disease: str = "toy_disallowed_disease"
        p = changed()
        bad = ExtraProposal(p.ranking, p.uncertainty, p.evidence, p.knowledge_ids)
        out = self.run_case(propose=lambda x: bad,
                            adjudicate=lambda x: self.fail("Extra fields reached V"))
        self.assertEqual((out.status, out.failure_stage), ("invalid", "K"))

    def test_e2_subclass_is_rejected(self):
        @dataclass(frozen=True)
        class ExtraParent(E2Result):
            private_field: str = "toy private field"
        bad = ExtraParent(DISEASES, DISEASES, PATTERNS, PATTERNS, "high")
        self.assertEqual(self.run_case(e2=lambda x: bad).status, "invalid")

    def test_clause_subclass_is_rejected(self):
        @dataclass(frozen=True)
        class ExtraClause(Clause):
            private_field: str = "toy private field"
        with self.assertRaises(ContractError):
            route_evidence((ExtraClause("C1", "chief_complaint", "toy"),))

    def test_neighbour_subclass_is_rejected(self):
        @dataclass(frozen=True)
        class ExtraNeighbour(Neighbour):
            private_field: str = "toy private field"
        memory = list(neighbours())
        n = memory[0]
        memory[0] = ExtraNeighbour(n.rank, n.similarity, n.pattern, n.snippets)
        with self.assertRaises(ContractError):
            self.run_case(memory=memory)

    def test_manual_neighbour_rejects_nonallowlisted_fields(self):
        for field in ("reference_label", "auxiliary_examination", "case_record_id"):
            memory = list(neighbours())
            memory[0] = replace(memory[0], snippets=((field, "toy forbidden field"),))
            with self.assertRaises(ContractError):
                self.run_case(memory=memory)

    def test_manual_neighbour_mutable_pairs_are_deep_frozen(self):
        pair = ["chief_complaint", "toy memory"]
        memory = list(neighbours())
        memory[0] = replace(memory[0], snippets=(pair,))
        def e2(request):
            self.assertIs(type(request.analogues[0].snippets[0]), tuple)
            with self.assertRaises(TypeError):
                request.analogues[0].snippets[0][0] = "auxiliary_examination"
            pair[0] = "reference_label"  # Caller object must not alias snapshots.
            return parent()
        def k(request):
            self.assertEqual(request.analogues[0].snippets,
                             (("chief_complaint", "toy memory"),))
            return changed()
        self.assertEqual(self.run_case(memory=memory, e2=e2, propose=k).status, "ok")

    def test_manual_neighbour_is_compacted(self):
        memory = list(neighbours())
        memory[0] = replace(memory[0], snippets=(("present_illness", "x" * 150),))
        def e2(request):
            self.assertEqual(len(request.analogues[0].snippets[0][1]), 120)
            return parent("low")
        self.assertEqual(self.run_case(memory=memory, e2=e2).status, "ok")

    def test_uncertainty_string_subclass_is_rejected(self):
        class ExtraString(str):
            disease = "toy disallowed field"
        out = self.run_case(e2=lambda x: replace(parent(), uncertainty=ExtraString("high")))
        self.assertEqual(out.status, "invalid")

    def test_knowledge_nested_objects_are_rejected(self):
        def bad_knowledge(patterns):
            rows = list(knowledge(patterns))
            rows[0] = replace(rows[0], supporting_features=(["toy nested object"],))
            return rows
        out = self.run_case(knowledge_for=bad_knowledge)
        self.assertEqual((out.status, out.failure_stage), ("invalid", "resources"))

    def test_knowledge_list_inputs_are_frozen(self):
        features = ["toy original feature"]
        def external(patterns):
            return [KnowledgeEntry(p, ["knowledge_" + p], features,
                                   "toy interpretation", ["toy limitation"]) for p in patterns]
        def k(request):
            features[0] = "toy changed caller feature"
            for entry in request.knowledge:
                self.assertIs(type(entry.identifiers), tuple)
                self.assertEqual(entry.supporting_features, ("toy original feature",))
                self.assertIs(type(entry.limitations), tuple)
            return changed()
        self.assertEqual(self.run_case(knowledge_for=external, propose=k).status, "ok")


if __name__ == "__main__":
    unittest.main()
