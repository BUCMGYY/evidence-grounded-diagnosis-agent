"""Dependency-free publication reference, not the archived study runner.

Only caller-supplied, authorized, de-identified resources may be used. This
module performs no network requests, file access, persistence, or logging.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Mapping, Sequence


class ContractError(ValueError):
    """A supplied object violates the declared reference contract."""


class OperationalFailure(RuntimeError):
    """Caller reports a transport/configuration failure, not a prediction."""


@dataclass(frozen=True, slots=True)
class Clause:
    identifier: str
    field: str
    text: str


@dataclass(frozen=True, slots=True)
class RoutedEvidence:
    clauses: tuple[Clause, ...]
    disease_ids: tuple[str, ...]
    pattern_ids: tuple[str, ...]


def route_evidence(clauses: Sequence[Clause]) -> RoutedEvidence:
    """Route already sanitized clauses; this is NOT a de-identification tool."""
    shared = {"chief_complaint", "present_illness", "physical_examination"}
    fields = shared | {"auxiliary_examination", "tongue_pulse"}
    rows = tuple(clauses)
    if any(type(c) is not Clause for c in rows):
        raise ContractError("Evidence requires exact Clause objects, without extra fields")
    if any(type(value) is not str for c in rows for value in (c.identifier, c.field, c.text)):
        raise ContractError("Clause identifiers, fields and text must be plain strings")
    ids = [c.identifier for c in rows]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ContractError("Clause identifiers must be unique and nonempty")
    if any(c.field not in fields or not c.text.strip() for c in rows):
        raise ContractError("Unsupported evidence field or empty clause")
    return RoutedEvidence(
        rows,
        tuple(c.identifier for c in rows if c.field in shared | {"auxiliary_examination"}),
        tuple(c.identifier for c in rows if c.field in shared | {"tongue_pulse"}),
    )


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """No label is accepted at similarity-selection time."""
    opaque_id: str
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Neighbour:
    """Model-visible analogue: no memory identifier or disease label."""
    rank: int
    similarity: float
    pattern: str
    snippets: tuple[tuple[str, str], ...]


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in vector)
    if not values or not all(isfinite(x) for x in values):
        raise ContractError("Embedding must be finite and nonempty")
    norm = sqrt(sum(x * x for x in values))
    if not isfinite(norm) or norm == 0:
        raise ContractError("Embedding must have finite positive norm")
    return tuple(x / norm for x in values)


def retrieve_five(
    query_id: str,
    query_embedding: Sequence[float],
    memory: Sequence[MemoryItem],
    attach: Callable[[str], tuple[str, Mapping[str, str]]],
) -> tuple[Neighbour, ...]:
    """Select by cosine similarity first, then request labels and snippets.

    The caller supplies only the permitted memory partition. This function
    cannot verify patient-level independence or how an encoder was trained.
    """
    query = _unit(query_embedding)
    items = tuple(memory)
    ids = [x.opaque_id for x in items]
    if len(items) < 5 or len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ContractError("At least five unique memory items are required")
    if query_id in ids:
        raise ContractError("The query cannot occur in its retrieval memory")
    scored = []
    for item in sorted(items, key=lambda x: x.opaque_id):
        vector = _unit(item.embedding)
        if len(vector) != len(query):
            raise ContractError("Embedding dimensions must agree")
        scored.append((sum(a * b for a, b in zip(query, vector)), item.opaque_id))
    # Stable sort retains lexical ID order on exact similarity ties.
    chosen = sorted(scored, key=lambda pair: -pair[0])[:5]
    limits = (("chief_complaint", 80), ("present_illness", 120),
              ("physical_examination", 80), ("tongue_pulse", 80))
    result = []
    for rank, (score, identifier) in enumerate(chosen, start=1):
        pattern, fields = attach(identifier)
        if not pattern:
            raise ContractError("Selected memory item lacks a pattern label")
        snippets = tuple((field, str(fields[field]).strip()[:limit])
                         for field, limit in limits
                         if fields.get(field) and str(fields[field]).strip())
        result.append(Neighbour(rank, round(score, 8), pattern, snippets))
    return tuple(result)


def _freeze_neighbours(neighbours: Sequence[Neighbour]) -> tuple[Neighbour, ...]:
    """Reject extra fields and create immutable, allowlisted snippet snapshots."""
    limits = {"chief_complaint": 80, "present_illness": 120,
              "physical_examination": 80, "tongue_pulse": 80}
    result = []
    for n in neighbours:
        if type(n) is not Neighbour:
            raise ContractError("Analogues require exact Neighbour objects")
        if type(n.pattern) is not str or type(n.rank) is not int:
            raise ContractError("Analogue label and rank have invalid types")
        if type(n.similarity) not in (int, float) or not isfinite(n.similarity):
            raise ContractError("Analogue similarity must be finite")
        if type(n.snippets) not in (list, tuple):
            raise ContractError("Analogue snippets must be a finite list or tuple")
        snippets = []
        seen = set()
        for pair in n.snippets:
            if type(pair) not in (list, tuple) or len(pair) != 2:
                raise ContractError("Malformed analogue snippet")
            field, text = pair
            if type(field) is not str or type(text) is not str or field not in limits:
                raise ContractError("Analogue snippet field is not permitted")
            if field in seen:
                raise ContractError("Duplicate analogue snippet field")
            seen.add(field)
            if text.strip():
                snippets.append((field, text.strip()[:limits[field]]))
        result.append(Neighbour(n.rank, float(n.similarity), n.pattern, tuple(snippets)))
    return tuple(result)


def parent_context(neighbours: Sequence[Neighbour]) -> tuple[Neighbour, ...]:
    """Parent E2 receives top three, without analogue examination snippets."""
    return tuple(Neighbour(n.rank, n.similarity, n.pattern,
                           tuple(s for s in n.snippets if s[0] != "physical_examination"))
                 for n in _freeze_neighbours(neighbours[:3]))


@dataclass(frozen=True, slots=True)
class E2Result:
    initial_disease: tuple[str, ...]
    final_disease: tuple[str, ...]
    initial_pattern: tuple[str, ...]
    final_pattern: tuple[str, ...]
    uncertainty: str
    disease_evidence: tuple[str, ...] = ()
    pattern_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PatternAnchor:
    """K and V receive no disease prediction or disease-only evidence."""
    initial_ranking: tuple[str, ...]
    final_ranking: tuple[str, ...]
    uncertainty: str


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    pattern: str
    identifiers: tuple[str, ...]
    supporting_features: tuple[str, ...] = ()
    interpretation: str = ""
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Proposal:
    ranking: tuple[str, ...]
    uncertainty: str
    evidence: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    choice: str
    evidence: tuple[str, ...]
    verified_knowledge: tuple[str, ...] = ()
    uncertainty: str = "low"


@dataclass(frozen=True, slots=True)
class E2Input:
    evidence: RoutedEvidence
    analogues: tuple[Neighbour, ...]
    disease_labels: tuple[str, ...]
    pattern_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KInput:
    clauses: tuple[Clause, ...]
    anchor: PatternAnchor
    analogues: tuple[Neighbour, ...]
    knowledge: tuple[KnowledgeEntry, ...]


@dataclass(frozen=True, slots=True)
class VInput:
    clauses: tuple[Clause, ...]
    anchor: PatternAnchor
    proposal: Proposal
    analogues: tuple[Neighbour, ...]
    cited_knowledge: tuple[KnowledgeEntry, ...]


@dataclass(frozen=True, slots=True)
class Trigger:
    active: bool
    high_uncertainty: bool
    no_majority: bool
    majority_disagrees: bool
    majority_pattern: str | None


def efficient_trigger(parent: E2Result, neighbours: Sequence[Neighbour]) -> Trigger:
    if len(neighbours) < 5:
        raise ContractError("The trigger requires five neighbours")
    if type(parent.uncertainty) is not str or parent.uncertainty not in {"low", "moderate", "high"}:
        raise ContractError("Unknown uncertainty category")
    counts = Counter(n.pattern for n in neighbours[:5])
    label, count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    majority = label if count >= 3 else None
    high = parent.uncertainty == "high"
    no_majority = majority is None
    disagreement = majority is not None and majority != parent.final_pattern[0]
    return Trigger(high or no_majority or disagreement, high, no_majority,
                   disagreement, majority)


def _ranking(values: tuple[str, ...], labels: tuple[str, ...]) -> None:
    if type(values) is not tuple or any(type(x) is not str for x in values):
        raise ContractError("Rankings require plain strings in an immutable tuple")
    if len(values) != len(labels) or set(values) != set(labels):
        raise ContractError("Ranking must contain each declared label exactly once")


def _citations(values: tuple[str, ...], allowed: set[str], *, minimum: int = 0,
               maximum: int = 12) -> None:
    if type(values) is not tuple or any(type(x) is not str for x in values):
        raise ContractError("Citations require plain strings in an immutable tuple")
    if not minimum <= len(values) <= maximum:
        raise ContractError("Citation cardinality is outside the contract")
    if not set(values).issubset(allowed):
        raise ContractError("Citation is unavailable or outside its evidence route")


def validate_e2(value: E2Result, request: E2Input) -> None:
    if type(value) is not E2Result:
        raise ContractError("E2 callback must return E2Result")
    for ranking in (value.initial_disease, value.final_disease):
        _ranking(ranking, request.disease_labels)
    for ranking in (value.initial_pattern, value.final_pattern):
        _ranking(ranking, request.pattern_labels)
    if value.final_disease != value.initial_disease:
        raise ContractError("E2 cannot revise its disease ranking")
    _citations(value.disease_evidence, set(request.evidence.disease_ids))
    _citations(value.pattern_evidence, set(request.evidence.pattern_ids))
    if type(value.uncertainty) is not str or value.uncertainty not in {"low", "moderate", "high"}:
        raise ContractError("Unknown uncertainty category")
    if value.final_pattern[0] != value.initial_pattern[0]:
        if value.final_pattern[0] not in value.initial_pattern[:3] or not value.pattern_evidence:
            raise ContractError("E2 revision requires top-three membership and current evidence")


def validate_proposal(value: Proposal, request: KInput, labels: tuple[str, ...]) -> None:
    if type(value) is not Proposal:
        raise ContractError("K callback must return Proposal")
    _ranking(value.ranking, labels)
    if value.ranking[0] not in request.anchor.initial_ranking[:3]:
        raise ContractError("K primary escaped the frozen E2 initial top three")
    changed = value.ranking[0] != request.anchor.final_ranking[0]
    _citations(value.evidence, {c.identifier for c in request.clauses}, minimum=int(changed))
    eligible = {identifier for k in request.knowledge for identifier in k.identifiers}
    _citations(value.knowledge_ids, eligible, minimum=int(changed), maximum=6)
    if type(value.uncertainty) is not str or value.uncertainty not in {"low", "moderate", "high"}:
        raise ContractError("Unknown uncertainty category")


def validate_decision(value: Decision, request: VInput) -> None:
    if type(value) is not Decision or type(value.choice) is not str or value.choice not in {"retain_e2", "accept_k_proposal"}:
        raise ContractError("V must make one of the two declared decisions")
    _citations(value.evidence, {c.identifier for c in request.clauses}, minimum=1)
    _citations(value.verified_knowledge, set(request.proposal.knowledge_ids),
               minimum=int(value.choice == "accept_k_proposal"), maximum=6)
    if type(value.uncertainty) is not str or value.uncertainty not in {"low", "moderate", "high"}:
        raise ContractError("Unknown uncertainty category")


@dataclass(frozen=True, slots=True)
class Result:
    status: str
    disease: str | None
    pattern: str | None
    called_stages: tuple[str, ...]
    trigger: Trigger | None = None
    failure_stage: str | None = None


def run_agent(
    evidence: RoutedEvidence,
    neighbours: Sequence[Neighbour],
    disease_labels: Sequence[str],
    pattern_labels: Sequence[str],
    knowledge_for: Callable[[tuple[str, ...]], Sequence[KnowledgeEntry]],
    e2: Callable[[E2Input], E2Result],
    propose: Callable[[KInput], Proposal],
    adjudicate: Callable[[VInput], Decision],
) -> Result:
    """Run 1--3 calls; caller owns resources, prompts and authorized LLM access.

    Structural checks cannot prove clinical support, evidence truth, privacy,
    or model independence. The caller must implement those requirements.
    Operational failures remain distinct from invalid model outputs.
    """
    diseases, patterns = tuple(disease_labels), tuple(pattern_labels)
    if any(type(x) is not str or not x for x in diseases + patterns):
        raise ContractError("Label universes require nonempty plain strings")
    if not diseases or len(patterns) != 5 or len(set(diseases)) != len(diseases) or len(set(patterns)) != 5:
        raise ContractError("Declare a unique disease universe and exactly five patterns")
    # Reconstruct routes so callers cannot broaden them by constructing a dataclass.
    evidence = route_evidence(evidence.clauses)
    top5 = _freeze_neighbours(neighbours[:5])
    if len(top5) != 5 or any(n.pattern not in patterns for n in top5):
        raise ContractError("Five neighbours from the declared pattern universe are required")
    if tuple(n.rank for n in top5) != (1, 2, 3, 4, 5):
        raise ContractError("Neighbours must be supplied in their selection order")
    request = E2Input(evidence, parent_context(top5), diseases, patterns)
    called: list[str] = []
    trigger = None
    stage = "E2"
    disease = None
    try:
        called.append(stage)
        parent = e2(request)
        validate_e2(parent, request)
        disease = parent.final_disease[0]
        trigger = efficient_trigger(parent, top5)
        if not trigger.active:
            return Result("ok", disease, parent.final_pattern[0], tuple(called), trigger)
        stage = "resources"
        eligible = parent.initial_pattern[:3]
        knowledge = tuple(knowledge_for(eligible))
        if any(type(k) is not KnowledgeEntry for k in knowledge):
            raise ContractError("Knowledge entries cannot include subclass fields")
        if any(type(values) not in (tuple, list) for k in knowledge
               for values in (k.identifiers, k.supporting_features, k.limitations)):
            raise ContractError("Knowledge collections must be finite lists or tuples")
        if any(type(x) is not str for k in knowledge
               for x in (k.pattern, k.interpretation, *k.identifiers,
                         *k.supporting_features, *k.limitations)):
            raise ContractError("Knowledge content requires plain strings")
        if {k.pattern for k in knowledge} != set(eligible) or len(knowledge) != 3:
            raise ContractError("Knowledge interface must return exactly the eligible patterns")
        knowledge = tuple(KnowledgeEntry(k.pattern, tuple(k.identifiers),
                                        tuple(k.supporting_features[:5]), k.interpretation,
                                        tuple(k.limitations)) for k in knowledge)
        all_ids = [identifier for k in knowledge for identifier in k.identifiers]
        if len(all_ids) != len(set(all_ids)) or any(not x for x in all_ids):
            raise ContractError("Knowledge identifiers must be unique and nonempty")
        clauses = tuple(c for c in evidence.clauses if c.identifier in evidence.pattern_ids)
        anchor = PatternAnchor(parent.initial_pattern, parent.final_pattern, parent.uncertainty)
        k_input = KInput(clauses, anchor, top5, knowledge)
        stage = "K"
        called.append(stage)
        proposal = propose(k_input)
        validate_proposal(proposal, k_input, patterns)
        if proposal.ranking[0] == parent.final_pattern[0]:
            return Result("ok", disease, parent.final_pattern[0], tuple(called), trigger)
        cited = set(proposal.knowledge_ids)
        selected = tuple(KnowledgeEntry(k.pattern, tuple(x for x in k.identifiers if x in cited),
                                        k.supporting_features, k.interpretation, k.limitations)
                         for k in knowledge if cited.intersection(k.identifiers))
        v_input = VInput(clauses, anchor, proposal, top5, selected)
        stage = "V"
        called.append(stage)
        decision = adjudicate(v_input)
        validate_decision(decision, v_input)
        pattern = proposal.ranking[0] if decision.choice == "accept_k_proposal" else parent.final_pattern[0]
        return Result("ok", disease, pattern, tuple(called), trigger)
    except OperationalFailure:
        return Result("operational_error", disease, None, tuple(called), trigger, stage)
    except ContractError:
        return Result("invalid", disease, None, tuple(called), trigger, stage)
