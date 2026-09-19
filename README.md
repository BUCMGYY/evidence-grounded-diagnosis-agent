# Evidence grounded diagnosis agent

A dependency-free **publication reference implementation** of an evidence-routed
parent (E2), conditional knowledge proposal (K), and independent binary
adjudication (V) workflow. It is not the archived production runner, a model,
a clinical knowledge base, or a ready-to-deploy diagnostic service.

## Included and excluded

Included: deterministic evidence routes, label-blind cosine retrieval followed
by pattern attachment, the efficient trigger, candidate/citation validation,
and conditional E2-to-K-to-V control flow. Tests use invented labels and toy
text only.

Excluded: patient data, reference labels, partitions, embeddings, model weights,
private or licensed knowledge, production prompts, provider adapters, account
information, credentials, experiment outputs and scientific result tables.
No network request, environment-variable lookup, file access, or logging occurs
in the implementation. Tests can run without installation or internet access.

## Run the toy tests

Python 3.10 or later, standard library only:

```text
python -m unittest -v test_evidence_agent.py
```

## Caller supplied interfaces

1. Supply authorized, de-identified and diagnosis-safe clauses to
   `route_evidence`. This function does not remove personal information or
   diagnose unsafe free text.
2. Supply vectors from a fixed external encoder and a permissible memory pool
   to `retrieve_five`. The attachment callback provides pattern labels and
   snippets only after similarity-based selection. No query reference label
   belongs in these interfaces.
3. Supply an external `knowledge_for` callback returning the three eligible
   pattern entries. Do not publish those private resources with this code.
4. Supply separate `e2`, `propose`, and `adjudicate` callbacks returning the
   declared result dataclasses. Their prompts, structured decoding, model
   access, consent, secure transport and privacy review are the caller's
   responsibility. The K and V inputs omit the disease prediction and
   disease-only clauses. Do not give callbacks hidden access to prohibited data.

The returned status is `ok`, `invalid` for a structural contract violation, or
`operational_error` for a caller-raised `OperationalFailure`. Other programming
exceptions propagate: a software defect must not silently become a scientific
prediction. A failed required K or V does not silently fall back to E2.
Any carried disease value in a non-`ok` result is diagnostic state, not a valid
prediction eligible for scoring or clinical use.

This core executes only the experimental pathway, so it makes one to three
callbacks. The study's paired Direct-prompting comparison, qualification
overrides, retry/resume system and statistical analysis are not reproduced.
See `METHODS.md` for preserved rules and intentional differences.

## Rights and intended use

No software license grant is provided with this draft release. All rights are
reserved pending the authors' rights review and explicit licensing decision.
Public visibility does not itself grant reuse rights. Do not infer endorsement
or clinical validation. This research reference is not for clinical decisions.

