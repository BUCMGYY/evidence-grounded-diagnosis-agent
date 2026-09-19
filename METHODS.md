# Reference scope and implementation provenance

This is a new, compact reference implementation, not a verbatim export of the
archived study code. It has not rerun model inference or reproduced the study's
performance estimates. No inference-label migration is implied by its generic
label interfaces. The original production schema fixed six disease labels;
later five-category evaluation was a separate scoring operation. This reference
does not claim that a native five-category inference experiment was performed.

## Preserved algorithmic rules

- Current-case chief complaint, present illness and physical examination can
  support both tasks. Ancillary examination is disease-only; tongue/pulse is
  pattern-only. Caller-provided clause identifiers are unique within a case.
- Select five analogues using only embedding cosine similarity. Resolve exact
  similarity ties by lexical memory identifier order. Attach pattern labels
  and snippets only after selection; do not expose identifiers to model inputs.
- E2 receives three analogues. Its compact analogue context includes complaint,
  present illness and tongue/pulse snippets, capped at 80/120/80 characters.
  K/V use five analogues and additionally include examination (80 characters).
- E2 returns complete initial/final rankings. With only routing and retrieval
  active, its entire disease ranking is unchanged. A changed primary pattern
  must be in its initial top three and cite current-case pattern evidence.
- Trigger K if E2 uncertainty is `high`, OR no pattern has at least three of
  five neighbour labels, OR that majority differs from E2's final primary.
  `moderate` uncertainty alone does not trigger the selected efficient policy.
- K receives only pattern evidence and knowledge restricted to the E2 initial
  top-three patterns. Its output is a complete five-pattern ranking. A changed
  primary must cite both current-case pattern evidence and eligible knowledge.
- Call V only if K changes E2's primary pattern. V receives the K proposal and
  only K-cited knowledge. It can only retain E2 or accept K; no third proposal.
  Every V decision requires current-case pattern evidence; acceptance additionally
  verifies at least one knowledge identifier actually cited by K.
- The final disease remains E2's disease. No trigger, unchanged K, or V rejection
  retains E2's pattern. A required stage that fails its contract makes the
  pathway invalid, rather than silently returning a successful E2 result.

These are structural checks; they cannot establish that a citation clinically
supports a prediction, that a supplied label is correct, or that an LLM truly
reasoned independently. The caller must provide a separate V invocation without
hidden conversational state or prohibited disease/reference information.

## Intentionally external or different

The implementation does not include the archived preprocessing marker lists,
clinical attention-cue lists, demographic context, exact prompt text, fixed
clinical taxonomy, BGE encoder/model cache, statistical analysis, study-specific
cohort gates, provider policy, checkpoint metadata or private knowledge.
Clause preparation and study-faithful prompt construction remain external.
The core accepts precomputed embeddings and normalizes them for cosine scoring;
it does not claim bitwise parity with float32 BGE retrieval. It requires unique
knowledge identifiers and validates supplied neighbour ordering as additional
interface safeguards, not new study results.

The archived implementation used a frozen BGE-M3 encoder, top-five retrieval and
top-three parent context. Reproducing that encoder, prompts, original resources
and archived environment is outside this package. Toy tests demonstrate branch
and contract behavior only, not clinical efficacy or production equivalence.

## Code-level sources reviewed

Paths below identify the archived implementation modules reviewed during the
reference audit; those files and their resource dependencies are not bundled.

| Preserved rule | Archived module/function |
|---|---|
| Field routes and parent context | `complete_cases_v4p_extension_v8_efficient/evidence_router.py`: `build_evidence_map`, `compact_analogues` |
| Label-blind selection | `complete_cases_v4p_extension_v7/retrieval.py`: `FrozenBGERetriever.neighbors` |
| Parent contract | `complete_cases_v4p_extension_v8_efficient/runner.py`: `_validate_result` |
| Efficient trigger | `complete_cases_v4p_extension_v9_adaptive/trigger.py`: `compute_pattern_reflection_trigger` |
| Candidate-restricted proposal | `complete_cases_v4p_extension_v9_adaptive/prompts.py`; `runner.py`: `_validate_child` |
| Binary V contract | `complete_cases_v4p_extension_v10_sequential/schemas.py`; `runner.py`: `_validate_v` |
| Final conditional pathway | `complete_cases_v4p_final_sequential_transfer/runner.py`: `run_case` |

Directly copying the historical runners would also import prior experiment
runners, schema/provider packages, a local-secret loader and fixed private
resource paths. The historical final constructor loads evidence, reference
patterns, partitions and knowledge even when provider access is disabled. This
reference instead has no such dependency or initialization side effect.
