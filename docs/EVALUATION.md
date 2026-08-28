# Evaluation Harness

The evaluation harness currently contains 5 cases: 3 ticket-triage cases and 2 account-health cases, including adversarial coverage.

## Task 1

The evaluator scores:
- category
- urgency
- product area
- KB retrieval
- responder team
- response presence
- rationale presence

Category and urgency are critical gates because they are the core triage output.

## Task 2

The evaluator scores:
- account identity
- health/usage synthesis
- 3–5 sentence executive summary
- data-join strategy
- evidence-backed ticket risks
- expected operational risk signal where applicable
- TAM talking points

## Why acceptance criteria instead of exact LLM strings?

LLM outputs are naturally variable. Exact string matching would reward formatting
rather than quality. The harness therefore uses deterministic checks for facts and
structure while leaving wording flexible.

The dataset's historical ticket `category` and `urgency` fields are not treated as
the only source of truth for Task 1 because the supplied synthetic records contain
semantic conflicts between those fields and the ticket text.
