# Evaluation Harness

The evaluation harness currently contains 11 cases: 6 ticket-triage cases and 5 account-health cases, including adversarial coverage.

## Task 1

The evaluator scores category, urgency, known-issue status, KB attribution,
rationale presence, and response presence. For a positive known-issue result,
the expected KB source must be cited and the cited `(source_file, heading)`
must resolve to a real KB chunk whose text supports the ticket. Model-written
headings and relevance reasons are never treated as KB evidence.

The first response must also contain at least one case-specific operational
guidance term. A merely non-empty acknowledgement is not treated as useful.

The triage fixture supplies the exact subject/body used for both inference and
scoring. This keeps the report reproducible even when a fixture intentionally
differs from an identically named synthetic dataset ticket.

Category, urgency, known-issue status, and KB evidence are critical gates.

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
