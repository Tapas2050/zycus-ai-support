# Prompt Changelog

## v1.5 / v1.2 — Current prompts

- `triage-v1.5`: content-grounded ticket classification, controlled responder teams,
  operation-specific known-issue checks, and concise first-response drafting. The
  known-issue decision flow was de-duplicated and the version changed to isolate
  cached output from the previous prompt contract.
- `health-v1.2`: 3–5 sentence summary, separation of account-level signals from
  ticket-level risks, and exact evidence-quote requirement.

The current prompt constants are defined in `src/app/prompts.py`. The application
also applies code-level schema and evidence guardrails after generation.

Future prompt changes should increment the version and record:
- what changed,
- why it changed,
- evaluation impact.
