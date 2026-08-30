# Phase 2 — Ticket Triage

## Flow

```text
raw ticket
   ↓
TicketInput validation
   ↓
local KB retrieval
   ↓
top-3 KB context
   ↓
LLM structured JSON
   ↓
Pydantic validation
   ↓
post-generation guardrails
   ↓
TriageResult
```

## Why this design

- Retrieval is deterministic and local.
- The LLM does not have filesystem access.
- KB citations are restricted to retrieved sources.
- Category and urgency are based on ticket semantics, not copied from the
  synthetic historical labels.
- Responder teams are a controlled vocabulary.
- Temperature is zero for repeatability.
- The first response is grounded in retrieved KB context.

## Prompt versioning

`triage-v1.5` is included in the output so prompt changes can be tracked later.

The active prompt requires operation-level evidence matching before
`known_issue=true` is allowed. A generic product-level performance document is
not sufficient when the ticket identifies a different module or workflow.

The final response has a strict two-way evidence invariant:

- `known_issue=true` requires at least one retrieved, semantically supporting
  KB citation.
- `known_issue=false` requires `knowledge_base_matches=[]`.

Each citation must exactly identify a retrieved `(source_file, heading)` pair.
The application checks the cited chunk's text against the ticket; filenames,
headings, product overlap, and model-written relevance reasons are not enough
on their own.

The final `known_issue` value is derived after generation from the retrieved
chunk text using deterministic product, operation, and symptom checks. The LLM
can provide the classification and response wording, but cannot turn a
non-matching KB result into a known issue or suppress a matching documented
pattern.
