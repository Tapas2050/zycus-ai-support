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

`triage-v1.2` is included in the output so prompt changes can be tracked later.
