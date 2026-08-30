# Architecture — Phase 1 Foundation

```text
                         supplied synthetic data
                    ┌────────────┬───────────────┐
                    │            │               │
              tickets.json  accounts.json   knowledge-base/
                    │            │               │
                    └──────┬─────┴───────┬───────┘
                           │             │
                    DataRepository   KBRetriever
                           │             │
                           └──────┬──────┘
                                  │
                         Task services (Phase 2+)
                         ┌────────┴────────┐
                         │                 │
                    TriageAgent      HealthAgent
                         │                 │
                         └────────┬────────┘
                                  │
                           Evaluation Harness
```

## Design decisions

1. **No database:** the assignment supplies small JSON datasets; an in-process repository is enough.
2. **No external data:** all retrieval is limited to the supplied corpus.
3. **Deterministic KB retrieval:** TF-IDF is reproducible and auditable.
4. **RAG is separate from LLM orchestration:** the model receives explicit retrieved context rather than searching the filesystem itself.
5. **Data reconciliation is explicit:** the account-ID/company fallback is observable rather than silently altering source data.
6. **Dataset as-of is explicit:** historical synthetic data is evaluated relative to its own latest timestamp unless overridden.

## UI ↔ API integration

`ui.py` (Gradio) does not import and call `TriageAgent` / `HealthAgent` directly.
Instead it drives the real FastAPI app in-process with Starlette's `TestClient`
(`client = TestClient(fastapi_app)`), so every UI action is a real HTTP call
through `app.api`'s routing, Pydantic request validation, and
`HTTPException` handling — the same code path a deployed `uvicorn` process
uses. This closes the gap where the UI could silently drift from the API
contract (e.g. calling repository methods the API doesn't expose).

To support this, `app.api` exposes two additional read endpoints used only
by the UI:

- `GET /accounts` — minimal `{account_id, company}` list for the account
  dropdown.
- `GET /accounts/{account_id}/tickets?days=90` — recent ticket history for
  an account plus the `join_strategy` used (`account_id` vs
  `company_fallback`), reusing `DataRepository.get_tickets_for_account`.

## Knowledge-base retrieval

The supplied Markdown corpus is small, so retrieval stays local and deterministic.
Documents are split at Markdown heading boundaries (while preserving the original
source file and heading as metadata), indexed with TF-IDF, and ranked with
deterministic error-code, product, heading, and document-family signals.
Operational queries restrict candidates to troubleshooting chunks when that corpus
is available, and a minimum score prevents weak matches from entering the prompt.

This is deliberately simpler than introducing a hosted vector database for a
four-hour take-home.