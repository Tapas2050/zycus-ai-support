# 🎯 Zycus AI Support Assistant

**AI-assisted support ticket triage and TAM account-health analysis** — a production-oriented take-home implementation for the *US Delivery Internship — Technical Interview Task Round*, built on FastAPI, Gradio, and structured-output LLM calls via OpenRouter.

## 🎯 Problem

The repo implements a local internal tool for two concrete tasks:

1. **Ticket triage** from a support subject/body (category, urgency, product area, responder team routing, suggested first response, and cited KB articles).
2. **TAM-style account health analysis** for a known account ID (executive summary, account-level risk signals, verbatim-cited ticket risks, talking points for next calls, and ticket join strategy tracking).

The crucial design constraint is **strict validation**: the app rejects invalid categories, invalid responder teams, ungrounded KB citations, wrong account IDs, and ticket-risk evidence that is not an exact substring of the source ticket body. In other words, the app is designed to keep the model honest by checking the generated response against dataset facts, KB retrieval results, and schema contracts before accepting or caching it.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **UI & UX** | Gradio 6.x (Blocks, background streaming generator workers, queue polling, custom CSS keyframe animations) |
| **LLM / Structured Output** | OpenAI SDK → OpenRouter (model selected through `LLM_MODEL`), JSON-schema-constrained generation, bounded retries, explicit request timeout, content-addressed on-disk caching |
| **Retrieval** | scikit-learn (`TfidfVectorizer` + cosine similarity) over local Markdown knowledge base |
| **Testing** | pytest, `pytest.ini`, unittest.mock |
| **CI/CD** | GitHub Actions (`.github/workflows/evals.yml`) with automated evaluation gates |

---

## 🔄 Actual Code Flow

This is the execution path in the repo:

```text
data/accounts.json + data/tickets.json
        │
        ├── DataRepository
        │      - loads accounts and tickets
        │      - resolves account lookup by account_id
        │      - applies company-based fallback for ticket/account matching
        │      - exposes get_tickets_for_account(account_id, days=90)
        │
        ├── KBRetriever
        │      - loads knowledge-base markdown files
        │      - vectorizes with TF-IDF
        │      - retrieves top chunks for a ticket query
        │      - formats retrieved context for LLM use
        │
        ├── TriageAgent
        │      - calls KBRetriever.retrieve(...)
        │      - builds user prompt with only retrieved KB sources
        │      - calls LLMClient.generate_json(..., response_model=TriageResult)
        │      - enforces category / urgency / responder-team constraints
        │      - validates KB source_file + heading against retrieved chunks
        │      - defends against false-positive generic KB citations
        │      - caches result only after all guardrails pass
        │
        ├── HealthAgent
        │      - resolves account via DataRepository.get_account(account_id)
        │      - calls repo.get_tickets_for_account(account_id, days=90)
        │      - builds account + recent-ticket prompt
        │      - calls LLMClient.generate_json(..., response_model=AccountHealthResult)
        │      - enforces account_id match and verbatim evidence-substring checks
        │      - caches result only after all guardrails pass
        │
        ├── FastAPI app in src/app/api.py
        │      - /health
        │      - /triage
        │      - /accounts
        │      - /accounts/{account_id}/health
        │      - /accounts/{account_id}/tickets
        │
        ├── Gradio UI in ui.py (Real-time Streaming Engine)
        │      - builds a TestClient around the FastAPI app
        │      - executes agent calls asynchronously in daemon worker threads
        │      - streams live status cards, elapsed time, and retry/patience notices
        │      - cycles non-repeating fun facts & industry insights every 15s
        │      - renders validated markdown + formatted raw JSON results
        │
        └── Evaluation harness
               - tests/ unit tests
               - src/app/evaluation/runner.py
               - writes eval_report.json
```

---

## ✨ Features Implemented in This Repo

- 🧠 **KB-grounded triage** — `TriageAgent` retrieves chunks from the markdown knowledge base before invoking the LLM, and it validates that every cited KB match came from the retrieved set. The code rejects unsupported citations rather than letting the model invent a source.
- 🔄 **Real-time UI streaming & wait engagement** — `ui.py` runs LLM pipelines in background threads while streaming live status cards to the Gradio interface:
  - Immediate initial acknowledgement so users never experience a frozen UI.
  - Multi-phase wait blurbs informing users during retries, provider latency, or deep reasoning calls (e.g. *"Grab a coffee — the AI is being extra thorough…"*, *"The model may be retrying a slow provider call…"*, *"Running guardrails and KB citation checks…"*, etc.).
  - 15-second heartbeat intervals that display elapsed time alongside non-repeating fun facts about AI, support engineering, and triage history.
  - Custom pulsing and shimmering CSS animations for the status panel during active processing.
- 🔗 **Data-quality-aware ticket/account matching** — `DataRepository` resolves account/ticket relations using the real dataset, and the repo’s health flow records the selected strategy (`account_id` vs fallback behavior) through the response contract rather than silently hiding it.
- 📎 **Evidence-verification for ticket risks** — `HealthAgent` requires every `evidence_quote` to be an exact substring of the source ticket body. This is enforced in code before the response is accepted.
- 💾 **Guardrail-gated on-disk caching** — `LLMClient` creates a SHA-256 content-addressed hash from model + prompt + schema and stores output under `.cache/llm/`. Caching is intentionally deferred until *after* all semantic guardrails pass, preventing bad or unvalidated responses from being permanently saved.
- 🛡️ **Bounded provider retry** — `LLMClient.generate_json()` makes at most three attempts, retries transient provider/network failures, uses a configurable request timeout, and fails credit/quota errors immediately. Retries keep the same strict JSON schema.
- 🔒 **Structured-output schema normalization** — `_normalize_json_schema()` explicitly adjusts nested array-item requirements to resolve Pydantic default-factory behaviors and ensure array fields (like `knowledge_base_matches` and `ticket_risks`) are treated as required by strict LLMs.
- 🚫 **Module/operation guardrails** — `TriageAgent` checks the relationship between a retrieved KB chunk and the ticket’s operating context so a generic KB hit cannot be treated as a valid answer for a different product area or operation.
- 🖥️ **UI uses the real API path** — `ui.py` instantiates `TestClient(fastapi_app)` from `src/app/api.py`; the UI is not calling repository classes directly. Each Gradio action hits the FastAPI endpoints with validation/error handling intact.
- 🧪 **Code-driven evaluation suite** — Includes unit tests and 11 evaluation cases (6 triage, 5 account health), including adversarial scenarios, writing `eval_report.json`.

---

## 📈 Evaluation Results

From `eval_report.json`:

| Metric | Value |
|---|---|
| Total cases | 11 (6 triage, 5 account-health) |
| Passed | Depends on the configured provider and current run |
| Overall quality score | Reported by the current `eval_report.json` |
| Adversarial coverage | ✅ (e.g. `healthy_account_with_customer_declared_p1` — account metadata says "Healthy" while a ticket body declares a P1) |
| Test suite status | Run `pytest` to verify the current checkout |

---

## 📁 Actual Repo Structure

```text
zycus-ai-support/
├── .env                          # local environment file; not committed
├── .cache/                       # LLM cache output directory
├── .github/workflows/
│   └── evals.yml                # CI workflow for tests + evaluation
├── data/
│   ├── accounts.json             # account dataset used by DataRepository
│   └── tickets.json              # ticket dataset used by DataRepository
├── docs/                         # architecture/design/evaluation docs
│   ├── ARCHITECTURE.md
│   ├── DATA_PROFILE.md
│   ├── DESIGN.md
│   ├── EVALUATION.md
│   ├── PHASE_2_TRIAGE.md
│   └── PROMPT_CHANGELOG.md
├── knowledge-base/
│   ├── billing/
│   ├── onboarding/
│   ├── products/
│   └── troubleshooting/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── __main__.py          # quick dataset/KB introspection utility
│       ├── api.py               # FastAPI app and route handlers
│       ├── config.py            # .env-loaded settings
│       ├── data_repository.py   # dataset loader + account/ticket lookups
│       ├── health_agent.py      # account health summarization
│       ├── kb_retriever.py      # TF-IDF retrieval over Markdown KB
│       ├── llm_client.py        # OpenRouter client, retries, caching, schema normalization
│       ├── models.py            # Pydantic models used by API + agents
│       ├── prompts.py           # current system prompts and prompt versions
│       ├── triage_agent.py      # triage classification + guardrails
│       └── evaluation/
│           ├── cases.py         # evaluation fixtures/cases
│           ├── runner.py        # executes evaluation cases
│           └── scorer.py        # scoring logic
├── run.py                        # uvicorn entrypoint for the API server
├── ui.py                         # Gradio UI with real-time progress streaming
├── health_check.py               # standalone simple health check
├── test_evaluation.py            # repo-level evaluation runner check script
├── test_openrouter.py            # OpenRouter connectivity check script
├── test_triage.py                # triage-specific smoke check script
├── pytest.ini                    # pytest configuration (pythonpath & test discovery)
├── tests/
│   ├── test_agents.py
│   ├── test_data_repository.py
│   └── test_kb_retriever.py
├── eval_report.json              # latest evaluation result file
├── requirements.txt              # Python dependencies (UTF-8 encoded)
├── README.md                     # project overview
└── .gitignore                    # repo ignore rules
```

---

## ⚙️ Setup

Requirements: Python 3.11+.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Create a local `.env` file in the project root:
```bash
# Windows
copy NUL .env

# Linux/macOS
touch .env
```

Set in `.env`:
```env
LLM_API_KEY=your-openrouter-api-key
LLM_MODEL=your-openrouter-model-id
LLM_FALLBACK_MODEL=optional-fallback-model-id
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_CACHE_DIR=.cache/llm
LLM_TIMEOUT_SECONDS=60
# Optional API protection
API_AUTH_TOKEN=replace-with-a-secret-token
API_RATE_LIMIT_PER_MINUTE=60
```

`API_AUTH_TOKEN` enables `X-API-Key` authentication for API routes other than
`/health`. The in-memory rate limit is per client host and is intended for a
single-process deployment; use an external gateway for distributed rate limiting.

---

## ▶️ How the App Runs

The repository contains two primary entrypoints:

### 1) Gradio UI (Recommended for interactive use)
```bash
python ui.py
```
- Opens locally at `http://127.0.0.1:7860`.
- Features real-time status streaming, retry progress notices, elapsed timer counters, and cycling domain facts while AI reasoning executes.
- Exercises the full FastAPI route stack in-process via `TestClient(fastapi_app)` to ensure UI and API never drift.

### 2) FastAPI Server (Direct API mode)
```bash
python run.py
```
- Launches Uvicorn on `http://localhost:8000`.
- Interactive Swagger docs available at `http://localhost:8000/docs`.

#### Ticket Triage Endpoint
`POST /triage`
```json
{
  "subject": "Dashboard loading times unacceptable — AnalyticsHub",
  "body": "Our Data Sources dashboard in AnalyticsHub is now taking over 100 seconds to load."
}
```

#### TAM Account Health Endpoint
`GET /accounts/ACC-6254/health`

Returns: executive summary, account-level risk signals, evidence-backed ticket risks, TAM talking points, and ticket join strategy used.

#### UI Helper Endpoints
- `GET /accounts` — `{account_id, company}` list for the UI account dropdown.
- `GET /accounts/{account_id}/tickets?days=90` — recent ticket history with join strategy metadata.

---

## 🧪 Testing & Validation Commands

### Unit Tests
Run all unit tests:
```bash
pytest -v
```

### Full AI Evaluation Suite
Run the Phase 3 evaluation harness and generate `eval_report.json`:
```bash
python -m app.evaluation.runner
```
Or via the root smoke script:
```bash
python test_evaluation.py
```

### Standalone Smoke Checks
```bash
python test_openrouter.py
python test_triage.py
python health_check.py
```

---

## 📚 Concepts & Architecture Summary

- **Retrieval-Augmented Generation (RAG)**: TF-IDF chunk indexing with strict citation grounding (rejects unretrieved sources).
- **Structured Outputs & Schema Normalization**: Pydantic v2 schemas combined with recursive JSON-schema normalization to enforce required fields across complex nested objects.
- **Asynchronous UI Streaming**: Daemon worker threading with queue polling to stream live status and feedback in Gradio without sacrificing full-schema response validation.
- **Content-Addressed Caching**: SHA-256 hash-keyed caching executed post-guardrail validation.
- **Entity Resolution**: Exact `account_id` joins with company-name fallback tracking.
- **Verbatim Evidence Verification**: Exact substring matching on all cited ticket quotes.
- **Resilient Retry Policies**: Bounded backoff for transient provider failures, with optional final-attempt fallback model support.
- **CI/CD Quality Gates**: Automated GitHub Actions workflow testing unit suites and end-to-end evaluation metrics on push.

---

## 👨‍💻 Author

**Tapas** — Software Engineer focused on AI/ML, Generative AI, and Backend Development. [github.com/Tapas2050](https://github.com/Tapas2050)

## ⭐

If you found this project useful, consider giving it a star!
