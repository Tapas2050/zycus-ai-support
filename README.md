# 🎯 Zycus AI Support Assistant

**AI-assisted support ticket triage and TAM account-health analysis** — a production-oriented take-home implementation for the *US Delivery Internship — Technical Interview Task Round*, built on FastAPI, Gradio, and structured-output LLM calls via OpenRouter.

## 🎯 Problem

Support and TAM teams need fast, consistent classification of incoming tickets and a trustworthy read on account health — without an LLM hallucinating a KB citation, inventing a churn signal, or joining the wrong customer's ticket history.

**Goal:** turn a raw ticket into a validated triage decision, and a raw account into an evidence-backed health summary, with every claim traceable back to real data.

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Backend API | FastAPI, Uvicorn, Pydantic v2 |
| UI | Gradio (Blocks, custom theme/CSS) |
| LLM / Structured Output | OpenAI SDK → OpenRouter, JSON-schema-constrained generation, content-addressed on-disk caching |
| Retrieval | scikit-learn (`TfidfVectorizer` + cosine similarity) over local Markdown knowledge base |
| Testing | pytest |
| CI/CD | GitHub Actions (`.github/workflows/evals.yml`) |

## 🔄 Project Workflow

```text
                     Synthetic Dataset
             ┌───────────┬──────────────┐
             │           │              │
          Tickets     Accounts        KB (Markdown)
             │           │              │
             └──────┬────┴──────┬───────┘
                    │           │
              DataRepository  KBRetriever
              (ID→company      (TF-IDF +
               join fallback)   cosine sim)
                    │           │
          ┌─────────┴───────────┴─────────┐
          │                               │
      TriageAgent                    HealthAgent
   (category/urgency/            (risk signals +
    routing + KB match)           quoted evidence)
          │                               │
          └──────────────┬────────────────┘
                         │
                 Pydantic validation
                         │
              FastAPI app (/triage, /accounts,
              /accounts/{id}/health, /accounts/{id}/tickets)
                         │
        Gradio UI (ui.py) — same process, calls the
        FastAPI app via TestClient (in-memory, no
        network hop, no separate uvicorn needed)
                         │
                 Evaluation Harness (CI-gated)
```

## ✨ Features / Highlights

- 🧠 **Retrieval-grounded triage** — the LLM only sees KB chunks actually retrieved by TF-IDF/cosine similarity, and KB citations that weren't retrieved are rejected, so it can't invent a source.
- 🔗 **Data-quality-aware account join** — `DataRepository` joins tickets to accounts by `account_id` first, validates against ticket `company`, and falls back to a company-name join when the ID is inconsistent — the strategy used (`account_id` vs `company_fallback`) is returned in the output, not hidden.
- 📎 **Verbatim evidence enforcement** — every ticket-level risk flag must include an exact substring quote from the source ticket body; non-verbatim evidence is rejected before it reaches the response.
- 💾 **Content-addressed LLM caching** — results are cached by a hash of model + system prompt + user prompt + response schema, so identical evaluation cases return identical results without re-calling the LLM.
- 🛡️ **Reasoning-budget-safe structured output** — the LLM client caps the reasoning-token budget separately from the output budget and retries once with a larger total budget on truncation, so reasoning-heavy models don't silently return an empty response.
- 🖥️ **Gradio UI with real account lookup** — the Account Health tab uses a searchable dropdown built from the actual dataset (not free-text ID guessing), and both tabs render results as formatted Markdown with the raw JSON available in a collapsible panel.
- 🔌 **UI drives the real API, in-process** — `ui.py` never imports and calls the agents directly. It wraps the FastAPI `app` in a `TestClient` and every UI action is a real HTTP call through routing/validation/error-handling, with no separate server process required.
- 🧪 **CI-gated evaluation** — GitHub Actions runs unit tests and the evaluation harness on every push and fails the build if any case regresses.

## 📈 Evaluation Results

From the current `eval_report.json`:

| Metric | Value |
|---|---|
| Total cases | 5 (3 triage, 2 account-health) |
| Passed | 5 / 5 |
| Overall quality score | 1.0 |
| Adversarial coverage | ✅ (e.g. `healthy_account_with_customer_declared_p1` — account metadata says "Healthy" while a ticket body declares a P1) |

## 📁 Project Structure

```text
zycus-ai-support/
├── run.py                        # FastAPI entry point
├── ui.py                         # Gradio UI entry point
├── health_check.py               # standalone health-check script
├── src/app/
│   ├── api.py                    # FastAPI routes (/triage, /accounts, /accounts/{id}/health, /accounts/{id}/tickets)
│   ├── config.py                 # env-driven settings (.env via python-dotenv)
│   ├── models.py                 # Pydantic models: Ticket, Account, TicketInput, KBChunk
│   ├── data_repository.py        # ticket/account loading + ID→company join fallback
│   ├── kb_retriever.py           # TF-IDF + cosine-similarity KB retrieval
│   ├── llm_client.py             # OpenRouter client, JSON-schema output, caching, retry
│   ├── triage_agent.py           # ticket → category/urgency/routing/KB matches
│   ├── health_agent.py           # account → risk signals + evidence-quoted ticket risks
│   ├── prompts.py                # versioned system prompts (TRIAGE/HEALTH_PROMPT_VERSION)
│   └── evaluation/
│       ├── cases.py               # evaluation case definitions (incl. adversarial cases)
│       ├── runner.py              # runs cases → eval_report.json
│       └── scorer.py              # scoring logic
├── data/                         # accounts.json, tickets.json (synthetic)
├── knowledge-base/               # Markdown KB: billing, onboarding, products, troubleshooting
├── tests/                        # pytest unit tests
├── docs/                         # ARCHITECTURE, DESIGN, EVALUATION, DATA_PROFILE, PROMPT_CHANGELOG
└── .github/workflows/evals.yml   # CI: pytest + evaluation harness, gated on pass rate
```

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

Create environment configuration:
```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Set in `.env`:
- `LLM_API_KEY`
- `LLM_MODEL`
- optional `LLM_BASE_URL`

Never commit `.env`.

## ▶️ Run

**Gradio UI (recommended):**
```bash
python ui.py
```
This is the normal way to run the app. `ui.py` builds a `TestClient(app)`
against the FastAPI app **in-process** — Gradio calls the same routing,
Pydantic validation, and `HTTPException` handling as a deployed API, with no
network hop and no separate server to start. You do **not** need to run
`run.py`/`uvicorn` alongside it.

**API standalone (optional):** only needed if something *other than this UI*
(curl, Postman, another service) needs to call the HTTP API directly.
```bash
python run.py
```
Available at `http://localhost:8000` (interactive docs at `/docs`).

### Ticket triage
`POST /triage`
```json
{
  "subject": "Dashboard loading times unacceptable — AnalyticsHub",
  "body": "Our Data Sources dashboard in AnalyticsHub is now taking over 100 seconds to load."
}
```

### TAM account health
`GET /accounts/ACC-6254/health`

Returns: executive summary, account-level risk signals, evidence-backed ticket risks, TAM talking points, and the ticket join strategy used.

### UI support endpoints
- `GET /accounts` — `{account_id, company}` list, used by the UI's account dropdown.
- `GET /accounts/{account_id}/tickets?days=90` — recent ticket history + `join_strategy` used, shown in the UI's "Recent Tickets" panel.

## 🧪 Evaluation & Tests

```bash
PYTHONPATH=src python -m app.evaluation.runner
```
Writes `eval_report.json`. Currently 5 cases (3 triage, 2 account-health), including adversarial coverage.

```bash
pytest -q
```

See [`docs/EVALUATION.md`](docs/EVALUATION.md).

## 📚 Concepts Used

- Retrieval-augmented generation with citation grounding (reject unretrieved sources)
- JSON-schema-constrained LLM output + Pydantic validation
- Content-addressed caching (hash of model + prompts + schema)
- Data-quality-aware entity resolution (ID join with company-name fallback)
- Evidence-verification for LLM-generated risk claims (verbatim substring checks)
- Reasoning-token budget management for reasoning-capable LLMs
- CI-gated model evaluation (regression-gated deployment)

## 🔧 Known Gaps / Future Improvements

- No authentication on the Gradio UI or FastAPI endpoints by default — `ui.py` has a commented-out `auth=` stub for Gradio's basic auth, off unless explicitly enabled via `UI_USER`/`UI_PASSWORD` env vars.
- No streaming in the UI — `LLMClient.generate_json()` uses JSON-schema-constrained generation validated as a whole response, which isn't compatible with token-by-token streaming without breaking schema validation.
- No `LICENSE` file in the repo — add one before treating this as reusable/open-source.
- Evaluation harness currently covers 5 cases; broader coverage (more triage categories, more account-health edge cases) would increase confidence before any production use.
- `.env` and `.venv/` are correctly `.gitignore`'d, but nothing prevents them from being included if the project folder is zipped/shared manually instead of pushed via git — rotate any key that's been shared this way.

## 👨‍💻 Author

**Tapas** — Software engineer focused on AI/ML, Generative AI, and backend/full-stack development. [github.com/Tapas2050](https://github.com/Tapas2050)

## ⭐

If you found this project useful, consider giving it a star!