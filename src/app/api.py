import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.data_repository import DataRepository
from app.config import settings
from app.models import TicketInput

app = FastAPI(
    title="Zycus AI Support Assistant",
    version="0.1.0",
    description="AI-assisted ticket triage and TAM account health tooling.",
)

_triage_agent = None
_health_agent = None
_repository = None
_rate_limit_state: dict[str, list[float]] = {}


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if request.url.path != "/health":
        configured_token = settings.api_auth_token
        if configured_token and request.headers.get("x-api-key") != configured_token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

        limit = settings.api_rate_limit_per_minute
        if limit > 0:
            client_host = request.client.host if request.client else "unknown"
            now = time.monotonic()
            recent = [
                timestamp
                for timestamp in _rate_limit_state.get(client_host, [])
                if now - timestamp < 60
            ]
            if len(recent) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                )
            recent.append(now)
            _rate_limit_state[client_host] = recent

    return await call_next(request)


def get_triage_agent():
    global _triage_agent
    if _triage_agent is None:
        from app.triage_agent import TriageAgent
        _triage_agent = TriageAgent()
    return _triage_agent


def get_health_agent():
    global _health_agent
    if _health_agent is None:
        from app.health_agent import HealthAgent
        _health_agent = HealthAgent()
    return _health_agent


def get_repository():
    global _repository
    if _repository is None:
        _repository = DataRepository()
    return _repository


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage")
def triage(ticket: TicketInput):
    try:
        return get_triage_agent().triage(ticket)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/accounts/{account_id}/health")
def account_health(account_id: str):
    try:
        return get_health_agent().summarise(account_id)
    except ValueError as exc:
        if "Unknown account_id" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/accounts")
def list_accounts() -> list[dict[str, str]]:
    """Minimal account list for UI lookups (dropdown population)."""
    repo = get_repository()
    return [
        {"account_id": a.account_id, "company": a.company}
        for a in sorted(repo.accounts, key=lambda a: a.account_id)
    ]


@app.get("/accounts/{account_id}/tickets")
def account_tickets(account_id: str, days: int = 90):
    """Recent ticket history for an account, with the join strategy used."""
    repo = get_repository()
    if repo.get_account(account_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown account_id: {account_id}"
        )
    tickets, join_strategy = repo.get_tickets_for_account(account_id, days=days)
    return {"tickets": tickets, "join_strategy": join_strategy}