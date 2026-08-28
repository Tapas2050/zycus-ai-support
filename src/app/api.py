from fastapi import FastAPI, HTTPException

from app.models import TicketInput

app = FastAPI(
    title="Zycus AI Support Assistant",
    version="0.1.0",
    description="AI-assisted ticket triage and TAM account health tooling.",
)

_triage_agent = None
_health_agent = None


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
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/accounts")
def list_accounts() -> list[dict[str, str]]:
    """Minimal account list for UI lookups (dropdown population)."""
    repo = get_health_agent().repo
    return [
        {"account_id": a.account_id, "company": a.company}
        for a in sorted(repo.accounts, key=lambda a: a.account_id)
    ]


@app.get("/accounts/{account_id}/tickets")
def account_tickets(account_id: str, days: int = 90):
    """Recent ticket history for an account, with the join strategy used."""
    repo = get_health_agent().repo
    if repo.get_account(account_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown account_id: {account_id}"
        )
    tickets, join_strategy = repo.get_tickets_for_account(account_id, days=days)
    return {"tickets": tickets, "join_strategy": join_strategy}