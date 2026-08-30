import re
from typing import Any
from pydantic import BaseModel, Field, model_validator

from app.data_repository import DataRepository
from app.llm_client import LLMClient
from app.prompts import HEALTH_PROMPT_VERSION, HEALTH_SYSTEM_PROMPT


class RiskFlag(BaseModel):
    ticket_id: str
    risk_type: str = "operational_risk"
    severity: str = "high"
    evidence_quote: str = ""
    reasoning: str = ""
    signal: str | None = None  # alias accepted from model

    @model_validator(mode="before")
    @classmethod
    def normalise_risk_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Accept 'signal' as alias for 'risk_type'
            if "risk_type" not in data and "signal" in data:
                data["risk_type"] = data["signal"]
            elif "signal" not in data and "risk_type" in data:
                data["signal"] = data["risk_type"]
            # Default missing fields
            if not data.get("severity"):
                data["severity"] = "high"
            if not data.get("evidence_quote"):
                data["evidence_quote"] = ""
            if not data.get("reasoning"):
                data["reasoning"] = ""
        return data


class AccountRiskSignal(BaseModel):
    signal: str
    severity: str = "high"
    reasoning: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalise_signal_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("severity"):
                data["severity"] = "high"
            # Accept 'description' as fallback for 'reasoning'
            if not data.get("reasoning") and data.get("description"):
                data["reasoning"] = data["description"]
        return data


class AccountHealthResult(BaseModel):
    prompt_version: str = HEALTH_PROMPT_VERSION
    account_id: str
    executive_summary: str
    account_level_risks: list[AccountRiskSignal] = Field(default_factory=list)
    ticket_risks: list[RiskFlag] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    ticket_join_strategy: str = "account_id"


def _find_verbatim_quote(quote: str, body: str) -> str | None:
    if not quote or not body:
        return None
    if quote in body:
        return quote

    import unicodedata

    def clean(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return s

    c_quote = clean(quote).strip().strip('"\'')
    if not c_quote:
        return None
    if c_quote in body:
        return c_quote

    c_quote_words = c_quote.split()
    if len(c_quote_words) < 2:
        return None

    pattern = r"\s+".join(re.escape(w) for w in c_quote_words)
    m = re.search(pattern, body, re.IGNORECASE)
    if m:
        return body[m.start():m.end()]

    min_words = max(3, int(len(c_quote_words) * 0.7))
    for size in range(len(c_quote_words), min_words - 1, -1):
        for start in range(len(c_quote_words) - size + 1):
            sub = c_quote_words[start:start + size]
            sub_pattern = r"\s+".join(re.escape(w) for w in sub)
            m = re.search(sub_pattern, body, re.IGNORECASE)
            if m:
                return body[m.start():m.end()]

    return None


class HealthAgent:
    def __init__(self, repo: DataRepository | None = None, llm: LLMClient | None = None):
        self.repo = repo or DataRepository()
        self.llm = llm or LLMClient()

    @staticmethod
    def _fallback_health_result(account, tickets, join_strategy) -> AccountHealthResult:
        risks: list[AccountRiskSignal] = []
        if account.health_status.lower() != "healthy":
            risks.append(
                AccountRiskSignal(
                    signal="account health status",
                    severity="high",
                    reasoning=f"The account is marked {account.health_status}.",
                )
            )
        if account.usage_trend.lower() in {"inactive", "declining"}:
            risks.append(
                AccountRiskSignal(
                    signal="usage trend",
                    severity="high",
                    reasoning=f"Recorded usage trend is {account.usage_trend}.",
                )
            )
        if account.open_tickets or account.p1_tickets_last_30d:
            risks.append(
                AccountRiskSignal(
                    signal="ticket backlog",
                    severity="medium",
                    reasoning=(
                        f"The account has {account.open_tickets} open tickets and "
                        f"{account.p1_tickets_last_30d} P1 tickets in the last 30 days."
                    ),
                )
            )

        ticket_risks: list[RiskFlag] = []
        risk_terms = (
            "critical",
            "blocked",
            "business continuity",
            "missing",
            "failed",
            "not delivered",
            "broken",
        )
        for ticket in tickets:
            body_lower = ticket.body.lower()
            if ticket.urgency in {"P1", "P2"} or any(
                term in body_lower for term in risk_terms
            ):
                first_sentence = next(
                    (part.strip() for part in ticket.body.splitlines() if part.strip()),
                    ticket.body[:200],
                )
                ticket_risks.append(
                    RiskFlag(
                        ticket_id=ticket.ticket_id,
                        risk_type="high_impact_ticket",
                        severity="high" if ticket.urgency == "P1" else "medium",
                        evidence_quote=first_sentence,
                        reasoning="The ticket contains a material operational or business-impact signal.",
                    )
                )

        summary = (
            f"{account.company} is currently marked {account.health_status} with "
            f"a {account.usage_trend.lower()} usage trend. "
            f"The account has {len(tickets)} tickets in the 90-day review window "
            f"using the {join_strategy} join strategy. "
            "The highest-impact tickets should be reviewed with the account team."
        )
        return AccountHealthResult(
            account_id=account.account_id,
            executive_summary=summary,
            account_level_risks=risks,
            ticket_risks=ticket_risks,
            talking_points=[
                "Review the highest-impact recent tickets with the account team.",
                f"Confirm the account's renewal plan for {account.renewal_date}.",
            ],
            ticket_join_strategy=join_strategy,
        )

    def summarise(self, account_id: str) -> AccountHealthResult:
        account = self.repo.get_account(account_id)
        if account is None:
            raise ValueError(f"Unknown account_id: {account_id}")

        tickets, join_strategy = self.repo.get_tickets_for_account(account_id, days=90)

        ticket_context = []
        for t in tickets:
            ticket_context.append(
                f"""TICKET {t.ticket_id}
created_at: {t.created_at.isoformat()}
status: {t.status}
urgency_metadata: {t.urgency}
category_metadata: {t.category}
subject: {t.subject}
body:
{t.body}
csat: {t.satisfaction_score}
"""
            )

        user_prompt = f"""ACCOUNT
account_id: {account.account_id}
company: {account.company}
tam: {account.tam}
plan_tier: {account.plan_tier}
arr_usd: {account.arr_usd}
seats_licensed: {account.seats_licensed}
seats_active: {account.seats_active}
products: {account.products}
health_status: {account.health_status}
usage_trend: {account.usage_trend}
open_tickets: {account.open_tickets}
p1_tickets_last_30d: {account.p1_tickets_last_30d}
customer_since: {account.customer_since}
renewal_date: {account.renewal_date}
last_qbr_date: {account.last_qbr_date}
primary_contact: {account.primary_contact.model_dump()}
escalation_notes: {account.escalation_notes}
nps_score: {account.nps_score}
last_login_days_ago: {account.last_login_days_ago}
integrations_active: {account.integrations_active}
region: {account.region}
industry: {account.industry}

TICKET JOIN STRATEGY: {join_strategy}
DATASET AS-OF: {self.repo.dataset_as_of.isoformat()}

TICKETS IN LAST 90 DAYS
{chr(10).join(ticket_context) if ticket_context else "(No matching ticket history.)"}
"""

        used_fallback = False
        try:
            result = self.llm.generate_json(
                system_prompt=HEALTH_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=AccountHealthResult,
                temperature=0,
            )
        except RuntimeError:
            used_fallback = True
            result = self._fallback_health_result(
                account,
                tickets,
                join_strategy,
            )

        if result.account_id != account_id:
            raise ValueError("LLM returned the wrong account_id.")

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", result.executive_summary.strip())
            if s.strip()
        ]
        if not 3 <= len(sentences) <= 5:
            raise ValueError(
                f"Executive summary must contain 3–5 sentences (found {len(sentences)})."
            )

        # Evidence integrity: every quote must occur verbatim in its source ticket.
        ticket_by_id = {t.ticket_id: t for t in tickets}
        validated_ticket_risks = []

        for risk in result.ticket_risks:
            ticket = ticket_by_id.get(risk.ticket_id)

            if not ticket:
                raise ValueError(
                    f"Risk references unknown ticket: {risk.ticket_id}"
                )

            verbatim_quote = _find_verbatim_quote(risk.evidence_quote, ticket.body)
            if verbatim_quote is None:
                raise ValueError(
                    f"Evidence quote is not an exact substring of {risk.ticket_id}."
                )

            validated_ticket_risks.append(
                risk.model_copy(update={"evidence_quote": verbatim_quote})
            )

        result = result.model_copy(update={"ticket_risks": validated_ticket_risks})

        # All health-agent guardrails passed.
        # Only now is the validated result safe to cache.
        if not used_fallback:
            self.llm.cache_result(
                system_prompt=HEALTH_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=AccountHealthResult,
                result=result,
            )

        return result
