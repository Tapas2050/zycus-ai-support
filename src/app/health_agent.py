from datetime import datetime

from pydantic import BaseModel, Field

from app.data_repository import DataRepository
from app.llm_client import LLMClient
from app.prompts import HEALTH_PROMPT_VERSION, HEALTH_SYSTEM_PROMPT


class RiskFlag(BaseModel):
    ticket_id: str
    risk_type: str
    severity: str
    evidence_quote: str
    reasoning: str


class AccountRiskSignal(BaseModel):
    signal: str
    severity: str
    reasoning: str


class AccountHealthResult(BaseModel):
    prompt_version: str = HEALTH_PROMPT_VERSION
    account_id: str
    executive_summary: str
    account_level_risks: list[AccountRiskSignal] = Field(default_factory=list)
    ticket_risks: list[RiskFlag] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    ticket_join_strategy: str


class HealthAgent:
    def __init__(self, repo: DataRepository | None = None, llm: LLMClient | None = None):
        self.repo = repo or DataRepository()
        self.llm = llm or LLMClient()

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

        result = self.llm.generate_json(
            system_prompt=HEALTH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=AccountHealthResult,
            temperature=0,
        )

        if result.account_id != account_id:
            raise ValueError("LLM returned the wrong account_id.")

        if not 3 <= len(
            [s for s in result.executive_summary.split(".") if s.strip()]
        ) <= 5:
            raise ValueError("Executive summary must contain 3–5 sentences.")

        # Evidence integrity: every quote must occur verbatim in its source ticket.
        # Evidence integrity: every quote must occur verbatim in its source ticket.
        ticket_by_id = {t.ticket_id: t for t in tickets}

        for risk in result.ticket_risks:
            ticket = ticket_by_id.get(risk.ticket_id)

            if not ticket:
                raise ValueError(
                    f"Risk references unknown ticket: {risk.ticket_id}"
                )

            if risk.evidence_quote not in ticket.body:
                raise ValueError(
                    f"Evidence quote is not an exact substring of {risk.ticket_id}."
                )

        # All health-agent guardrails passed.
        # Only now is the validated result safe to cache.
        self.llm.cache_result(
            system_prompt=HEALTH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=AccountHealthResult,
            result=result,
        )

        return result
