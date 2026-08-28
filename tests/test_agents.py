import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from app.health_agent import AccountHealthResult, AccountRiskSignal, HealthAgent
from app.models import TicketInput
from app.triage_agent import KBMatch, TriageAgent, TriageResult


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, **kwargs):
        return kwargs["response_model"].model_validate(self.payload)

    def cache_result(self, **kwargs):
        # No-op for unit tests.
        #
        # The real LLMClient persists validated results in its cache,
        # but these tests only need to verify agent behaviour.
        pass


def test_triage_agent_validates_structured_output():
    ticket = TicketInput(
        subject="AnalyticsHub dashboard is timing out",
        body="The dashboard takes over 100 seconds to load.",
    )

    kb_match = KBMatch(
        source_file="knowledge-base/troubleshooting/performance-and-integrations.md",
        heading="AnalyticsHub: Dashboard Timeout",
        relevance_reason="Direct dashboard timeout troubleshooting guidance.",
    )

    fake = FakeLLM(
        {
            "product": "AnalyticsHub",
            "product_area": "Dashboard",
            "category": "Performance",
            "urgency": "P2",
            "rationale": (
                "Severe dashboard latency is blocking a time-sensitive "
                "business presentation."
            ),
            "known_issue": True,
            "knowledge_base_matches": [kb_match.model_dump()],
            "recommended_responder_team": "Performance Engineering",
            "first_response": (
                "We can help investigate the dashboard latency."
            ),
        }
    )

    agent = TriageAgent(llm=fake)
    result = agent.triage(ticket)

    assert isinstance(result, TriageResult)
    assert result.category == "Performance"
    assert result.urgency == "P2"


def test_health_agent_rejects_non_verbatim_ticket_evidence():
    from app.data_repository import DataRepository

    repo = DataRepository("data")
    tickets, _ = repo.get_tickets_for_account("ACC-6254")
    target = next(
        t for t in tickets
        if t.ticket_id == "TKT-10196"
    )

    fake = FakeLLM(
        {
            "account_id": "ACC-6254",
            "executive_summary": (
                "The account is churning. Usage is inactive. "
                "Reliability concerns require attention."
            ),
            "account_level_risks": [
                {
                    "signal": "Churn risk",
                    "severity": "High",
                    "reasoning": "The account is marked Churning.",
                }
            ],
            "ticket_risks": [
                {
                    "ticket_id": target.ticket_id,
                    "risk_type": "escalation",
                    "severity": "High",
                    "evidence_quote": (
                        "This quote is not actually in the ticket."
                    ),
                    "reasoning": "Invalid evidence.",
                }
            ],
            "talking_points": [
                "Review reliability incidents before renewal."
            ],
            "ticket_join_strategy": "company_fallback",
        }
    )

    agent = HealthAgent(repo=repo, llm=fake)

    try:
        agent.summarise("ACC-6254")
    except ValueError as exc:
        assert "exact substring" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid evidence quote to be rejected."
        )