import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from app.health_agent import AccountHealthResult, AccountRiskSignal, HealthAgent
from app.llm_client import _normalize_json_schema
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


def test_llm_schema_requires_nested_fields_for_array_items():
    schema = AccountHealthResult.model_json_schema()
    normalized = _normalize_json_schema(schema)

    ticket_item = normalized["properties"]["ticket_risks"]["items"]
    assert "ticket_id" in ticket_item["required"]
    assert "risk_type" in ticket_item["required"]
    assert "severity" in ticket_item["required"]
    assert "evidence_quote" in ticket_item["required"]
    assert "reasoning" in ticket_item["required"]

    account_item = normalized["properties"]["account_level_risks"]["items"]
    assert "signal" in account_item["required"]
    assert "severity" in account_item["required"]
    assert "reasoning" in account_item["required"]


def test_llm_client_retries_transient_api_failures():
    import tempfile

    import app.config as config
    from app.llm_client import LLMClient

    object.__setattr__(config.settings, "llm_api_key", "test-key")
    object.__setattr__(config.settings, "llm_model", "test-model")
    object.__setattr__(config.settings, "llm_base_url", None)
    object.__setattr__(config.settings, "llm_cache_dir", tempfile.mkdtemp(prefix="llm-cache-"))

    llm = LLMClient()
    expected = AccountHealthResult(
        account_id="ACC-9999",
        executive_summary="The account is stable and healthy.",
        account_level_risks=[
            {
                "signal": "engagement_conflict",
                "severity": "medium",
                "reasoning": "Recent engagement is uneven.",
            }
        ],
        ticket_risks=[],
        talking_points=["Keep the QBR focused on usage and adoption."],
        ticket_join_strategy="account_id",
    )

    class FakeResponse:
        class Choice:
            message = type(
                "Message",
                (),
                {"content": expected.model_dump_json()},
            )()
            finish_reason = "stop"

        choices = [Choice()]

    class RetryableRateLimit(Exception):
        status_code = 429

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RetryableRateLimit("rate limited")
            return FakeResponse()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    llm.client = type("FakeClient", (), {"chat": FakeChat()})()

    result = llm.generate_json(
        system_prompt="system prompt",
        user_prompt="user prompt",
        response_model=AccountHealthResult,
        temperature=0,
    )

    assert result.account_id == "ACC-9999"
    assert result.account_level_risks[0].signal == "engagement_conflict"


def test_llm_client_retries_null_provider_response():
    import tempfile

    import app.config as config
    from app.llm_client import LLMClient

    object.__setattr__(config.settings, "llm_api_key", "test-key")
    object.__setattr__(config.settings, "llm_model", "test-model")
    object.__setattr__(config.settings, "llm_base_url", None)
    object.__setattr__(config.settings, "llm_cache_dir", tempfile.mkdtemp(prefix="llm-cache-"))

    llm = LLMClient()
    expected = AccountHealthResult(
        account_id="ACC-1000",
        executive_summary="The account remains healthy.",
        account_level_risks=[],
        ticket_risks=[],
        talking_points=["Retain the current onboarding path."],
        ticket_join_strategy="account_id",
    )

    class FakeResponse:
        class Choice:
            message = type(
                "Message",
                (),
                {"content": expected.model_dump_json()},
            )()
            finish_reason = "stop"

        choices = [Choice()]

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return None
            return FakeResponse()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    llm.client = type("FakeClient", (), {"chat": FakeChat()})()

    result = llm.generate_json(
        system_prompt="system prompt",
        user_prompt="user prompt",
        response_model=AccountHealthResult,
        temperature=0,
    )

    assert result.account_id == "ACC-1000"
    assert result.executive_summary == "The account remains healthy."


def test_triage_agent_rewrites_known_issue_when_kb_only_matches_generic_performance_terms():
    ticket = TicketInput(
        subject="AnalyticsHub running extremely slowly for our team",
        body=(
            "We've noticed significant performance degradation in AnalyticsHub. "
            "Exports operations are timing out for 435 users in EU-West."
        ),
    )

    fake = FakeLLM(
        {
            "product": "AnalyticsHub",
            "product_area": "Exports",
            "category": "Performance",
            "urgency": "P2",
            "rationale": "The issue is performance-related and likely tied to exports.",
            "known_issue": True,
            "knowledge_base_matches": [
                {
                    "source_file": "knowledge-base/troubleshooting/performance-and-integrations.md",
                    "heading": "AnalyticsHub: Dashboard Timeout",
                    "relevance_reason": "General analytics performance issue.",
                }
            ],
            "recommended_responder_team": "Performance Engineering",
            "first_response": "We are investigating performance degradation in exports.",
        }
    )

    agent = TriageAgent(llm=fake)
    result = agent.triage(ticket)

    assert result.known_issue is False
    assert result.knowledge_base_matches == []
