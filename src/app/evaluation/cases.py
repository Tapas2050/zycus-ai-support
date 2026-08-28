from dataclasses import dataclass


# ============================================================
# TRIAGE EVALUATION
# ============================================================


@dataclass(frozen=True)
class TriageEvaluationCase:
    name: str
    ticket_id: str
    subject: str
    body: str

    expected_category: str
    expected_urgency: str
    expected_known_issue: bool

    expected_kb_sources: tuple[str, ...]


TRIAGE_CASES = [
    TriageEvaluationCase(
        name="known_dashboard_performance_issue",
        ticket_id="TKT-10073",
        subject="AnalyticsHub running extremely slowly for our team",
        body=(
            "Hi Support,\n\n"
            "We've noticed significant performance degradation in AnalyticsHub "
            "over the past 11 days. Page loads are taking 63+ seconds and "
            "Dashboard operations are timing out.\n\n"
            "Affected users: 120\n"
            "Region: US-East"
        ),
        expected_category="Performance",
        expected_urgency="P2",
        expected_known_issue=True,
        expected_kb_sources=(
            "knowledge-base/troubleshooting/performance-and-integrations.md",
        ),
    ),

    TriageEvaluationCase(
        name="high_impact_exports_performance_issue",
        ticket_id="TKT-10045",
        subject="AnalyticsHub running extremely slowly for our team",
        body=(
            "Hi Support,\n\n"
            "We've noticed significant performance degradation in AnalyticsHub "
            "over the past 7 days. Page loads are taking 79+ seconds and "
            "Exports operations are timing out.\n\n"
            "Affected users: 435\n"
            "Region: EU-West\n\n"
            "Is there a known issue or maintenance window we should be aware of?"
        ),
        expected_category="Performance",
        expected_urgency="P2",
        expected_known_issue=False,
        expected_kb_sources=(),
    ),

    TriageEvaluationCase(
        name="metadata_conflicts_with_ticket_content",
        ticket_id="TKT-10036",
        subject="Dashboard loading times unacceptable — AnalyticsHub",
        body=(
            "Our Data Sources dashboard in AnalyticsHub is now taking over "
            "100 seconds to load. This was not an issue 2 weeks ago. We have "
            "a board presentation on Wednesday and need this resolved.\n\n"
            "Customer ID: ACC-1982\n"
            "Priority: High"
        ),
        expected_category="Performance",
        expected_urgency="P2",
        expected_known_issue=True,
        expected_kb_sources=(
            "knowledge-base/troubleshooting/performance-and-integrations.md",
        ),
    ),
]


# ============================================================
# HEALTH EVALUATION
# ============================================================


@dataclass(frozen=True)
class HealthEvaluationCase:
    name: str
    account_id: str

    expected_account_level_signals: tuple[str, ...]

    expected_ticket_risk_ticket_ids: tuple[str, ...]

    adversarial: bool = False


HEALTH_CASES = [
    HealthEvaluationCase(
        name="at_risk_account_with_high_impact_tickets",
        account_id="ACC-3336",

        expected_account_level_signals=(
            "health_status + usage_trend + renewal_date",
        ),

        expected_ticket_risk_ticket_ids=(
            "TKT-10393",
            "TKT-10398",
        ),

        adversarial=False,
    ),

    HealthEvaluationCase(
        name="healthy_account_with_customer_declared_p1",
        account_id="ACC-3033",

        expected_account_level_signals=(
            "Renewal proximity with limited recent engagement",
            "Conflicting engagement signals",
        ),

        expected_ticket_risk_ticket_ids=(
            "TKT-10209",
        ),

        adversarial=True,
    ),
]