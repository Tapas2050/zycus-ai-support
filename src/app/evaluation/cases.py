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
    required_response_terms: tuple[str, ...] = ()
    adversarial: bool = False


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
        required_response_terms=("query profiler", "query cache", "date filter"),
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
        required_response_terms=("export", "region", "request id"),
        adversarial=True,
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
        required_response_terms=("query profiler", "query cache", "date filter"),
    ),

    TriageEvaluationCase(
        name="explicit_data_loss_overrides_metadata_priority",
        ticket_id="TKT-10249",
        subject="URGENT: Missing data in SecureVault Encryption",
        body=(
            "URGENT — We are missing critical data in SecureVault's Encryption module.\n\n"
            "Last known good state: Monday at 09:00 UTC\n"
            "Missing records: approximately 3312\n"
            "Affected workflows: Engineering team operations\n\n"
            "This is a P1 for us. Please escalate immediately. We have business continuity at risk."
        ),
        expected_category="Data Loss",
        expected_urgency="P1",
        expected_known_issue=False,
        expected_kb_sources=(),
        required_response_terms=("missing records", "last known good", "scope"),
    ),

    TriageEvaluationCase(
        name="configuration_question_is_how_to",
        ticket_id="TKT-10255",
        subject="How do I configure Dashboard in AnalyticsHub?",
        body=(
            "Hi,\n\n"
            "I'm trying to set up Dashboard for our team but can't find clear documentation. Specifically, I need to know:\n\n"
            "1. How to assign to team\n"
            "2. What permissions are required\n"
            "3. Whether this integrates with HubSpot\n\n"
            "We're on the Enterprise plan. Thanks in advance."
        ),
        expected_category="How-To",
        expected_urgency="P4",
        expected_known_issue=False,
        expected_kb_sources=(),
        required_response_terms=("dashboard", "permissions", "hubspot"),
    ),

    TriageEvaluationCase(
        name="capability_request_is_feature_request",
        ticket_id="TKT-10336",
        subject="Feature request: export Triggers data to Jira",
        body=(
            "Hi team,\n\n"
            "We'd love to see native export functionality from Triggers directly to Jira. Currently we're using a manual workaround which takes our team hours each week.\n\n"
            "Use case: third-party audit requirements\n\n"
            "Would this be on the roadmap? Happy to join a beta."
        ),
        expected_category="Feature Request",
        expected_urgency="P4",
        expected_known_issue=False,
        expected_kb_sources=(),
        required_response_terms=("roadmap", "beta", "use case"),
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

    HealthEvaluationCase(
        name="churning_account_with_data_loss_ticket",
        account_id="ACC-6254",
        expected_account_level_signals=(
            "engagement_conflict",
            "renewal_risk",
        ),
        expected_ticket_risk_ticket_ids=(
            "TKT-10196",
        ),
    ),

    HealthEvaluationCase(
        name="healthy_account_with_declining_usage",
        account_id="ACC-8014",
        expected_account_level_signals=(
            "engagement_conflict",
        ),
        expected_ticket_risk_ticket_ids=(),
    ),

    HealthEvaluationCase(
        name="new_account_with_sparse_history",
        account_id="ACC-7893",
        expected_account_level_signals=(),
        expected_ticket_risk_ticket_ids=(),
    ),
]
