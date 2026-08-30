from app.evaluation.cases import (
    HealthEvaluationCase,
    TriageEvaluationCase,
)
from app.health_agent import AccountHealthResult
from app.kb_retriever import KBRetriever
from app.models import TicketInput
from app.triage_agent import (
    TriageResult,
    _kb_match_supports_ticket,
    _ticket_has_explicit_p1_evidence,
)


# ============================================================
# TRIAGE SCORING
# ============================================================


def _kb_match_has_grounded_evidence(
    ticket: TicketInput,
    match,
    kb: KBRetriever,
) -> bool:
    """Verify a cited KB identity against the actual text in the KB."""
    chunk = next(
        (
            candidate
            for candidate in kb.chunks
            if candidate.source_file == match.source_file
            and candidate.heading == match.heading
        ),
        None,
    )
    return chunk is not None and _kb_match_supports_ticket(ticket, match, chunk.text)


def score_triage(
    result: TriageResult,
    case: TriageEvaluationCase,
) -> dict:
    """
    Evaluate a TriageAgent result against an evaluation case.

    Strict checks:
        - category
        - urgency
        - known_issue

    KB attribution is evaluated semantically:
    the model must cite at least the expected source when the
    evaluation case requires one, but additional legitimate
    retrieved sources do not automatically cause failure.
    """

    # ---------------------------------------------------------
    # 1. CATEGORY
    # ---------------------------------------------------------

    category_passed = (
        result.category == case.expected_category
    )

    # ---------------------------------------------------------
    # 2. URGENCY
    # ---------------------------------------------------------

    urgency_passed = (
        result.urgency == case.expected_urgency
    )

    # ---------------------------------------------------------
    # 3. KNOWN ISSUE
    # ---------------------------------------------------------

    known_issue_passed = (
        result.known_issue == case.expected_known_issue
    )

    # ---------------------------------------------------------
    # 4. KB SOURCES
    # ---------------------------------------------------------

    ticket = TicketInput(subject=case.subject, body=case.body)

    actual_kb_sources = tuple(
        sorted(
            {
                match.source_file
                for match in result.knowledge_base_matches
            }
        )
    )

    expected_kb_sources = tuple(
        sorted(case.expected_kb_sources)
    )

    kb = KBRetriever()
    grounded_matches = [
        match
        for match in result.knowledge_base_matches
        if _kb_match_has_grounded_evidence(ticket, match, kb)
    ]
    grounded_sources = {match.source_file for match in grounded_matches}
    all_citations_grounded = (
        len(grounded_matches) == len(result.knowledge_base_matches)
    )

    if expected_kb_sources:
        # Every expected source must be present.
        #
        # Additional retrieved/cited sources are allowed because
        # the LLM may legitimately identify more than one relevant
        # document. However, the citation must still align with the
        # expected ticket semantics; a same-file but wrong-operation
        # heading should fail evaluation.
        kb_sources_passed = (
            all(
                source in grounded_sources
                for source in expected_kb_sources
            )
            and all_citations_grounded
        )
    else:
        # When the expected answer says that no KB source should
        # support the issue, there must be no KB citations.
        kb_sources_passed = (
            len(actual_kb_sources) == 0
        )

    # ---------------------------------------------------------
    # 5. OVERALL PASS / FAIL
    # ---------------------------------------------------------

    urgency_evidence_passed = not (
        result.urgency == "P1" and not _ticket_has_explicit_p1_evidence(ticket)
    )
    rationale_quality_passed = bool(result.rationale.strip())
    first_response_quality_passed = bool(result.first_response.strip())
    required_response_terms = tuple(
        term.lower() for term in case.required_response_terms
    )
    response_guidance_passed = (
        not required_response_terms
        or any(term in result.first_response.lower() for term in required_response_terms)
    )

    citation_quality_passed = all(
        bool(match.source_file.strip())
        and bool(match.relevance_reason.strip())
        and _kb_match_has_grounded_evidence(ticket, match, kb)
        for match in result.knowledge_base_matches
    )

    semantic_kb_gate = all_citations_grounded

    passed = all(
        [
            category_passed,
            urgency_passed,
            urgency_evidence_passed,
            known_issue_passed,
            kb_sources_passed,
            semantic_kb_gate,
            rationale_quality_passed,
            first_response_quality_passed,
            response_guidance_passed,
            citation_quality_passed,
        ]
    )

    # ---------------------------------------------------------
    # 6. QUALITY SCORE
    # ---------------------------------------------------------

    quality_score = sum(
        [
            category_passed,
            urgency_passed,
            urgency_evidence_passed,
            known_issue_passed,
            kb_sources_passed,
            semantic_kb_gate,
            rationale_quality_passed,
            first_response_quality_passed,
            response_guidance_passed,
            citation_quality_passed,
        ]
    ) / 10

    return {
        "case_id": case.ticket_id,
        "case_name": case.name,
        "adversarial": getattr(case, "adversarial", False),

        "passed": passed,

        "quality_score": round(
            quality_score,
            3,
        ),

        "category": {
            "expected": case.expected_category,
            "actual": result.category,
            "passed": category_passed,
        },

        "urgency": {
            "expected": case.expected_urgency,
            "actual": result.urgency,
            "passed": urgency_passed,
        },

        "urgency_evidence": {
            "passed": urgency_evidence_passed,
        },

        "known_issue": {
            "expected": case.expected_known_issue,
            "actual": result.known_issue,
            "passed": known_issue_passed,
        },

        "kb_sources": {
            "expected": expected_kb_sources,
            "actual": actual_kb_sources,
            "passed": kb_sources_passed,
        },

        "semantic_kb_gate": {
            "passed": semantic_kb_gate,
        },

        "grounded_kb_sources": {
            "actual": tuple(sorted(grounded_sources)),
            "passed": all_citations_grounded,
        },

        "citation_quality": {
            "passed": citation_quality_passed,
        },

        "rationale_quality": {
            "passed": rationale_quality_passed,
        },

        "first_response_quality": {
            "passed": first_response_quality_passed,
        },

        "response_guidance": {
            "expected_any_of": required_response_terms,
            "passed": response_guidance_passed,
        },
    }


# ============================================================
# HEALTH SIGNAL NORMALISATION
# ============================================================


def _normalise_signal(signal: str) -> str:
    """
    Convert model-generated account-risk signal names into
    stable evaluation concepts.

    The LLM is allowed to phrase the same risk differently.

    Example:

        "Conflicting engagement signals"
            -> "engagement_conflict"

        "Login inactivity vs. health status"
            -> "engagement_conflict"

        "Renewal proximity with limited recent engagement"
            -> "renewal_risk"
    """

    value = signal.strip().lower()

    # ---------------------------------------------------------
    # Engagement / adoption conflict
    # ---------------------------------------------------------

    if (
        "engagement" in value
        or "login inactivity" in value
        or "usage trend" in value
        or "adoption" in value
    ):
        return "engagement_conflict"

    # ---------------------------------------------------------
    # Renewal risk
    # ---------------------------------------------------------

    if (
        "renewal" in value
        or "renewal date" in value
    ):
        return "renewal_risk"

    # ---------------------------------------------------------
    # Escalation / P1 data conflict
    # ---------------------------------------------------------

    if (
        "escalation" in value
        or "p1" in value
    ):
        return "escalation_data_conflict"

    # ---------------------------------------------------------
    # Sentiment
    # ---------------------------------------------------------

    if (
        "nps" in value
        or "sentiment" in value
    ):
        return "sentiment_gap"

    # ---------------------------------------------------------
    # Ticket backlog
    # ---------------------------------------------------------

    if (
        "ticket backlog" in value
        or "open ticket" in value
        or "open risk item" in value
    ):
        return "ticket_backlog"

    # ---------------------------------------------------------
    # Unknown signal
    # ---------------------------------------------------------

    return value


# ============================================================
# HEALTH SCORING
# ============================================================


def score_health(
    result: AccountHealthResult,
    case: HealthEvaluationCase,
) -> dict:
    """
    Evaluate an AccountHealthResult against an evaluation case.

    Account-level risk signals are compared semantically rather
    than requiring the LLM to reproduce the exact wording used
    in the evaluation case.

    Ticket-risk expectations remain ID-based because ticket IDs
    are deterministic and unambiguous.
    """

    # ---------------------------------------------------------
    # 1. ACCOUNT ID
    # ---------------------------------------------------------

    account_id_passed = (
        result.account_id == case.account_id
    )

    # ---------------------------------------------------------
    # 2. ACCOUNT-LEVEL SIGNALS
    # ---------------------------------------------------------

    actual_account_signals = {
        _normalise_signal(signal.signal)
        for signal in result.account_level_risks
    }

    expected_account_signals = {
        _normalise_signal(signal)
        for signal in case.expected_account_level_signals
    }

    # Expected concepts must be present.
    #
    # Additional risks are allowed because a health summariser
    # should be able to surface more than the minimum expected
    # set of risks.
    account_signals_passed = (
        expected_account_signals.issubset(
            actual_account_signals
        )
    )

    # ---------------------------------------------------------
    # 3. EXPECTED TICKET RISKS
    # ---------------------------------------------------------

    actual_ticket_ids = {
        risk.ticket_id
        for risk in result.ticket_risks
    }

    expected_ticket_ids = set(
        case.expected_ticket_risk_ticket_ids
    )

    # Again, expected risks must exist.
    # Additional legitimate risks are allowed.
    ticket_risks_passed = (
        expected_ticket_ids.issubset(
            actual_ticket_ids
        )
    )

    # ---------------------------------------------------------
    # 4. OVERALL PASS / FAIL
    # ---------------------------------------------------------

    passed = all(
        [
            account_id_passed,
            account_signals_passed,
            ticket_risks_passed,
        ]
    )

    # ---------------------------------------------------------
    # 5. QUALITY SCORE
    # ---------------------------------------------------------

    quality_score = sum(
        [
            account_id_passed,
            account_signals_passed,
            ticket_risks_passed,
        ]
    ) / 3

    return {
        "case_id": case.account_id,
        "case_name": case.name,
        "adversarial": getattr(case, "adversarial", False),

        "passed": passed,

        "quality_score": round(
            quality_score,
            3,
        ),

        "account_id": {
            "expected": case.account_id,
            "actual": result.account_id,
            "passed": account_id_passed,
        },

        "account_level_signals": {
            "expected": sorted(
                expected_account_signals
            ),
            "actual": sorted(
                actual_account_signals
            ),
            "passed": account_signals_passed,
        },

        "ticket_risks": {
            "expected": sorted(
                expected_ticket_ids
            ),
            "actual": sorted(
                actual_ticket_ids
            ),
            "passed": ticket_risks_passed,
        },
    }
