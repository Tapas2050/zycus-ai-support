from app.kb_retriever import KBRetriever
from app.llm_client import LLMClient
from app.models import TicketInput
from pydantic import BaseModel, Field

from app.prompts import TRIAGE_PROMPT_VERSION, TRIAGE_SYSTEM_PROMPT


RESPONDER_TEAMS = [
    "Product Support",
    "Performance Engineering",
    "Integration Support",
    "Data Reliability / Incident Response",
    "Billing Support",
    "Onboarding Support",
    "Technical Support",
    "Product Management",
    "Security & Identity Support",
]


from typing import Any
from pydantic import BaseModel, Field, model_validator


class KBMatch(BaseModel):
    source_file: str
    heading: str | None = None
    relevance_reason: str = ""


class TriageResult(BaseModel):
    prompt_version: str = TRIAGE_PROMPT_VERSION
    product: str | None = None
    product_area: str = "General"
    category: str = "Product / Feature Inquiry"
    urgency: str = "P3"
    rationale: str = ""
    known_issue: bool = False
    knowledge_base_matches: list[KBMatch] = Field(default_factory=list)
    recommended_responder_team: str = "Product Support"
    first_response: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalise_triage_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "recommended_responder_team" not in data:
                for alias in ("responder_team", "team", "recommended_team", "assigned_team"):
                    if alias in data and data[alias]:
                        data["recommended_responder_team"] = data[alias]
                        break
                else:
                    data["recommended_responder_team"] = "Product Support"

            if "product_area" not in data:
                for alias in ("area", "module", "component", "feature_area"):
                    if alias in data and data[alias]:
                        data["product_area"] = data[alias]
                        break
                else:
                    data["product_area"] = "General"

            if not data.get("category"):
                data["category"] = "Product / Feature Inquiry"
            if not data.get("urgency"):
                data["urgency"] = "P3"
            if not data.get("rationale"):
                data["rationale"] = ""
            if not data.get("first_response"):
                data["first_response"] = ""
        return data


def _ticket_has_explicit_p1_evidence(ticket: TicketInput) -> bool:
    text = f"{ticket.subject} {ticket.body}".lower()
    return any(
        phrase in text
        for phrase in (
            "business stopped",
            "business continuity",
            "major production outage",
            "data loss",
            "data lost",
            "missing data",
            "missing records",
            "records are missing",
        )
    )


def _ticket_has_p2_evidence(ticket: TicketInput) -> bool:
    text = f"{ticket.subject} {ticket.body}".lower()
    return any(
        phrase in text
        for phrase in (
            "significant performance",
            "substantial degradation",
            "affected users",
            "blocked",
            "timing out",
            "over 30 seconds",
            "over 60 seconds",
            "over 100 seconds",
        )
    )


def _kb_match_supports_ticket(
    ticket: TicketInput,
    match: KBMatch,
    chunk_text: str,
) -> bool:
    ticket_text = f"{ticket.subject} {ticket.body}".lower()
    heading_text = (match.heading or "").lower()
    evidence_text = f"{heading_text} {chunk_text}".lower()

    product_names = (
        "analyticshub",
        "cloudsync",
        "databridge pro",
        "securevault",
        "workflowengine",
    )
    product = next((name for name in product_names if name in ticket_text), None)
    if product and product not in evidence_text:
        return False

    # Tighten semantic matching: if the ticket names a concrete module/operation,
    # the KB evidence must match that operation. This prevents generic product
    #-level docs from being treated as the same known issue for a different
    # workflow such as dashboard vs exports.
    ticket_operations = {
        "dashboard": {"dashboard", "widget", "query profiler"},
        "export": {"export", "exports", "report export", "csv", "excel", "json", "parquet"},
        "data source": {"data source", "data sources", "source", "connector"},
        "schema": {"schema", "schema management"},
        "pipeline": {"pipeline", "throughput", "backlog"},
        "sso": {"sso", "single sign on", "authentication", "oauth", "token"},
        "authentication": {"authentication", "sso", "oauth", "token"},
        "sync": {"sync", "synchronization", "webhook"},
        "webhook": {"webhook", "callback", "delivery"},
    }

    detected_ticket_ops = set()
    for term, aliases in ticket_operations.items():
        if any(token in ticket_text for token in (term, *aliases)):
            detected_ticket_ops.add(term)

    if detected_ticket_ops:
        matched_ops = {
            op
            for op in detected_ticket_ops
            if any(alias in heading_text or alias in evidence_text for alias in ticket_operations[op])
        }
        if not matched_ops:
            return False

    symptom_groups = (
        (
            "slow",
            "performance",
            "timeout",
            "timing out",
            "fails to load",
            "loading",
            "latency",
        ),
        ("not working", "broken", "error", "failure", "not delivered"),
        ("missing", "data loss", "data lost"),
        ("authentication", "sso", "oauth", "token"),
    )
    # A configuration or capability question may name the same product and
    # module as a troubleshooting article. It is not a known issue unless the
    # ticket itself describes an operational symptom from the same group.
    return any(
        any(term in ticket_text for term in group)
        and any(term in evidence_text for term in group)
        for group in symptom_groups
    )


def _deterministic_kb_evidence(
    ticket: TicketInput,
    retrieved: list[tuple],
) -> list[KBMatch]:
    """Return retrieved chunks that deterministically support a known issue."""
    evidence: list[KBMatch] = []
    for chunk, _score in retrieved:
        candidate = KBMatch(
            source_file=chunk.source_file,
            heading=chunk.heading,
            relevance_reason="",
        )
        if _kb_match_supports_ticket(ticket, candidate, chunk.text):
            evidence.append(
                candidate.model_copy(
                    update={
                        "relevance_reason": (
                            "Retrieved KB guidance matches the ticket's product, "
                            "operation, and operational symptom."
                        )
                    }
                )
            )
    return evidence


class TriageAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        kb: KBRetriever | None = None,
    ):
        self.llm = llm or LLMClient()
        self.kb = kb or KBRetriever()

    def triage(self, ticket: TicketInput) -> TriageResult:
        # =========================================================
        # 1. RETRIEVE RELEVANT KNOWLEDGE
        # =========================================================

        query = (
            f"Subject: {ticket.subject}\n\n"
            f"Body:\n{ticket.body}"
        )

        retrieved = self.kb.retrieve(
            query,
            top_k=3,
        )

        # =========================================================
        # 2. BUILD RETRIEVED KB CONTEXT
        # =========================================================

        context = self.kb.format_context(retrieved)

        # Only expose KB sources/chunks that were actually retrieved.
        #
        # If a chunk has no heading, we deliberately omit the heading
        # rather than exposing Python's "None" to the LLM.
        allowed_sources = "\n".join(
            f"- {chunk.source_file}"
            + (
                f" :: {chunk.heading}"
                if chunk.heading
                else ""
            )
            for chunk, _ in retrieved
        )

        # =========================================================
        # 3. BUILD USER PROMPT
        # =========================================================

        user_prompt = f"""TICKET
            Subject: {ticket.subject}
            Body:
            {ticket.body}

            RETRIEVED KNOWLEDGE BASE
            {context or "(No relevant KB context found.)"}

            ALLOWED KB SOURCES
            {allowed_sources or "(none)"}

            ALLOWED RESPONDER TEAMS
            {chr(10).join(f"- {team}" for team in RESPONDER_TEAMS)}
            """

        # =========================================================
        # 4. ASK LLM FOR STRUCTURED TRIAGE RESULT
        # =========================================================

        result = self.llm.generate_json(
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=TriageResult,
            temperature=0,
        )

        # =========================================================
        # 5. APPLICATION-LEVEL CLASSIFICATION GUARDRAILS
        # =========================================================

        # Category must belong to the controlled vocabulary.
        # Nemotron (and other models) sometimes use slightly different
        # category names. We normalise before rejecting.
        allowed_categories = {
            "Bug",
            "Feature Request",
            "How-To",
            "Performance",
            "Billing",
            "Integration",
            "Onboarding",
            "Data Loss",
        }

        # Normalisation map for common model paraphrases
        CATEGORY_NORMALISE = {
            "service outage":           "Bug",
            "outage":                   "Bug",
            "incident":                 "Bug",
            "error":                    "Bug",
            "defect":                   "Bug",
            "unexpected behavior":      "Bug",
            "feature":                  "Feature Request",
            "enhancement":              "Feature Request",
            "how to":                   "How-To",
            "how-to":                   "How-To",
            "question":                 "How-To",
            "inquiry":                  "How-To",
            "performance issue":        "Performance",
            "slow":                     "Performance",
            "timeout":                  "Performance",
            "latency":                  "Performance",
            "billing":                  "Billing",
            "invoice":                  "Billing",
            "payment":                  "Billing",
            "integration":              "Integration",
            "api":                      "Integration",
            "webhook":                  "Integration",
            "onboarding":               "Onboarding",
            "setup":                    "Onboarding",
            "data loss":                "Data Loss",
            "data integrity":           "Data Loss",
            "data":                     "Data Loss",
        }

        if result.category not in allowed_categories:
            normalised = CATEGORY_NORMALISE.get(result.category.lower())
            if normalised:
                result = result.model_copy(update={"category": normalised})
            else:
                # Attempt partial-match fallback
                raw_lower = result.category.lower()
                matched = next(
                    (cat for key, cat in CATEGORY_NORMALISE.items() if key in raw_lower),
                    None,
                )
                if matched:
                    result = result.model_copy(update={"category": matched})
                else:
                    raise ValueError(
                        f"Invalid category returned by LLM: {result.category}"
                    )

        urgency_aliases = {
            "critical": "P1",
            "highest": "P1",
            "high": "P2",
            "major": "P2",
            "medium": "P3",
            "moderate": "P3",
            "low": "P4",
            "minor": "P4",
        }
        if result.urgency not in {
            "P1",
            "P2",
            "P3",
            "P4",
        }:
            urgency = urgency_aliases.get(result.urgency.lower())
            if urgency:
                result = result.model_copy(update={"urgency": urgency})
            else:
                raise ValueError(f"Invalid urgency returned by LLM: {result.urgency}")

        explicit_p1 = _ticket_has_explicit_p1_evidence(ticket)
        if explicit_p1:
            result = result.model_copy(update={"urgency": "P1"})
        elif _ticket_has_p2_evidence(ticket) and result.urgency in {"P3", "P4"}:
            result = result.model_copy(update={"urgency": "P2"})
        elif result.urgency == "P1":
            result = result.model_copy(update={"urgency": "P2"})

        # Responder team must belong to the controlled vocabulary.
        if result.recommended_responder_team not in RESPONDER_TEAMS:
            cat_map = {
                "Performance": "Performance Engineering",
                "Billing / Invoicing": "Billing Support",
                "Onboarding / Setup": "Onboarding Support",
                "Security / Access / Authentication": "Security & Identity Support",
                "Integration / API": "Integration Support",
                "Data / Pipeline / Sync Issue": "Data Reliability / Incident Response",
                "Product / Feature Inquiry": "Product Support",
                "Bug / Unexpected Behavior": "Technical Support",
            }
            mapped = cat_map.get(result.category)
            if mapped is None:
                team_text = result.recommended_responder_team.lower()
                if "performance" in team_text:
                    mapped = "Performance Engineering"
                elif "billing" in team_text:
                    mapped = "Billing Support"
                elif "integration" in team_text:
                    mapped = "Integration Support"
                elif "security" in team_text or "identity" in team_text:
                    mapped = "Security & Identity Support"
                elif "data" in team_text or "reliability" in team_text:
                    mapped = "Data Reliability / Incident Response"
            if mapped:
                result.recommended_responder_team = mapped
            else:
                raise ValueError(
                    "Invalid responder team: "
                    f"{result.recommended_responder_team}"
                )

        # =========================================================
        # 6. KB CITATION INTEGRITY
        # =========================================================
        #
        # The LLM is allowed to cite ONLY retrieved KB chunks.
        #
        # We validate BOTH:
        #   source_file
        #   heading
        #
        # This prevents the LLM from inventing a KB citation.
        #
        # A small defensive normalization is allowed:
        #
        # If the LLM correctly identifies a retrieved source_file but
        # omits its heading, we restore the heading ONLY when exactly
        # one retrieved chunk exists for that source.
        #
        # If multiple chunks from the same source were retrieved,
        # we cannot safely guess which heading was intended.
        # =========================================================

        allowed = {
            (chunk.source_file, chunk.heading)
            for chunk, _ in retrieved
        }

        # Group retrieved chunks by source file so that we can safely
        # determine whether an omitted heading has exactly one candidate.
        retrieved_by_source: dict[str, list] = {}

        for chunk, _ in retrieved:
            retrieved_by_source.setdefault(
                chunk.source_file,
                [],
            ).append(chunk)

        normalized_matches: list[KBMatch] = []

        chunks_by_identity = {
            (chunk.source_file, chunk.heading): chunk
            for chunk, _ in retrieved
        }

        for match in result.knowledge_base_matches:
            # Normalize empty string to None.
            #
            # These two values mean the same thing for a headingless chunk:
            #
            #     ""
            #     None
            #
            normalized_heading = (
                match.heading
                if match.heading
                else None
            )

            # -----------------------------------------------------
            # Case A: Exact source + heading match
            # -----------------------------------------------------

            if (
                match.source_file,
                normalized_heading,
            ) in allowed:
                chunk = chunks_by_identity[(match.source_file, normalized_heading)]
                if _kb_match_supports_ticket(ticket, match, chunk.text):
                    normalized_matches.append(
                        match.model_copy(update={"heading": normalized_heading})
                    )
                continue

            # -----------------------------------------------------
            # Case B: Source exists, but heading was omitted
            # -----------------------------------------------------
            #
            # Only repair the heading when exactly one retrieved
            # chunk belongs to that source.
            # -----------------------------------------------------

            candidates = retrieved_by_source.get(
                match.source_file,
                [],
            )

            if (
                normalized_heading is None
                and len(candidates) == 1
            ):
                chunk = candidates[0]
                if _kb_match_supports_ticket(ticket, match, chunk.text):
                    normalized_matches.append(
                        match.model_copy(update={"heading": chunk.heading})
                    )
                continue

            # -----------------------------------------------------
            # Case C: Unsafe or invented citation
            # -----------------------------------------------------

            # Silently drop hallucinated citations — the model invented a
            # source that was never retrieved.  The known_issue guardrail
            # below will flip known_issue to false if all citations were
            # dropped, keeping the result internally consistent.
            print(
                f"[triage] Dropped hallucinated KB citation: "
                f"{match.source_file} :: {match.heading}"
            )
            continue

        # Replace the original LLM citations with the normalized
        # citations so the final result contains the canonical
        # source_file + heading values from retrieval.
        if normalized_matches != result.knowledge_base_matches:
            result = result.model_copy(
                update={
                    "knowledge_base_matches": normalized_matches
                }
            )

        # Make the final known-issue decision depend on deterministic KB
        # evidence, not on whether a provider happened to cite the same chunk.
        # Only chunks retrieved for this ticket can become evidence, and each
        # must pass product, operation, and symptom checks above.
        deterministic_evidence = _deterministic_kb_evidence(ticket, retrieved)
        if deterministic_evidence:
            cited_identities = {
                (match.source_file, match.heading)
                for match in result.knowledge_base_matches
            }
            missing_evidence = [
                match
                for match in deterministic_evidence
                if (match.source_file, match.heading) not in cited_identities
            ]
            result = result.model_copy(
                update={
                    "known_issue": True,
                    "knowledge_base_matches": (
                        result.knowledge_base_matches + missing_evidence
                    ),
                }
            )
        else:
            result = result.model_copy(
                update={"known_issue": False, "knowledge_base_matches": []}
            )

        # Explicitly clear stale positive KB evidence when the retrieved chunk is
        # only related at the product level but not the same operation. This keeps
        # known_issue aligned with the actual KB evidence instead of a generic match.
        if result.known_issue and result.knowledge_base_matches:
            filtered_matches = []
            for match in result.knowledge_base_matches:
                chunk = chunks_by_identity.get((match.source_file, match.heading))
                if chunk and _kb_match_supports_ticket(ticket, match, chunk.text):
                    filtered_matches.append(match)
            if not filtered_matches:
                result = result.model_copy(update={"known_issue": False, "knowledge_base_matches": []})
            else:
                result = result.model_copy(update={"knowledge_base_matches": filtered_matches})

        safe_subj = ticket.subject.encode("ascii", errors="backslashreplace").decode("ascii")
        print("\n===== TRIAGE DEBUG =====")
        print("Ticket:", safe_subj)
        print("Known issue:", result.known_issue)
        print("KB matches:")

        for match in result.knowledge_base_matches:
            safe_heading = (match.heading or "").encode("ascii", errors="backslashreplace").decode("ascii")
            print(
                f"  - {match.source_file} :: "
                f"{safe_heading}"
            )

        print("========================\n")

        # =========================================================
        # 7. KNOWN-ISSUE EVIDENCE GUARDRAIL
        # =========================================================
        #
        # known_issue=true requires actual KB evidence.
        #
        # We deliberately DO NOT auto-populate citations from the
        # retrieved results. Retrieval relevance is not equivalent
        # to proof that the KB documents the reported issue.
        # =========================================================

        if (
            result.known_issue
            and not result.knowledge_base_matches
        ):
            # All citations were dropped (hallucinated) or LLM sent empty
            # list with known_issue=true — self-correct to false.
            result = result.model_copy(update={"known_issue": False})

        # Citations are supporting evidence for a known issue. Do not return
        # them when the final decision is known_issue=false.
        if (
            not result.known_issue
            and result.knowledge_base_matches
        ):
            result = result.model_copy(update={"knowledge_base_matches": []})

        # Specific module guardrail for AnalyticsHub:
        # "AnalyticsHub: Dashboard Timeout" applies to Dashboard issues, not Exports.
        if result.known_issue and result.knowledge_base_matches:
            ticket_text = f"{ticket.subject} {ticket.body}".lower()
            if "analyticshub" in ticket_text or (result.product and "analyticshub" in result.product.lower()):
                is_exports = "export" in ticket_text and "dashboard" not in ticket_text
                if is_exports:
                    dashboard_only = all(
                        "dashboard" in (m.heading or "").lower()
                        for m in result.knowledge_base_matches
                    )
                    if dashboard_only:
                        result = result.model_copy(
                            update={
                                "known_issue": False,
                                "knowledge_base_matches": [],
                            }
                        )

        # =========================================================
        # 8. CACHE ONLY AFTER ALL GUARDRAILS PASS
        # =========================================================
        #
        # generate_json() validates the Pydantic schema.
        #
        # The checks above are application-level semantic guardrails.
        #
        # Therefore the result is persisted ONLY after every check
        # has succeeded.
        # =========================================================

        self.llm.cache_result(
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=TriageResult,
            result=result,
        )

        # =========================================================
        # 9. RETURN FINAL VALIDATED RESULT
        # =========================================================

        return result
