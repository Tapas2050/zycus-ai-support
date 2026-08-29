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


class KBMatch(BaseModel):
    source_file: str
    heading: str | None = None
    relevance_reason: str


class TriageResult(BaseModel):
    prompt_version: str = TRIAGE_PROMPT_VERSION
    product: str | None = None
    product_area: str
    category: str
    urgency: str
    rationale: str
    known_issue: bool
    knowledge_base_matches: list[KBMatch] = Field(default_factory=list)
    recommended_responder_team: str
    first_response: str


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

        if result.category not in allowed_categories:
            raise ValueError(
                f"Invalid category returned by LLM: "
                f"{result.category}"
            )

        # Urgency must belong to the controlled vocabulary.
        if result.urgency not in {
            "P1",
            "P2",
            "P3",
            "P4",
        }:
            raise ValueError(
                f"Invalid urgency returned by LLM: "
                f"{result.urgency}"
            )

        # Responder team must belong to the controlled vocabulary.
        if result.recommended_responder_team not in RESPONDER_TEAMS:
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
                normalized_matches.append(
                    match.model_copy(
                        update={
                            "heading": normalized_heading,
                        }
                    )
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

                normalized_matches.append(
                    match.model_copy(
                        update={
                            "heading": chunk.heading,
                        }
                    )
                )
                continue

            # -----------------------------------------------------
            # Case C: Unsafe or invented citation
            # -----------------------------------------------------

            raise ValueError(
                "LLM referenced a KB source that was not retrieved: "
                f"{match.source_file} :: {match.heading}"
            )

        # Replace the original LLM citations with the normalized
        # citations so the final result contains the canonical
        # source_file + heading values from retrieval.
        if normalized_matches != result.knowledge_base_matches:
            result = result.model_copy(
                update={
                    "knowledge_base_matches": normalized_matches
                }
            )
        print("\n===== TRIAGE DEBUG =====")
        print("Ticket:", ticket.subject)
        print("Known issue:", result.known_issue)
        print("KB matches:")

        for match in result.knowledge_base_matches:
            print(
                f"  - {match.source_file} :: "
                f"{match.heading}"
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
            raise ValueError(
                "LLM marked known_issue=true but provided "
                "no KB matches."
            )

        # Conversely, known_issue=false must not contain KB
        # citations.
        if (
            not result.known_issue
            and result.knowledge_base_matches
        ):
            raise ValueError(
                "LLM marked known_issue=false but provided "
                "KB matches."
            )

        if result.known_issue and result.knowledge_base_matches:
            ticket_text = f"{ticket.subject} {ticket.body}".lower()
            operation_terms = {
                "dashboard",
                "exports",
                "export",
                "data sources",
                "login",
                "sso",
                "sync",
                "api",
                "report",
                "query",
            }
            specific_ticket_ops = {
                term for term in operation_terms if term in ticket_text
            }

            if specific_ticket_ops:
                mismatched = False
                for match in result.knowledge_base_matches:
                    kb_heading = (match.heading or "").lower()
                    kb_source = match.source_file.lower()
                    if any(
                        term in kb_heading or term in kb_source
                        for term in specific_ticket_ops
                    ):
                        continue
                    mismatched = True
                    break

                if mismatched:
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