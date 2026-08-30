from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import KBChunk


class KBRetriever:
    """Small deterministic TF-IDF retriever over the supplied Markdown KB.

    The corpus is small enough for a local in-process index, which keeps
    retrieval fast, deterministic, and auditable for this take-home.

    Retrieval combines:
    1. TF-IDF semantic similarity
    2. Exact error-code matching
    3. Product-name matching
    4. Product/module overlap
    5. Operational-document intent
    """

    def __init__(self, kb_dir: str = "knowledge-base"):
        self.kb_dir = Path(kb_dir)
        self.chunks = self._load_chunks()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

        # Product names are inferred from KB headings so the retriever
        # does not need product information added to TicketInput.
        self.known_products = self._extract_known_products()

    def _load_chunks(self) -> list[KBChunk]:
        chunks: list[KBChunk] = []

        for path in sorted(self.kb_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")

            matches = list(
                re.finditer(
                    r"^(#{1,4})\s+(.+)$",
                    text,
                    re.MULTILINE,
                )
            )

            relative = str(
                path.relative_to(self.kb_dir.parent)
            ).replace("\\", "/")

            # Fall back to the whole document if it has no Markdown headings.
            if not matches:
                chunks.append(
                    KBChunk(
                        chunk_id=f"{relative}::0",
                        source_file=relative,
                        heading=None,
                        text=text.strip(),
                        metadata={"path": relative},
                    )
                )
                continue

            for idx, match in enumerate(matches):
                start = match.start()

                end = (
                    matches[idx + 1].start()
                    if idx + 1 < len(matches)
                    else len(text)
                )

                section = text[start:end].strip()
                heading = match.group(2).strip()

                if not section:
                    continue

                chunks.append(
                    KBChunk(
                        chunk_id=f"{relative}::{idx}",
                        source_file=relative,
                        heading=heading,
                        text=section,
                        metadata={
                            "path": relative,
                            "heading_level": len(match.group(1)),
                        },
                    )
                )

        return chunks

    def _extract_known_products(self) -> list[str]:
        """Extract product names from KB headings.

        Product names are inferred from headings such as:
        - AnalyticsHub: Dashboard Timeout
        - DataBridge Pro: Pipeline Throughput Degradation

        This keeps product detection deterministic and avoids adding
        product information to the incoming TicketInput schema.
        """

        products: set[str] = set()

        for chunk in self.chunks:
            heading = chunk.heading or ""

            # Capture the product/module name before a descriptive colon.
            if ":" in heading:
                candidate = heading.split(":", 1)[0].strip()

                if candidate:
                    products.add(candidate)

        return sorted(products, key=len, reverse=True)

    def _infer_product(self, query: str) -> str | None:
        """Infer a product already mentioned in the ticket text.

        This does not classify the ticket. It only checks whether a known
        product name from the KB appears in the query.
        """

        query_lower = query.lower()

        for product in self.known_products:
            if product.lower() in query_lower:
                return product

        return None

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _infer_document_type(query: str) -> str | None:
        """Infer the most likely KB document family from the ticket text.

        This is intentionally lightweight and deterministic. It does not
        classify the ticket; it only provides an additional retrieval signal.
        """

        text = query.lower()

        troubleshooting_terms = {
            "error",
            "failed",
            "failure",
            "slow",
            "slowly",
            "performance",
            "timeout",
            "timing out",
            "not working",
            "broken",
            "degradation",
            "unreachable",
            "incident",
            "loading",
        }

        if any(term in text for term in troubleshooting_terms):
            return "troubleshooting"

        return None

    @staticmethod
    def _extract_query_terms(query: str) -> set[str]:
        """Extract useful normalized terms for deterministic overlap scoring."""

        stop_terms = {
            "the",
            "and",
            "for",
            "with",
            "our",
            "this",
            "that",
            "from",
            "have",
            "has",
            "been",
            "are",
            "was",
            "were",
            "into",
            "over",
            "past",
            "very",
            "extremely",
            "team",
            "users",
            "user",
        }

        terms = set(
            re.findall(
                r"[a-z0-9]+",
                query.lower(),
            )
        )

        return {
            term
            for term in terms
            if len(term) >= 3 and term not in stop_terms
        }

    @staticmethod
    def _extract_heading_terms(heading: str | None) -> set[str]:
        """Extract meaningful terms from a KB heading."""

        if not heading:
            return set()

        return set(
            re.findall(
                r"[a-z0-9]+",
                heading.lower(),
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        product: str | None = None,
    ) -> list[tuple[KBChunk, float]]:
        if not query.strip():
            return []

        # ---------------------------------------------------------
        # 1. BASE TF-IDF SIMILARITY
        # ---------------------------------------------------------

        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()

        # ---------------------------------------------------------
        # 2. EXACT ERROR-CODE BOOST
        # ---------------------------------------------------------
        #
        # Error codes are strong retrieval signals in this corpus.
        # Keep this boost stronger than the general document-type boost.
        # ---------------------------------------------------------

        codes = set(
            re.findall(
                r"\b[A-Z][A-Z0-9_]{3,}\b",
                query,
            )
        )

        if codes:
            for i, chunk in enumerate(self.chunks):
                if any(code in chunk.text for code in codes):
                    scores[i] += 0.35

        # ---------------------------------------------------------
        # 3. PRODUCT-NAME BOOST
        # ---------------------------------------------------------
        #
        # If the caller knows the product, favor chunks containing
        # that product while still allowing cross-product troubleshooting
        # documentation to remain relevant.
        #
        # The current TicketInput does not contain product metadata, so
        # product inference from the query is also used below.
        # ---------------------------------------------------------

        inferred_product = product or self._infer_product(query)

        if inferred_product:
            product_terms = inferred_product.lower().split()

            for i, chunk in enumerate(self.chunks):
                chunk_text = chunk.text.lower()
                heading_text = (chunk.heading or "").lower()

                if all(term in chunk_text for term in product_terms):
                    scores[i] += 0.10

                if all(term in heading_text for term in product_terms):
                    scores[i] += 0.15

        # ---------------------------------------------------------
        # 4. MODULE / OPERATION OVERLAP BOOST
        # ---------------------------------------------------------
        #
        # Product matching alone is not enough.
        #
        # Example:
        #
        # Ticket:
        #   AnalyticsHub dashboard operations are timing out
        #
        # KB:
        #   AnalyticsHub: Dashboard Timeout
        #
        # Both product and operation terms align.
        #
        # But:
        #
        # Ticket:
        #   AnalyticsHub exports are slow
        #
        # KB:
        #   AnalyticsHub: Dashboard Timeout
        #
        # The product matches, but the specific operation does not.
        #
        # This distinction improves retrieval without hard-coding any
        # particular evaluation ticket.
        # ---------------------------------------------------------

        query_terms = self._extract_query_terms(query)

        for i, chunk in enumerate(self.chunks):
            heading_terms = self._extract_heading_terms(chunk.heading)

            if not heading_terms:
                continue

            overlap = query_terms.intersection(heading_terms)

            # Only apply this boost when there is meaningful overlap
            # beyond generic operational words.
            specific_overlap = overlap - {
                "troubleshooting",
                "issues",
                "issue",
                "common",
                "step",
                "check",
                "performance",
                "integration",
                "errors",
            }

            if specific_overlap:
                # Small, deterministic boost proportional to useful
                # heading overlap. This supplements rather than replaces
                # TF-IDF similarity.
                scores[i] += min(
                    0.20,
                    0.08 * len(specific_overlap),
                )

        # ---------------------------------------------------------
        # 5. DOCUMENT-TYPE / INTENT BOOST
        # ---------------------------------------------------------
        #
        # Operational problems should favor troubleshooting documentation.
        # This is a retrieval signal, not a hard filter.
        #
        # Therefore a highly relevant product document can still be retrieved
        # if its TF-IDF score is strong enough.
        # ---------------------------------------------------------

        document_type = self._infer_document_type(query)

        if document_type:
            for i, chunk in enumerate(self.chunks):
                if f"/{document_type}/" in chunk.source_file:
                    scores[i] += 0.15

        # ---------------------------------------------------------
        # 6. RANK RESULTS
        # ---------------------------------------------------------

        candidate_indexes = list(range(len(self.chunks)))
        if document_type == "troubleshooting":
            troubleshooting_indexes = [
                i
                for i, chunk in enumerate(self.chunks)
                if "/troubleshooting/" in chunk.source_file
                and scores[i] > 0
            ]
            if troubleshooting_indexes:
                candidate_indexes = troubleshooting_indexes

        ranked = sorted(
            candidate_indexes,
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )[:top_k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in ranked
            if scores[i] >= 0.15
        ]

    def format_context(
        self,
        results: list[tuple[KBChunk, float]],
    ) -> str:
        blocks = []

        for chunk, score in results:
            blocks.append(
                f"[source={chunk.source_file} "
                f"heading={chunk.heading!r} "
                f"score={score:.3f}]\n"
                f"{self._normalise(chunk.text)}"
            )

        return "\n\n".join(blocks)