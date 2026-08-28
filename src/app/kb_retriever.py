from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import KBChunk


class KBRetriever:
    """Small deterministic TF-IDF retriever over the supplied Markdown KB.

    The corpus is small enough for a local in-process index, which keeps
    retrieval fast, deterministic, and auditable for this take-home.
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
        # ---------------------------------------------------------

        if product:
            product_terms = product.lower().split()

            for i, chunk in enumerate(self.chunks):
                chunk_text = chunk.text.lower()

                if all(term in chunk_text for term in product_terms):
                    scores[i] += 0.10

        # ---------------------------------------------------------
        # 4. DOCUMENT-TYPE / INTENT BOOST
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
        # 5. RANK RESULTS
        # ---------------------------------------------------------

        ranked = scores.argsort()[::-1][:top_k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0
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