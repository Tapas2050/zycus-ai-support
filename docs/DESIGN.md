# Design Note

## 1. Production failure modes, detection, and mitigation

**Failure mode 1 — Incorrect triage or hallucinated KB guidance.**  
An LLM can confidently assign the wrong category/priority or cite a document that does not support the issue. I mitigate this by separating retrieval from generation, passing only retrieved KB context to the model, using a controlled category/urgency vocabulary, validating the response with Pydantic, and rejecting KB citations that were not actually retrieved. The final known-issue decision is deterministic: the cited or retrieved chunk must match the ticket's product, operation, and operational symptom. The evaluation harness checks core classification fields, actual KB-chunk relevance, and case-specific first-response guidance. A content-addressed cache stores successful structured results for the exact model/prompt/schema/input combination after agent-level guardrails pass, so repeated evaluation of the same case can avoid a provider call. In production I would monitor disagreement rates, low-confidence cases, and human overrides.

**Failure mode 2 — Incorrect account/ticket linkage.**  
Customer data can contain stale or inconsistent identifiers. This dataset demonstrates that risk: ticket account IDs are frequently inconsistent. The system first attempts an ID join and then validates the ticket company before falling back to the supplied company field. The selected join strategy is included in the output, so the behavior is observable. In production I would add a canonical customer-data service and reconciliation metrics rather than silently guessing.

**Failure mode 3 — Unsupported churn/escalation claims.**  
A summariser could infer churn from weak evidence or invent a quotation. The design separates account-level signals from ticket-level risks. Every ticket risk must contain an exact substring from the source ticket; the application rejects non-verbatim evidence. In production I would additionally retain source ticket IDs and audit the evidence used for every high-severity flag.

## 2. Latency vs quality

The concrete trade-off is local deterministic retrieval plus one LLM call rather than sending the complete knowledge base to the model. Retrieval reduces prompt size and keeps irrelevant documents out of context, improving both latency and answer quality. I use a small TF-IDF index because the supplied corpus is only 106 heading-level chunks; introducing a remote vector database would add network latency and operational complexity without meaningful benefit for this assignment. If latency became the hard constraint, I would cache the KB index, cache repeated ticket classifications, reduce retrieved context to the top 2–3 chunks, use a smaller model, and move expensive semantic evaluation out of the synchronous request path.

## 3. Data sensitivity

The supplied dataset is synthetic, but the architecture assumes real support tickets may contain PII. The application does not scrape external sources or enrich records with outside customer information. Only the minimum ticket/account context required for the task is sent to the LLM adapter. API credentials are read from environment variables and excluded through `.gitignore`; the repository does not include an `.env.example` template. For a real deployment, I would add PII detection/redaction where policy requires it, use an approved enterprise LLM endpoint with appropriate retention controls, encrypt traffic, restrict logs, and ensure prompts/responses containing customer data are not written to ordinary application logs.

## 4. Scaling

With 10× the ticket volume, the first pressure point is not the JSON repository but repeated LLM calls and retrieval/index rebuilds. The local TF-IDF index remains inexpensive at this scale, but loading and re-indexing data on every process start should eventually become a build-time or service-level artifact. I would also batch/index tickets separately, cache account histories, and paginate or pre-aggregate ticket metrics. For higher throughput, the API layer can remain stateless while retrieval and LLM calls scale horizontally. The next bottleneck would be model latency and rate limits, followed by evaluation cost. Production would therefore add request queues for non-interactive work, caching, observability, rate limiting, and asynchronous evaluation.
