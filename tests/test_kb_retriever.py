import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from app.kb_retriever import KBRetriever


def test_kb_chunk_count():
    kb = KBRetriever("knowledge-base")
    assert len(kb.chunks) == 106


def test_specific_workflow_query_retrieves_duplicate_execution_section():
    kb = KBRetriever("knowledge-base")
    results = kb.retrieve(
        "WorkflowEngine webhook duplicate executions idempotency key",
        top_k=3,
        product="WorkflowEngine",
    )
    assert results
    assert any(
        "Duplicate workflow executions" in chunk.heading
        for chunk, _ in results
        if chunk.heading
    )


def test_analytics_dashboard_timeout_retrieval():
    kb = KBRetriever("knowledge-base")
    results = kb.retrieve(
        "AnalyticsHub dashboard timeout slow loading",
        top_k=3,
        product="AnalyticsHub",
    )
    assert results
    assert any(
        "Dashboard Timeout" in (chunk.heading or "")
        for chunk, _ in results
    )
