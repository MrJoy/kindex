"""Tests for hybrid retrieval."""

import pytest

from kindex.config import Config
from kindex.retrieve import federate_graph_results, format_context_block, hybrid_search
from kindex.store import Store


@pytest.fixture
def populated_store(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    s = Store(cfg)
    s.add_node("Stigmergy", content="Coordination through environmental traces",
               node_id="stig", domains=["systems", "coordination"], weight=1.0)
    s.add_node("Emergence Architecture", content="Stigmergic task coordination",
               node_id="emerge", domains=["systems", "engineering"], weight=0.9)
    s.add_node("Patent Filing", content="ASD mesh patent for organizational health",
               node_id="patent", domains=["ip", "research"], weight=1.0)
    s.add_node("Database Design", content="Schema normalization and indexes",
               node_id="db", domains=["engineering"], weight=0.5)

    s.add_edge("stig", "emerge", weight=0.9, provenance="same principles")
    s.add_edge("patent", "stig", weight=1.0, provenance="ASD uses stigmergy")
    s.add_edge("patent", "emerge", weight=0.8, provenance="both coordination")

    yield s
    s.close()


class TestHybridSearch:
    def test_finds_by_keyword(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy")
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "stig" in ids

    def test_graph_expansion(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", expand_graph=True)
        ids = [r["id"] for r in results]
        # Should find emerge via graph edge from stig
        assert "emerge" in ids or "patent" in ids

    def test_no_results(self, populated_store):
        results = hybrid_search(populated_store, "zzzz nothing")
        assert results == []


class TestContextBlock:
    def test_format(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=3)
        block = format_context_block(populated_store, results, query="stigmergy")
        assert "Kindex" in block
        assert "Stigmergy" in block
        assert "Active tags:" in block

    def test_empty(self, populated_store):
        block = format_context_block(populated_store, [], query="nothing")
        assert "No relevant context" in block


def test_graph_federation_preserves_sole_global_order_and_limits_results():
    global_hits = [
        {"id": "semantic", "rrf_score": 0.9},
        {"id": "literal", "rrf_score": 0.1},
    ]

    results = federate_graph_results([], global_hits, top_k=1)

    assert results == [{**global_hits[0], "_graph_source": "global"}]
    assert "_merge_rank_score" not in results[0]


def test_graph_federation_uses_graph_local_identity_and_equal_rank_weight():
    project_hits = [{"id": "shared", "confidence": 0.99}]
    global_hits = [{"id": "shared", "confidence": 0.01}]

    results = federate_graph_results(project_hits, global_hits, top_k=2)

    assert [(row["_graph_source"], row["id"]) for row in results] == [
        ("global", "shared"), ("project", "shared")]
    assert results[0]["_merge_rank_score"] == results[1]["_merge_rank_score"] == 1 / 61
