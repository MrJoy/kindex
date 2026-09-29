"""Tests for five-tier context retrieval system."""

import pytest
from types import SimpleNamespace

from kindex.config import Config
from kindex.retrieve import (
    TIER_BUDGETS,
    TIER_ORDER,
    auto_select_tier,
    format_context_block,
    hybrid_search,
)
from kindex.store import Store


@pytest.fixture
def populated_store(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    s = Store(cfg)
    s.add_node("Stigmergy", content="Coordination through environmental traces",
               node_id="stig", domains=["systems", "coordination"], weight=1.0)
    s.add_node("Emergence Architecture", content="Stigmergic task coordination for distributed systems",
               node_id="emerge", domains=["systems", "engineering"], weight=0.9)
    s.add_node("Patent Filing", content="ASD mesh patent for organizational health monitoring",
               node_id="patent", domains=["ip", "research"], weight=1.0)
    s.add_node("Database Design", content="Schema normalization and indexes for graph storage",
               node_id="db", domains=["engineering"], weight=0.5)

    s.add_edge("stig", "emerge", weight=0.9, provenance="same principles")
    s.add_edge("patent", "stig", weight=1.0, provenance="ASD uses stigmergy")
    s.add_edge("patent", "emerge", weight=0.8, provenance="both coordination")

    yield s
    s.close()


class TestOperationalScoping:
    def test_format_context_block_scopes_operational_by_adapter(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        store = Store(cfg)
        store.add_node("Stigmergy", content="Coordination through environmental traces",
                       node_id="stig", domains=["systems"], weight=1.0)
        # An Antigravity-scoped constraint — an operational node that _append_operational
        # always renders in the full/abridged tiers, regardless of the search results.
        store.add_node(
            "Antigravity hook must return decision JSON",
            node_type="constraint",
            node_id="ag-con",
            content="Antigravity PreToolUse stdout must be decision JSON.",
            domains=["antigravity"],
            extra={"action": "warn"},
        )
        results = hybrid_search(store, "stigmergy", top_k=5)
        marker = "Antigravity hook must return decision JSON"

        # No adapter: every operational node surfaces (human-facing kin context).
        assert marker in format_context_block(store, results, query="stigmergy", level="full")
        # Claude / Codex: the Antigravity-scoped constraint is dropped.
        assert marker not in format_context_block(
            store, results, query="stigmergy", level="full", adapter="claude")
        assert marker not in format_context_block(
            store, results, query="stigmergy", level="abridged", adapter="codex")
        # Antigravity itself still sees it.
        assert marker in format_context_block(
            store, results, query="stigmergy", level="full", adapter="antigravity")
        store.close()


    def test_mcp_scope_results_filters_foreign_client(self):
        from kindex.mcp_server import _scope_results
        rows = [{"id": "a", "tags": ["antigravity"]}, {"id": "b", "tags": ["systems"]}]
        # A client-scoped search hit is dropped for a foreign client...
        assert [r["id"] for r in _scope_results(rows, "claude")] == ["b"]
        # ...kept for its own client, and unfiltered when no client is set.
        assert [r["id"] for r in _scope_results(rows, "antigravity")] == ["a", "b"]
        assert [r["id"] for r in _scope_results(rows, None)] == ["a", "b"]


def test_tiny_budget_preserves_grounding_warning_without_graph_payload(populated_store):
    from kindex.retrieve import GRAPH_DATA_NOTE

    node = populated_store.get_node("stig")
    block = format_context_block(
        populated_store, [node], max_tokens_approx=8,
        grounding={"verdict": SimpleNamespace(note=lambda: "UNGROUNDED warning")},
    )

    assert "UNGROUNDED warning" in block
    assert GRAPH_DATA_NOTE in block
    assert "Coordination through environmental traces" not in block
    assert len(block) < 200


class TestAutoSelectTier:
    def test_default_is_abridged(self):
        assert auto_select_tier(None) == "abridged"

    def test_large_budget_gets_full(self):
        assert auto_select_tier(5000) == "full"

    def test_medium_budget_gets_abridged(self):
        assert auto_select_tier(1500) == "abridged"

    def test_small_budget_gets_summarized(self):
        assert auto_select_tier(750) == "summarized"

    def test_tiny_budget_gets_executive(self):
        assert auto_select_tier(200) == "executive"

    def test_minimal_budget_gets_index(self):
        assert auto_select_tier(50) == "index"


class TestFullTier:
    def test_includes_provenance(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="full")
        assert "full" in block.lower()
        assert "Stigmergy" in block
        assert "Key concepts" in block

    def test_includes_edges(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="full")
        assert "Connects" in block or "Emergence" in block


class TestAbridgedTier:
    def test_shorter_than_full(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        full = format_context_block(populated_store, results, query="stigmergy", level="full")
        abridged = format_context_block(populated_store, results, query="stigmergy", level="abridged")
        assert len(abridged) < len(full)

    def test_preserves_structure(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="abridged")
        assert "Active tags:" in block
        assert "Key concepts" in block


class TestSummarizedTier:
    def test_paragraph_form(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="summarized")
        assert "Tags:" in block
        # Should be shorter than abridged
        abridged = format_context_block(populated_store, results, query="stigmergy", level="abridged")
        assert len(block) <= len(abridged)


class TestExecutiveTier:
    def test_very_short(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="executive")
        assert len(block) < 1000
        assert "Kindex" in block

    def test_contains_key_terms(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="executive")
        assert "Stigmergy" in block or "stigmergy" in block


class TestIndexTier:
    def test_minimal_output(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="index")
        assert "Kindex index:" in block
        # Should be the shortest tier
        exec_block = format_context_block(populated_store, results, query="stigmergy", level="executive")
        assert len(block) <= len(exec_block)

    def test_no_content(self, populated_store):
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        block = format_context_block(populated_store, results, query="stigmergy", level="index")
        # Index should not contain full content
        assert "environmental traces" not in block


class TestEmptyResults:
    def test_all_tiers_handle_empty(self, populated_store):
        for tier in TIER_ORDER:
            block = format_context_block(populated_store, [], query="nothing", level=tier)
            assert "No relevant context" in block


class TestTierMonotonicity:
    def test_tiers_decrease_in_size(self, populated_store):
        """Each tier should be equal to or smaller than the previous one."""
        results = hybrid_search(populated_store, "stigmergy", top_k=5)
        sizes = {}
        for tier in TIER_ORDER:
            block = format_context_block(populated_store, results, query="stigmergy", level=tier)
            sizes[tier] = len(block)

        # Full >= Abridged >= Summarized >= Executive >= Index
        for i in range(len(TIER_ORDER) - 1):
            assert sizes[TIER_ORDER[i]] >= sizes[TIER_ORDER[i + 1]], \
                f"{TIER_ORDER[i]} ({sizes[TIER_ORDER[i]]}) should be >= {TIER_ORDER[i+1]} ({sizes[TIER_ORDER[i+1]]})"
