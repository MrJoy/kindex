"""MCP discovery across an implicit project graph and the home graph."""

import pytest
import sqlite3
import json

pytest.importorskip("mcp")

from kindex.config import Config, ProfileEntry
from kindex.store import Store
from kindex.tasks import create_task


@pytest.fixture
def graphs(tmp_path, monkeypatch):
    import kindex.mcp_server as server

    project = tmp_path / "s2s-framework"
    project.mkdir()
    home_dir = tmp_path / "home-graph"
    local_dir = project / ".kin" / "local" / "kindex"
    cfg = Config(data_dir=str(local_dir))
    cfg._project_path = project
    cfg._global_data_dir = str(home_dir)
    local = Store(cfg)
    home = Store(Config(data_dir=str(home_dir)))
    monkeypatch.setattr(server, "_store", local)
    monkeypatch.setattr(server, "_config", cfg)
    yield server, local, home, project
    local.close()
    home.close()


def test_search_merges_home_hit_with_nonempty_noisy_project(graphs):
    server, local, home, _ = graphs
    local.add_node(title="S2S notes", content="S2S generic unrelated notes")
    home.add_node(title="S2S backlog", content="S2S production backlog next task")

    output = server.search("S2S backlog", top_k=2)

    assert "S2S backlog" in output
    assert "graph=global" in output
    assert "S2S notes" in output
    assert "graph=project" in output
    assert "S2S backlog" in server.search("S2S backlog", top_k=1)


def test_equal_cross_graph_hits_prefer_global(graphs):
    server, local, home, _ = graphs
    local.add_node("Shared topic", content="same phrase")
    home.add_node("Shared topic", content="same phrase")

    output = server.search("Shared topic", top_k=1)

    assert "graph=global" in output
    assert "id=global:" in output


@pytest.mark.parametrize("filtered_secondary", [False, True])
def test_existing_but_noncontributing_secondary_preserves_native_order_and_scores(
    graphs, monkeypatch, filtered_secondary,
):
    import kindex.retrieve as retrieve

    server, local, home, _ = graphs
    semantic_id = local.add_node("Semantic local", tags=["relevant"])
    literal_id = local.add_node("Literal local", tags=["relevant"])
    global_id = home.add_node("Global filtered", tags=["other"])

    def ranked_hybrid(store, query, **kwargs):
        if store.read_only:
            return ([{**store.get_node(global_id), "rrf_score": 0.8}]
                    if filtered_secondary else [])
        return [
            {**store.get_node(semantic_id), "rrf_score": 0.9, "confidence": 0.1},
            {**store.get_node(literal_id), "rrf_score": 0.1, "confidence": 0.9},
        ]

    monkeypatch.setattr(retrieve, "hybrid_search", ranked_hybrid)
    output = server.search("literal local", tags="relevant" if filtered_secondary else "",
                           top_k=2)

    assert output.index("Semantic local") < output.index("Literal local")
    assert "Semantic local (score=0.900" in output
    assert "Literal local (score=0.100" in output
    assert "rank_score=" not in output
    assert "Global filtered" not in output


@pytest.mark.parametrize("first_title", ["Semantic project", "Authoritative project"])
def test_two_graph_merge_preserves_native_project_order(
    graphs, monkeypatch, first_title,
):
    import kindex.retrieve as retrieve

    server, local, home, _ = graphs
    first_id = local.add_node(first_title)
    literal_id = local.add_node("Literal project")
    global_id = home.add_node("Global rank one")

    def ranked_hybrid(store, query, **kwargs):
        if store.read_only:
            return [{**store.get_node(global_id), "rrf_score": 0.2, "confidence": 0.2}]
        return [
            {**store.get_node(first_id), "rrf_score": 0.1, "confidence": 0.1,
             "standing": "authoritative" if first_title.startswith("Authoritative") else "present"},
            {**store.get_node(literal_id), "rrf_score": 0.9, "confidence": 0.9},
        ]

    monkeypatch.setattr(retrieve, "hybrid_search", ranked_hybrid)
    output = server.search("literal project", top_k=3)

    assert output.index("Global rank one") < output.index(first_title)
    assert output.index(first_title) < output.index("Literal project")
    assert "rank_score=0.016393" in output
    assert "score=0.900" not in output


def test_cross_graph_merge_keeps_colliding_raw_ids_distinct(graphs, monkeypatch):
    import kindex.retrieve as retrieve

    server, local, home, _ = graphs
    shared = "abc123def456"
    local.add_node("Project copy", node_id=shared)
    home.add_node("Global copy", node_id=shared)

    def ranked_hybrid(store, query, **kwargs):
        return [{**store.get_node(shared), "rrf_score": 0.5}]

    monkeypatch.setattr(retrieve, "hybrid_search", ranked_hybrid)
    output = server.search("shared", top_k=2)

    assert "Found 2 results" in output
    assert output.index("Global copy") < output.index("Project copy")
    assert f"id={server._graph_ref('global', shared)}" in output
    assert f"id={server._graph_ref('project', shared)}" in output


@pytest.mark.parametrize("single_graph_reason", ["explicit_profile", "missing_global"])
def test_single_graph_search_preserves_hybrid_order_and_scores(
    graphs, monkeypatch, single_graph_reason,
):
    import kindex.retrieve as retrieve

    server, local, home, _ = graphs
    if single_graph_reason == "explicit_profile":
        local.config.active_profile = "work"
    else:
        home.close()
        home.db_path.unlink(missing_ok=True)
    semantic_id = local.add_node("Semantic neighbor", content="related evidence")
    literal_id = local.add_node("Deploy guide", content="deploy guide")

    def ranked_hybrid(store, query, **kwargs):
        return [
            {**store.get_node(semantic_id), "rrf_score": 0.9, "confidence": 0.1},
            {**store.get_node(literal_id), "rrf_score": 0.1, "confidence": 0.9},
        ]

    monkeypatch.setattr(retrieve, "hybrid_search", ranked_hybrid)
    output = server.search("deploy guide", top_k=2)

    assert output.index("Semantic neighbor") < output.index("Deploy guide")
    assert "Semantic neighbor (score=0.900" in output
    assert "Deploy guide (score=0.100" in output


def test_home_search_is_read_only_and_does_not_create_missing_graph(graphs):
    server, _, home, _ = graphs
    node_id = home.add_node(title="Home only", content="distinct home content")
    before = home.conn.execute(
        "SELECT last_accessed FROM nodes WHERE id=?", (node_id,)).fetchone()[0]
    home.conn.execute("UPDATE nodes SET last_accessed='2000-01-01' WHERE id=?", (node_id,))
    home.conn.commit()
    assert "Home only" in server.search("distinct home content")
    after = home.conn.execute(
        "SELECT last_accessed FROM nodes WHERE id=?", (node_id,)).fetchone()[0]
    assert after == "2000-01-01"
    reader = server._global_read_store(server._store, server._config)
    try:
        with pytest.raises(sqlite3.OperationalError):
            reader.conn.execute("UPDATE nodes SET title='wrong' WHERE id=?", (node_id,))
    finally:
        reader.close()
    home.conn.execute("UPDATE nodes SET last_accessed=? WHERE id=?", (before, node_id))
    home.conn.commit()


def test_task_list_discovers_relevant_home_tasks_without_routing_mutations(graphs):
    server, _, home, project = graphs
    task_id = create_task(home, "S2S backlog item", project_path=str(project))
    create_task(home, "Different project task", project_path=str(project.parent / "other"))

    output = server.task_list()

    assert "S2S backlog item" in output
    assert "graph:global" in output
    assert "Different project task" not in output
    ref = server._graph_ref("global", task_id)
    assert ref in output
    assert "Completed:" in server.task_done(ref)
    assert home.get_node(task_id)["extra"]["task_status"] == "done"


def test_explicit_search_and_task_graph_scopes(graphs):
    server, local, home, project = graphs
    local_id = local.add_node("Project scoped result", content="scopeprobe")
    home.add_node("Global scoped result", content="scopeprobe")
    create_task(local, "Project scoped task")
    create_task(home, "Global scoped task", scope="global")
    create_task(home, "Foreign backlog item",
                project_path=str(project.parent / "other"))

    project_search = server.search("scopeprobe", graph="project")
    global_search = server.search("scopeprobe", graph="global")
    assert "Project scoped result" in project_search
    assert "Global scoped result" not in project_search
    local_ref = server._graph_ref("project", local_id)
    assert local_ref in project_search
    assert "Edited Project scoped result" in server.edit(local_ref, append="verified")
    assert "Global scoped result" in global_search
    assert "Project scoped result" not in global_search
    assert "id=global:" in global_search

    project_tasks = server.task_list(graph="project")
    global_tasks = server.task_list(graph="global")
    auto_tasks = server.task_list()
    assert "Project scoped task" in project_tasks
    assert "Global scoped task" not in project_tasks
    assert "Global scoped task" in global_tasks
    assert "Project scoped task" not in global_tasks
    assert "Foreign backlog item" in global_tasks
    assert "Foreign backlog item" not in auto_tasks
    assert "Foreign backlog item" not in server.task_list(
        graph="global", project_path=str(project))
    assert "Global scoped task" not in server.task_list(graph="global", scope="contextual")


def test_listing_limits_and_colliding_node_refs_round_trip(graphs):
    server, local, home, _ = graphs
    shared = "abc123def456"
    local.add_node("Project listing collision", node_id=shared, tags=["listprobe"])
    home.add_node("Global listing collision", node_id=shared, tags=["listprobe"])

    listed = server.list_nodes(tags="listprobe", limit=2)
    assert "2 node(s)" in listed
    assert server._graph_ref("project", shared) in listed
    assert server._graph_ref("global", shared) in listed
    assert "1 node(s)" in server.list_nodes(tags="listprobe", limit=1)
    assert "Project listing collision" in server.list_nodes(
        tags="listprobe", graph="project")
    assert "Global listing collision" not in server.list_nodes(
        tags="listprobe", graph="project")
    ref = server._graph_ref("global", shared)
    assert "Edited Global listing collision" in server.edit(ref, append="checked")
    assert "checked" in home.get_node(shared)["content"]
    assert "checked" not in local.get_node(shared)["content"]


def test_status_graph_diagnostics_and_resources_separate_sources(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Project diagnostic")
    global_id = home.add_node("Global diagnostic")
    global_peer = home.add_node("Global peer")
    home.add_edge(global_id, global_peer)

    status = server.status()
    stats = server.graph_stats()
    heal = server.graph_heal()
    orient = server.orient()
    resource = json.loads(server.resource_status())

    for output in (status, stats, heal, orient):
        assert "## Project graph" in output and "## Global graph" in output
    assert resource["project"]["nodes"] == local.stats()["nodes"]
    assert resource["global"]["nodes"] == home.stats()["nodes"]
    assert "Global diagnostic" in stats
    assert server._graph_ref("global", global_id) in stats
    assert server._graph_ref("global", global_id) in heal
    assert server._graph_ref("project", local_id) in orient
    assert "Global diagnostic" not in stats.split("## Global graph")[0]
    assert "Global diagnostic" not in heal.split("## Global graph")[0]
    assert "Global diagnostic" in server.resource_recent()
    assert server._graph_ref("global", global_id) in server.resource_recent()
    assert server._graph_ref("project", local_id) in server.resource_orphans()


def test_global_watch_suggestion_changelog_and_orphan_refs(graphs):
    server, local, home, _ = graphs
    local_a = local.add_node("Local suggestion origin")
    local_b = local.add_node("Local suggestion target")
    local.add_suggestion(local_a, local_b, identity_kind="node_id")
    outer_id = home.add_node("Outer orphan")
    watch_id = home.add_node("Outer watch", node_type="watch")
    home.add_suggestion(outer_id, watch_id, identity_kind="node_id")

    watches = server.watch_list()
    suggestion = server.suggest()
    changes = server.changelog(since="1970-01-01")
    orphans = server.resource_orphans()

    watch_ref = server._graph_ref("global", watch_id)
    assert watch_ref in watches
    assert f"global #1:" in suggestion
    assert "project #1:" in suggestion
    assert server._graph_ref("global", outer_id) in suggestion
    assert server._graph_ref("project", local_a) in suggestion
    assert "1 pending suggestion(s)" in server.suggest(limit=1)
    assert server._graph_ref("global", outer_id) in changes
    assert server._graph_ref("global", outer_id) in orphans
    assert "Resolved watch" in server.watch_resolve(watch_ref)
    assert home.get_node(watch_id)["status"] == "archived"
    assert local.get_node(watch_id) is None


def test_auto_resources_bound_combined_rows(graphs, monkeypatch):
    server, local, home, _ = graphs
    monkeypatch.setattr(server, "MAX_TOOL_ROWS", 2)
    for i in range(3):
        local.add_node(f"Project orphan {i}")
        home.add_node(f"Global orphan {i}")

    orphan_output = server.resource_orphans()
    listed = server.list_nodes(limit=2)

    assert orphan_output.count("id=") == 2
    assert "more (use graph_heal or list_nodes)" in orphan_output
    assert listed.count("id=") == 2


def test_resource_status_keeps_flat_json_with_missing_secondary(graphs):
    server, local, home, _ = graphs
    local.add_node("Only selected graph")
    home.close()
    home.db_path.unlink(missing_ok=True)

    resource = json.loads(server.resource_status())

    assert resource == local.stats()
    assert "project" not in resource and "global" not in resource
    assert not home.db_path.exists()
    assert server.status().startswith("# Kindex Status")
    assert "Only selected graph" in server.list_nodes()
    for tool in (server.list_nodes, server.status, server.suggest,
                 server.graph_stats, server.graph_heal, server.changelog,
                 server.watch_list, server.orient):
        assert "unavailable or missing" in tool(graph="global")


def test_changelog_discloses_truncation(graphs):
    server, local, _, _ = graphs
    for i in range(55):
        local.add_node(f"Activity canary {i}")

    output = server.changelog(since="1970-01-01", graph="project")

    assert "Latest 50 change(s)" in output
    assert "more may exist" in output


def test_remaining_global_reads_are_read_only(graphs):
    server, _, home, _ = graphs
    node_id = home.add_node("Untouched outer read")
    home.conn.execute("UPDATE nodes SET last_accessed='2000-01-01' WHERE id=?", (node_id,))
    home.conn.commit()
    before_schema = home.get_meta("schema_version")

    outputs = (server.list_nodes(graph="global"), server.status(graph="global"),
               server.graph_stats(graph="global"), server.graph_heal(graph="global"),
               server.changelog(graph="global"), server.resource_recent(),
               server.resource_orphans(), server.resource_status())

    assert all(not output.startswith("Error:") for output in outputs)
    assert home.conn.execute("SELECT last_accessed FROM nodes WHERE id=?",
                             (node_id,)).fetchone()[0] == "2000-01-01"
    assert home.get_meta("schema_version") == before_schema


@pytest.mark.parametrize("tool", ["list_nodes", "status", "suggest", "graph_stats",
                                  "graph_heal", "changelog", "watch_list", "orient"])
def test_remaining_read_tools_reject_invalid_scope(graphs, tool):
    server, _, _, _ = graphs
    assert getattr(server, tool)(graph="invalid") == (
        "Error: graph must be 'auto', 'project', or 'global'")


def test_remaining_read_scopes_bypass_broken_secondary(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Local read surface")
    home.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    home.conn.commit()

    scoped = [server.list_nodes(graph="project"), server.status(graph="project"),
              server.suggest(graph="project"), server.graph_stats(graph="project"),
              server.graph_heal(graph="project"), server.changelog(graph="project"),
              server.watch_list(graph="project"), server.orient(graph="project")]
    assert server._graph_ref("project", local_id) in scoped[0]
    assert all(not output.startswith("Error:") for output in scoped)
    auto = [server.list_nodes(), server.status(), server.suggest(),
            server.graph_stats(), server.graph_heal(), server.changelog(),
            server.watch_list(), server.orient(), server.resource_status(),
            server.resource_recent(), server.resource_orphans()]
    assert all(output.startswith("Error: memory unavailable (SchemaMigrationPending)")
               for output in auto)
    assert home.get_meta("schema_version") == "3"


def test_remaining_read_surfaces_respect_explicit_profile(graphs):
    server, local, home, _ = graphs
    local.config.active_profile = "work"
    local.add_node("Local profile listing")
    home.add_node("Outer profile listing")

    for output in (server.list_nodes(), server.status(), server.graph_stats(),
                   server.graph_heal(), server.orient(), server.resource_recent(),
                   server.resource_orphans()):
        assert "Outer profile listing" not in output
    assert "explicit profile" in server.list_nodes(graph="global")
    assert "explicit profile" in server.status(graph="global")
    assert "explicit profile" in server.graph_stats(graph="global")
    assert "explicit profile" in server.graph_heal(graph="global")
    assert "explicit profile" in server.orient(graph="global")


@pytest.mark.parametrize("tool", ["context", "ask", "prime"])
def test_context_tools_discover_global_and_return_editable_ref(graphs, tool):
    server, local, home, _ = graphs
    node_id = home.add_node("Outer atlas fact", content="outeratlasprobe")
    ref = server._graph_ref("global", node_id)

    output = (server.ask("what is outeratlasprobe") if tool == "ask" else
              getattr(server, tool)(topic="outeratlasprobe"))

    assert "Outer atlas fact" in output
    assert ref in output
    assert "Edited Outer atlas fact" in server.edit(ref, append="confirmed")
    assert "confirmed" in home.get_node(node_id)["content"]
    assert local.get_node(node_id) is None


def test_context_sections_keep_colliding_ids_and_edges_in_own_graph(graphs):
    server, local, home, _ = graphs
    shared = "abc123def456"
    local.add_node("Project collision", content="collidingcontextprobe", node_id=shared)
    home.add_node("Global collision", content="collidingcontextprobe", node_id=shared)
    target = home.add_node("Outer connection", content="connection detail")
    home.add_edge(shared, target)

    output = server.context(topic="collidingcontextprobe", level="full")

    assert output.count("## Project graph") == 1
    assert output.count("## Global graph") == 1
    assert server._graph_ref("project", shared) in output
    assert server._graph_ref("global", shared) in output
    assert f"Outer connection [{server._graph_ref('global', target)}]" in output
    assert "Outer connection" not in output.split("## Global graph")[0]
    assert "Project collision" not in output.split("## Global graph")[1]


def test_empty_topic_context_and_prime_read_both_and_count_sources(graphs):
    server, local, home, _ = graphs
    local.add_node("Recent project canary")
    outer_id = home.add_node("Recent outer canary")

    context = server.context()
    prime = server.prime()

    assert "Recent project canary" in context
    assert "Recent outer canary" in context
    assert server._graph_ref("global", outer_id) in context
    assert "Recent project canary" in prime
    assert "Recent outer canary" in prime
    assert "Project graph:" in prime and "Global graph:" in prime


def test_context_project_scope_avoids_broken_secondary_and_profile_isolates(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Project context safe", content="localsafecontextprobe")
    home.add_node("Outer context", content="localsafecontextprobe")
    home.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    home.conn.commit()

    for tool in (lambda: server.context(topic="localsafecontextprobe", graph="project"),
                 lambda: server.ask("what is localsafecontextprobe", graph="project"),
                 lambda: server.prime(topic="localsafecontextprobe", graph="project")):
        result = tool()
        assert server._graph_ref("project", local_id) in result
        assert "Outer context" not in result
    for tool in (lambda: server.context(topic="localsafecontextprobe"),
                 lambda: server.ask("what is localsafecontextprobe"),
                 lambda: server.prime(topic="localsafecontextprobe")):
        assert tool().startswith("Error: memory unavailable (SchemaMigrationPending)")
    assert home.get_meta("schema_version") == "3"
    local.config.active_profile = "work"
    for tool in (lambda: server.context(topic="localsafecontextprobe"),
                 lambda: server.ask("what is localsafecontextprobe"),
                 lambda: server.prime(topic="localsafecontextprobe")):
        assert "Project context safe" in tool()
    assert "explicit profile" in server.context(graph="global")


def test_context_global_grounding_and_trusted_empty_source_warning(graphs, monkeypatch):
    import kindex.retrieve as retrieve
    from kindex.grounding import RetrievalVerdict, UNGROUNDED

    server, local, home, _ = graphs
    project_id = local.add_node("Verified project", content="trustcontextprobe")
    home.add_node("Unverified outer", content="trustcontextprobe")
    local.verify_node(project_id, verified_by="tester", prov_method="inspection")
    original = retrieve.hybrid_search

    def grounded(store, query, *, grounding=None, **kwargs):
        rows = original(store, query, grounding=grounding, **kwargs)
        if store.read_only:
            grounding["verdict"] = RetrievalVerdict(
                verdict=UNGROUNDED, floor=0.8, best_similarity=0.2)
        return rows

    monkeypatch.setattr(retrieve, "hybrid_search", grounded)
    assert "UNGROUNDED" in server.ask("what is trustcontextprobe")
    output = server.context(topic="trustcontextprobe", trusted_only=True)
    assert "Verified project" in output
    assert "Unverified outer" not in output
    assert "Global graph: (trusted-only omissions: legacy/unverified=1)" in output


def test_context_small_budget_is_split_and_auto_selects_tier(graphs):
    from kindex.retrieve import _estimate_tokens

    server, local, home, _ = graphs
    local.add_node("Project budget", content="budgetcontextprobe " * 100)
    home.add_node("Global budget", content="budgetcontextprobe " * 100)

    output = server.context(topic="budgetcontextprobe", level="full", max_tokens=300)

    assert "Project budget" in output and "Global budget" in output
    assert "**Level:** full" not in output
    assert _estimate_tokens(output) <= 315  # rounding across two sections


def test_context_auxiliary_refs_and_evidence_use_source_identity(graphs, monkeypatch):
    import kindex.kinbase as kinbase

    server, _, home, _ = graphs
    result_id = home.add_node("Outer main fact", content="auxiliarycontextprobe")
    question_id = home.add_node("Outer open question", node_type="question")
    decision_id = home.add_node("Outer recent decision", node_type="decision")
    watch_id = home.add_node("Outer active watch", node_type="watch",
                             extra={"owner": "tester"})
    monkeypatch.setattr(kinbase, "evidence_note",
                        lambda node: "source evidence" if node["id"] == result_id else "")

    output = server.context(topic="auxiliarycontextprobe", graph="global", level="full")

    for node_id in (result_id, question_id, decision_id, watch_id):
        assert server._graph_ref("global", node_id) in output
    assert f"Evidence for Outer main fact [{server._graph_ref('global', result_id)}]:" in output


def test_context_uses_one_evaluation_instant_across_graphs(graphs, monkeypatch):
    import kindex.retrieve as retrieve

    server, local, home, _ = graphs
    local.add_node("Project instant", content="instantcontextprobe")
    home.add_node("Global instant", content="instantcontextprobe")
    instants = []
    original = retrieve.hybrid_search

    def observe(store, query, *, evaluation_time=None, **kwargs):
        instants.append(evaluation_time)
        return original(store, query, evaluation_time=evaluation_time, **kwargs)

    monkeypatch.setattr(retrieve, "hybrid_search", observe)
    monkeypatch.setattr(server, "operation_now", lambda: "2026-09-28T12:00:00Z")

    assert "Project instant" in server.context(topic="instantcontextprobe")
    assert instants == ["2026-09-28T12:00:00Z"] * 2


def test_recent_context_keeps_legacy_local_day_expiry(graphs, monkeypatch):
    import kindex.store as store_module

    server, local, _, _ = graphs
    local.add_node("Local date canary")
    observed = []
    original = store_module.node_expired

    def expiry(node, today=None):
        if node.get("title") == "Local date canary":
            observed.append(today)
        return original(node, today=today)

    monkeypatch.setattr(store_module, "node_expired", expiry)
    monkeypatch.setattr(server, "operation_now", lambda: "2026-09-28T00:00:00Z")

    assert "Local date canary" in server.context(graph="project")
    assert "Local date canary" in server.prime(graph="project")
    assert observed == [None, None]


def test_context_client_filter_and_outer_read_does_not_touch_access_time(graphs, monkeypatch):
    server, _, home, _ = graphs
    allowed = home.add_node("Claude scoped outer", content="clientcontextprobe",
                            tags=["client:claude"])
    home.add_node("Other client outer", content="clientcontextprobe",
                  tags=["client:codex"])
    home.conn.execute("UPDATE nodes SET last_accessed='2000-01-01' WHERE id=?", (allowed,))
    home.conn.commit()
    monkeypatch.setenv("KIN_CLIENT", "claude")

    output = server.context(topic="clientcontextprobe")

    assert "Claude scoped outer" in output
    assert "Other client outer" not in output
    assert home.conn.execute("SELECT last_accessed FROM nodes WHERE id=?", (allowed,)).fetchone()[0] == "2000-01-01"


@pytest.mark.parametrize("tool", ["context", "ask", "prime"])
def test_context_tools_reject_invalid_graph_scope(graphs, tool):
    server, _, _, _ = graphs
    result = (server.ask("anything", graph="wrong") if tool == "ask" else
              getattr(server, tool)(graph="wrong"))
    assert result == "Error: graph must be 'auto', 'project', or 'global'"


@pytest.mark.parametrize("tool", ["search", "task_list"])
def test_invalid_read_graph_scope_is_ordinary_error(graphs, tool):
    server, _, _, _ = graphs

    result = getattr(server, tool)(graph="wrong") if tool == "task_list" else server.search(
        "anything", graph="wrong")

    assert result == "Error: graph must be 'auto', 'project', or 'global'"


def test_qualified_id_routes_edit_and_rejects_cross_graph_link(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node(title="Local evidence", content="project observation")
    home_id = home.add_node(title="Global evidence", content="outer observation")

    ref = server._graph_ref("global", home_id)
    assert "Edited Global evidence" in server.edit(ref, append="verified")
    assert ref in server.show(ref)
    assert "verified" in home.get_node(home_id)["content"]
    assert "verified" not in local.get_node(local_id)["content"]
    assert "cross-graph links" in server.link(local_id, ref)
    assert not local.edges_from(local_id)


def test_duplicate_id_needs_qualified_reference(graphs):
    server, local, home, _ = graphs
    shared = "abcdef123456"
    local.add_node(title="Local version", node_id=shared)
    home.add_node(title="Global version", node_id=shared)

    assert "exists in both graphs" in server.edit(shared, append="unsafe")
    assert "Edited Global version" in server.edit(server._graph_ref("global", shared), append="safe")
    assert "safe" not in local.get_node(shared)["content"]


@pytest.mark.parametrize("name, global_title, global_alias", [
    ("shared title", "SHARED TITLE", None),
    ("shared alias", "Other global title", "SHARED ALIAS"),
])
def test_bare_write_rejects_canonical_cross_graph_name_collision(
    graphs, name, global_title, global_alias,
):
    server, local, home, _ = graphs
    local_id = local.add_node(name, content="local original")
    home.add_node(global_title, aka=[global_alias] if global_alias else None)

    result = server.edit(name, append="wrong graph")

    assert "exists in both graphs" in result
    assert local.get_node(local_id)["content"] == "local original"


def test_global_task_dependencies_round_trip_as_qualified_references(graphs):
    server, local, home, _ = graphs
    dependency = create_task(home, "Global prerequisite")
    task_id = create_task(home, "Global dependent", dependencies=[dependency])
    task_ref = server._graph_ref("global", task_id)
    dependency_ref = server._graph_ref("global", dependency)
    local.add_node("Collision", node_id=dependency)

    fetched = server.task_get(task_ref)
    assert fetched["ok"]
    assert fetched["task"]["dependencies"] == [dependency_ref]
    updated = server.task_update(task_ref, priority=2,
                                 dependencies=fetched["task"]["dependencies"])
    assert updated["ok"]
    assert updated["task"]["dependencies"] == [dependency_ref]
    assert server.task_get(updated["task"]["dependencies"][0])["task"]["id"] == dependency_ref


def test_global_search_surfaces_its_grounding_warning(graphs, monkeypatch):
    import kindex.retrieve as retrieve
    from kindex.grounding import RetrievalVerdict, UNGROUNDED

    server, _, home, _ = graphs
    home.add_node("Global evidence", content="unfamiliar specific query")
    original = retrieve.hybrid_search

    def search_with_grounding(store, query, *, grounding=None, **kwargs):
        results = original(store, query, grounding=grounding, **kwargs)
        if store.read_only:
            assert grounding is not None
            grounding["verdict"] = RetrievalVerdict(
                verdict=UNGROUNDED, floor=0.8, best_similarity=0.2)
        return results

    monkeypatch.setattr(retrieve, "hybrid_search", search_with_grounding)
    output = server.search("unfamiliar specific query")

    assert "Global evidence" in output
    assert "UNGROUNDED" in output
    assert "global" in output


def test_outdated_global_schema_is_typed_and_does_not_mutate_project(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Local evidence", content="original")
    task_id = create_task(local, "Local task")
    home.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    home.conn.commit()

    edit_result = server.edit(local_id, append="unsafe")
    failures = [edit_result, server.search("Local evidence"),
                server.search("Local evidence", graph="global"),
                server.task_list(), server.task_list(graph="global")]

    for result in failures:
        assert result.startswith("Error: memory unavailable (SchemaMigrationPending)")
        assert "kin doctor --fix" in result
    project_search = server.search("Local evidence", graph="project")
    project_tasks = server.task_list(graph="project")
    assert server._graph_ref("project", local_id) in project_search
    assert "Local task" in project_tasks
    assert "graph:project" in project_tasks
    task_ref = server._graph_ref("project", task_id)
    assert task_ref in project_tasks
    assert server.task_get(task_ref)["task"]["title"] == "Local task"
    assert "Edited Local evidence" in server.edit(
        server._graph_ref("project", local_id), append="safe")
    assert "safe" in local.get_node(local_id)["content"]
    assert "unsafe" not in local.get_node(local_id)["content"]
    assert home.get_meta("schema_version") == "3"


def test_stale_qualified_reference_cannot_mutate_new_graph_selection(graphs, tmp_path, monkeypatch):
    server, _, home, _ = graphs
    shared = "abcdef123456"
    home.add_node("Original global", node_id=shared)
    old_ref = server._graph_ref("global", shared)
    old_project_ref = server._graph_ref("project", shared)

    next_project = tmp_path / "next-project"
    next_project.mkdir()
    next_local_dir = next_project / ".kin" / "local" / "kindex"
    next_global_dir = tmp_path / "next-global"
    next_config = Config(data_dir=str(next_local_dir))
    next_config._project_path = next_project
    next_config._global_data_dir = str(next_global_dir)
    next_local = Store(next_config)
    next_global = Store(Config(data_dir=str(next_global_dir)))
    next_local.add_node("Next project", node_id=shared)
    next_global.add_node("Next global", node_id=shared)
    monkeypatch.setattr(server, "_store", next_local)
    monkeypatch.setattr(server, "_config", next_config)
    try:
        assert "Stale graph reference" in server.edit(old_ref, append="wrong")
        assert "Stale graph reference" in server.edit(old_project_ref, append="wrong")
        assert "Stale graph reference" in server.lock_acquire(old_ref)
        assert "Stale graph reference" in server.show(old_project_ref)
        assert "Stale graph reference" in server.add("Wrong derivation", source_refs=old_ref)
        assert next_global.get_node(shared)["content"] == ""
        assert next_local.get_node(shared)["content"] == ""
        assert next_global.get_node_by_title("Wrong derivation") is None
    finally:
        next_local.close()
        next_global.close()


def test_derived_add_can_explicitly_target_global_graph(graphs):
    server, local, home, _ = graphs
    global_id = home.add_node("Global source")
    local_id = local.add_node("Project source")
    output = server.add("Outer graph derived observation",
                        source_refs=f"{server._graph_ref('project', local_id)},{server._graph_ref('global', global_id)}")
    ref = output.split("Created node: ", 1)[1].split(" ", 1)[0]

    assert ref.startswith("global:")
    assert home.get_node(ref.rsplit(":", 1)[1])
    assert local.get_node(ref.rsplit(":", 1)[1]) is None
    denied = server.add("Wrong graph observation", graph="project",
                        source_refs=server._graph_ref("global", global_id))
    assert "Global source requires" in denied
    assert local.get_node_by_title("Wrong graph observation") is None
    missing_ref = server._graph_ref("global", "000000000000")
    assert "unavailable" in server.add("Missing source", source_refs=missing_ref)
    assert home.get_node_by_title("Missing source") is None


def test_derived_task_and_qualified_update_stay_global(graphs):
    server, local, home, project = graphs
    source = home.add_node("Global backlog source")
    output = server.task_add("Follow global backlog", project_path=str(project),
                             source_refs=server._graph_ref("global", source))
    task_ref = output.split("Created task: ", 1)[1].split(" ", 1)[0]

    assert task_ref.startswith("global:")
    updated = server.task_update(task_ref, priority=1)
    assert updated["ok"]
    assert updated["task"]["id"] == task_ref
    assert home.get_node(task_ref.rsplit(":", 1)[1])["extra"]["priority"] == 1
    assert local.get_node(task_ref.rsplit(":", 1)[1]) is None
    local_source = local.add_node("Local source")
    denied = server.task_add("Impossible cross-store task",
                             link_to=server._graph_ref("project", local_source),
                             source_refs=server._graph_ref("global", source))
    assert "cross-graph links" in denied
    assert home.get_node_by_title("Impossible cross-store task") is None


def test_global_derived_contextual_task_is_visible_to_bound_project(graphs):
    server, local, home, project = graphs
    source = home.add_node("Outer contextual source")
    create_task(home, "Other project's task", project_path=str(project.parent / "other"))

    result = server.task_add(
        "Derived task for selected project",
        source_refs=server._graph_ref("global", source))
    task_ref = result.split("Created task: ", 1)[1].split(" ", 1)[0]
    task_id = task_ref.rsplit(":", 1)[1]

    assert task_ref.startswith("global:")
    assert home.get_node(task_id)["extra"]["project_path"] == str(project)
    assert local.get_node(task_id) is None
    listed = server.task_list()
    assert task_ref in listed
    assert "Other project's task" not in listed
    assert server.task_get(task_ref)["task"]["id"] == task_ref
    updated = server.task_update(task_ref, priority=1)
    assert updated["ok"] and updated["task"]["id"] == task_ref
    assert home.get_node(task_id)["extra"]["priority"] == 1
    assert local.edges_from(task_id) == []


def test_global_contextual_task_uses_bound_project_after_cwd_env_changes(
    graphs, tmp_path, monkeypatch,
):
    server, _, home, project = graphs
    source = home.add_node("Bound project source")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv("PWD", str(unrelated))
    monkeypatch.setenv("KIN_PROJECT", str(unrelated))
    monkeypatch.setenv("KIN_PROJECT_PATH", str(unrelated))

    result = server.task_add(
        "Task retains selected project",
        link_to=server._graph_ref("global", source))

    assert "Created task: global:" in result
    assert home.get_node_by_title("Task retains selected project")["extra"]["project_path"] == str(project)
    assert "Task retains selected project" in server.task_list()


def test_global_task_explicit_path_and_global_scope_are_not_inferred(graphs):
    server, _, home, project = graphs
    source = home.add_node("Scope source")
    other = project.parent / "other"
    other.mkdir()

    explicit = server.task_add(
        "Explicit other project", project_path=str(other),
        source_refs=server._graph_ref("global", source))
    global_task = server.task_add(
        "Globally scoped derived task", scope="global",
        source_refs=server._graph_ref("global", source))

    assert "Created task: global:" in explicit
    assert "Created task: global:" in global_task
    assert home.get_node_by_title("Explicit other project")["extra"]["project_path"] == str(other)
    assert "project_path" not in home.get_node_by_title("Globally scoped derived task")["extra"]
    listed = server.task_list()
    assert "Explicit other project" not in listed
    assert "Globally scoped derived task" in listed
    assert "Explicit other project" in server.task_list(graph="global")


def test_explicit_profile_task_creation_keeps_single_store_scope(graphs):
    server, local, home, _ = graphs
    local.config.active_profile = "work"

    result = server.task_add("Profile-local task")

    task = local.get_node_by_title("Profile-local task")
    assert f"Created task: {task['id']} " in result
    assert "project_path" not in task["extra"]
    assert home.get_node_by_title("Profile-local task") is None


def test_global_watch_resolves_in_its_source_graph(graphs):
    server, local, home, _ = graphs
    watch_id = home.add_node("Global watch", node_type="watch")

    assert "Resolved watch" in server.watch_resolve(server._graph_ref("global", watch_id))
    assert home.get_node(watch_id)["status"] == "archived"
    assert local.get_node(watch_id) is None


def test_global_lock_response_round_trips_to_unlock(graphs):
    server, local, home, _ = graphs
    node_id = home.add_node("Global lock target")
    source_ref = server._graph_ref("global", node_id)

    acquired = server.lock_acquire(source_ref)
    returned_ref = acquired.split("(", 1)[1].split(")", 1)[0]

    assert returned_ref == source_ref
    assert f"Unlocked Global lock target ({source_ref})" == server.lock_release(returned_ref)
    assert local.get_node(node_id) is None


def test_project_task_response_dependencies_round_trip_with_id_collision(graphs):
    server, local, home, _ = graphs
    dependency = create_task(local, "Project dependency")
    task_id = create_task(local, "Project dependent", dependencies=[dependency])
    home.add_node("Global collision", node_id=dependency)
    home.add_node("Another global collision", node_id=task_id)
    task_ref = server._graph_ref("project", task_id)
    dependency_ref = server._graph_ref("project", dependency)

    fetched = server.task_get(task_ref)
    assert fetched["task"]["id"] == task_ref
    assert fetched["task"]["dependencies"] == [dependency_ref]
    updated = server.task_update(fetched["task"]["id"], priority=2,
                                 dependencies=fetched["task"]["dependencies"])
    assert updated["ok"]
    assert updated["task"]["id"] == task_ref
    assert updated["task"]["dependencies"] == [dependency_ref]
    assert server.task_get(dependency_ref)["task"]["title"] == "Project dependency"


def test_task_update_rejects_bare_colliding_dependency_without_mutation(graphs):
    server, local, home, _ = graphs
    dependency_id = create_task(local, "Project dependency")
    task_id = create_task(local, "Project dependent")
    home.add_node("Global dependency collision", node_id=dependency_id)
    task_ref = server._graph_ref("project", task_id)
    dependency_ref = server._graph_ref("project", dependency_id)
    before = server.task_get(task_ref)["task"]

    rejected = server.task_update(task_ref, dependencies=[dependency_id])

    assert rejected["ok"] is False
    assert "exists in both graphs" in rejected["error"]["message"]
    unchanged = server.task_get(task_ref)["task"]
    assert unchanged["version"] == before["version"]
    assert unchanged["dependencies"] == before["dependencies"]

    accepted = server.task_update(task_ref, dependencies=[dependency_ref])
    assert accepted["ok"]
    assert accepted["task"]["dependencies"] == [dependency_ref]


def test_node_detail_and_resource_preserve_global_connection_refs(graphs):
    server, local, home, _ = graphs
    node_id = home.add_node("Global detail", content="User content has raw id 123")
    neighbor_id = home.add_node("Global neighbor")
    home.add_edge(node_id, neighbor_id)
    local.add_node("Local collision", node_id=neighbor_id)
    source_ref = server._graph_ref("global", node_id)
    neighbor_ref = server._graph_ref("global", neighbor_id)

    for detail in (server.show(source_ref), server.resource_node(source_ref)):
        assert f"ID: {source_ref}" in detail
        assert f"id={neighbor_ref}" in detail
        assert "User content has raw id 123" in detail


def test_new_project_responses_are_qualified_for_followup_mutations(graphs):
    server, local, home, _ = graphs
    created = server.add("New project concept")
    node_ref = created.split("Created node: ", 1)[1].split(" ", 1)[0]
    assert node_ref.startswith("project:")
    assert "Edited New project concept" in server.edit(node_ref, append="verified")

    task = server.task_add("New project task")
    task_ref = task.split("Created task: ", 1)[1].split(" ", 1)[0]
    assert task_ref.startswith("project:")
    assert server.task_get(task_ref)["task"]["id"] == task_ref

    watch = server.watch_add("New project watch")
    watch_ref = watch.split("id=", 1)[1].split(")", 1)[0]
    assert watch_ref.startswith("project:")
    assert watch_ref in server.watch_resolve(watch_ref)
    assert home.get_node_by_title("New project concept") is None
    assert local.get_node_by_title("New project concept") is not None


def test_project_response_formatting_does_not_open_secondary(graphs, monkeypatch):
    server, local, _, _ = graphs
    node_id = local.add_node("Project detail")
    neighbor_id = local.add_node("Project neighbor")
    local.add_edge(node_id, neighbor_id)
    task_id = create_task(local, "Project task detail", dependencies=[])
    source_ref = server._graph_ref("project", node_id)
    neighbor_ref = server._graph_ref("project", neighbor_id)
    task_ref = server._graph_ref("project", task_id)

    def forbidden_secondary(*args, **kwargs):
        raise AssertionError("response formatting opened secondary graph")

    monkeypatch.setattr(server, "_global_read_store", forbidden_secondary)
    for detail in (server.show(source_ref), server.resource_node(source_ref)):
        assert f"ID: {source_ref}" in detail
        assert f"id={neighbor_ref}" in detail
    assert server.task_get(task_ref)["task"]["id"] == task_ref
    assert source_ref in server.lock_acquire(source_ref)
    assert source_ref in server.lock_release(source_ref)


def test_project_list_responses_expose_qualified_ids(graphs):
    server, local, _, _ = graphs
    node_id = local.add_node("Listed project node")
    watch_id = local.add_node("Listed project watch", node_type="watch",
                              extra={"watch_status": "active"})

    assert server._graph_ref("project", node_id) in server.list_nodes()
    assert server._graph_ref("project", watch_id) in server.watch_list()


def test_explicit_profile_preserves_incoming_project_qualification(graphs):
    server, local, _, _ = graphs
    node_id = local.add_node("Profiled project node")
    local.config.active_profile = "work"
    source_ref = server._graph_ref("project", node_id)

    assert f"ID: {source_ref}" in server.show(source_ref)
    assert source_ref in server.lock_acquire(source_ref)
    assert source_ref in server.lock_release(source_ref)
    assert "Created node: project:" not in server.add("Profile source-free node")


def test_watch_add_links_qualified_global_result_only_in_global_graph(graphs):
    server, local, home, _ = graphs
    source_id = home.add_node("Global watch context")
    source_ref = server._graph_ref("global", source_id)

    result = server.watch_add("Monitor global context", owner="platform",
                              link_to=source_ref)

    assert "Watch created" in result
    watch_ref = result.split("id=", 1)[1].split(",", 1)[0].split(")", 1)[0]
    assert watch_ref.startswith("global:")
    watch_id = watch_ref.rsplit(":", 1)[1]
    assert home.get_node(watch_id)["extra"]["owner"] == "platform"
    assert local.get_node(watch_id) is None
    assert any(edge["to_id"] == source_id for edge in home.edges_from(watch_id))
    assert not local.edges_from(watch_id)


@pytest.mark.parametrize("bad_ref, expected", [
    ("missing", "unavailable"),
    ("stale", "Stale graph reference"),
    ("cross_store", "cross-graph links"),
    ("ambiguous", "title_collision"),
])
def test_watch_add_rejects_bad_links_before_creating_node(graphs, bad_ref, expected):
    server, local, home, _ = graphs
    global_id = home.add_node("Global watch source")
    local_id = local.add_node("Local watch source")
    if bad_ref == "missing":
        link_to = f"{server._graph_ref('global', global_id)},missing target"
    elif bad_ref == "stale":
        link_to = f"{server._graph_ref('global', global_id)},global:oldscope:{global_id}"
    elif bad_ref == "cross_store":
        link_to = (f"{server._graph_ref('global', global_id)},"
                   f"{server._graph_ref('project', local_id)}")
    else:
        home.add_node("Duplicate target")
        home.add_node("Duplicate target")
        link_to = f"{server._graph_ref('global', global_id)},Duplicate target"

    result = server.watch_add("Rejected watch", link_to=link_to)

    assert expected in result
    assert local.get_node_by_title("Rejected watch") is None
    assert home.get_node_by_title("Rejected watch") is None


def test_watch_add_rejects_bare_cross_graph_collision(graphs):
    server, local, home, _ = graphs
    local.add_node("Shared target")
    home.add_node("Shared target")

    assert "exists in both graphs" in server.watch_add(
        "Ambiguous watch", link_to="Shared target")
    assert local.get_node_by_title("Ambiguous watch") is None
    assert home.get_node_by_title("Ambiguous watch") is None


def test_watch_add_preserves_local_and_source_free_selection(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Local watch source")

    linked = server.watch_add("Local linked watch", link_to=local_id)
    server.watch_add("Local source-free watch")

    linked_id = linked.split("id=", 1)[1].split(")", 1)[0]
    assert linked_id.startswith("project:")
    raw_id = linked_id.rsplit(":", 1)[1]
    assert local.get_node(raw_id)["type"] == "watch"
    assert any(edge["to_id"] == local_id for edge in local.edges_from(raw_id))
    assert local.get_node_by_title("Local source-free watch") is not None
    assert home.get_node_by_title("Local linked watch") is None
    assert home.get_node_by_title("Local source-free watch") is None


def test_watch_add_project_qualified_link_does_not_open_outdated_secondary(graphs):
    server, local, home, _ = graphs
    local_id = local.add_node("Project-qualified watch source")
    home.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    home.conn.commit()

    result = server.watch_add(
        "Project watch despite outdated secondary",
        link_to=server._graph_ref("project", local_id))

    assert "Watch created" in result
    watch = local.get_node_by_title("Project watch despite outdated secondary")
    assert watch is not None
    assert any(edge["to_id"] == local_id for edge in local.edges_from(watch["id"]))
    assert home.get_meta("schema_version") == "3"


def test_watch_add_global_source_ref_routes_without_link(graphs):
    server, local, home, _ = graphs
    source_id = home.add_node("Global source for watch")

    result = server.watch_add("Derived global watch",
                              source_refs=server._graph_ref("global", source_id))

    assert "id=global:" in result
    assert home.get_node_by_title("Derived global watch") is not None
    assert local.get_node_by_title("Derived global watch") is None


def test_watch_add_explicit_profile_cannot_cross_to_global(graphs):
    server, local, home, _ = graphs
    global_id = home.add_node("Global target")
    local.config.active_profile = "work"

    result = server.watch_add("Profile-isolated watch",
                              link_to=server._graph_ref("global", global_id))

    assert "global graph is unavailable" in result.lower()
    assert local.get_node_by_title("Profile-isolated watch") is None
    assert home.get_node_by_title("Profile-isolated watch") is None


def test_global_task_add_links_only_to_qualified_global_target(graphs):
    server, local, home, _ = graphs
    shared_id = "abc123def456"
    local.add_node("Project collision", node_id=shared_id)
    home.add_node("Global target", node_id=shared_id)

    result = server.task_add("Global linked task",
                             link_to=server._graph_ref("global", shared_id))

    assert "Created task: global:" in result
    task = home.get_node_by_title("Global linked task")
    assert task is not None
    assert any(edge["to_id"] == shared_id for edge in home.edges_from(task["id"]))
    assert local.get_node_by_title("Global linked task") is None


def test_task_add_accepts_unique_bare_local_alias(graphs):
    server, local, home, _ = graphs
    target_id = local.add_node("Canonical local target", aka=["Local alias"])

    result = server.task_add("Bare alias linked task", link_to="Local alias")

    assert "Created task" in result
    task = local.get_node_by_title("Bare alias linked task")
    assert any(edge["to_id"] == target_id for edge in local.edges_from(task["id"]))
    assert home.get_node_by_title("Bare alias linked task") is None


@pytest.mark.parametrize("bad_ref, expected", [
    ("missing", "unavailable"),
    ("stale", "Stale graph reference"),
    ("mixed", "cross-graph links"),
    ("duplicate_title", "title_collision"),
    ("alias_collision", "exists in both graphs"),
])
def test_task_add_rejects_all_invalid_links_before_creation(graphs, bad_ref, expected):
    server, local, home, _ = graphs
    project_id = local.add_node("Project link")
    global_id = home.add_node("Global link")
    if bad_ref == "missing":
        refs = f"{server._graph_ref('project', project_id)},missing target"
    elif bad_ref == "stale":
        refs = f"{server._graph_ref('project', project_id)},project:oldscope:{project_id}"
    elif bad_ref == "mixed":
        refs = (f"{server._graph_ref('project', project_id)},"
                f"{server._graph_ref('global', global_id)}")
    elif bad_ref == "duplicate_title":
        local.add_node("Duplicate title")
        local.add_node("Duplicate title")
        refs = f"{server._graph_ref('project', project_id)},Duplicate title"
    else:
        local.add_node("Local alias owner", aka=["Shared alias"])
        home.add_node("Global alias owner", aka=["SHARED ALIAS"])
        refs = f"{server._graph_ref('project', project_id)},Shared alias"

    result = server.task_add("Rejected linked task", link_to=refs)

    assert expected in result
    assert local.get_node_by_title("Rejected linked task") is None
    assert home.get_node_by_title("Rejected linked task") is None


@pytest.mark.parametrize("tool", ["add", "task_add", "watch_add", "learn"])
@pytest.mark.parametrize("bad_source, expected", [
    ("bare", "must be a graph-qualified result ID"),
    ("empty", "must be graph-qualified result IDs"),
    ("stale", "Stale graph reference"),
    ("missing", "unavailable"),
])
def test_invalid_source_refs_are_rejected_before_creation(
    graphs, tool, bad_source, expected,
):
    server, local, home, _ = graphs
    source_id = local.add_node("Bare provenance")
    source_ref = {
        "bare": "Bare provenance",
        "empty": ",",
        "stale": f"project:oldscope:{source_id}",
        "missing": server._graph_ref("project", "000000000000"),
    }[bad_source]

    result = getattr(server, tool)("Rejected provenance", source_refs=source_ref)

    assert expected in result
    assert local.get_node_by_title("Rejected provenance") is None
    assert home.get_node_by_title("Rejected provenance") is None


def test_selected_store_mode_and_coord_accept_qualified_project_refs(graphs):
    server, local, home, _ = graphs
    target_id = local.add_node("Mode context")
    task_id = create_task(local, "Coordination task")

    mode = server.mode_create("project-mode", "primer", "boundary", "permissions",
                              link_to=server._graph_ref("project", target_id))
    coord = server.coord_start("Project room", task_id=server._graph_ref("project", task_id))

    assert "Created mode" in mode
    mode_node = local.get_node_by_title("mode:project-mode")
    assert any(edge["to_id"] == target_id for edge in local.edges_from(mode_node["id"]))
    assert "Started coordination conversation" in coord
    from kindex.coordination import get_conversation
    room = get_conversation(local, "project-room")
    assert room["extra"]["task_id"] == task_id
    assert any(edge["to_id"] == task_id for edge in local.edges_from(room["id"]))
    assert home.get_node_by_title("mode:project-mode") is None


def test_coord_start_rejects_non_task_before_creating_conversation(graphs):
    server, local, _, _ = graphs
    concept_id = local.add_node("Not a task")

    result = server.coord_start(
        "Invalid task room", task_id=server._graph_ref("project", concept_id))

    assert "is not a task" in result
    from kindex.coordination import get_conversation
    assert get_conversation(local, "invalid-task-room") is None


@pytest.mark.parametrize("kind, bad_ref, expected", [
    ("mode", "global", "cross-graph links"),
    ("mode", "stale", "Stale graph reference"),
    ("mode", "missing", "unavailable"),
    ("mode", "ambiguous", "title_collision"),
    ("coord", "global", "cross-graph links"),
    ("coord", "stale", "Stale graph reference"),
    ("coord", "missing", "unavailable"),
    ("coord", "ambiguous", "title_collision"),
])
def test_selected_store_facilities_reject_refs_before_creation(
    graphs, kind, bad_ref, expected,
):
    server, local, home, _ = graphs
    local_id = (create_task(local, "Valid project task") if kind == "coord"
                else local.add_node("Valid project context"))
    global_id = home.add_node("Global context")
    if bad_ref == "global":
        ref = server._graph_ref("global", global_id)
    elif bad_ref == "stale":
        ref = f"project:oldscope:{local_id}"
    elif bad_ref == "missing":
        missing = server._graph_ref("project", "000000000000")
        ref = (f"{server._graph_ref('project', local_id)},{missing}"
               if kind == "mode" else missing)
    else:
        if kind == "coord":
            create_task(local, "Duplicate task")
            create_task(local, "Duplicate task")
            ref = "Duplicate task"
        else:
            local.add_node("Duplicate context")
            local.add_node("Duplicate context")
            ref = "Duplicate context"
    if kind == "mode":
        result = server.mode_create("rejected-mode", "primer", "boundary",
                                    "permissions", link_to=ref)
        assert local.get_node_by_title("mode:rejected-mode") is None
    else:
        result = server.coord_start("Rejected room", task_id=ref)
        from kindex.coordination import get_conversation
        assert get_conversation(local, "rejected-room") is None
    assert expected in result


def test_qualified_project_links_do_not_open_secondary_for_writes(graphs, monkeypatch):
    server, local, _, _ = graphs
    node_id = local.add_node("Local context")
    task_id = create_task(local, "Local task")

    def forbidden_secondary(*args, **kwargs):
        raise AssertionError("qualified project reference opened secondary")

    monkeypatch.setattr(server, "_global_read_store", forbidden_secondary)
    assert "Created task" in server.task_add(
        "Qualified local task", link_to=server._graph_ref("project", node_id))
    assert "Created mode" in server.mode_create(
        "qualified-mode", "primer", "boundary", "permissions",
        link_to=server._graph_ref("project", node_id))
    assert "Started coordination" in server.coord_start(
        "Qualified room", task_id=server._graph_ref("project", task_id))


def test_explicit_profile_keeps_search_isolated(graphs):
    server, local, home, _ = graphs
    local.config.active_profile = "work"
    home.add_node(title="Home only", content="distinct home content")
    create_task(home, "Home-only task", scope="global")

    assert "Home only" not in server.search("distinct home content")
    assert "Home only" not in server.search("distinct home content", graph="project")
    assert "Home-only task" not in server.task_list()
    assert "explicit profile" in server.search("distinct home content", graph="global")
    assert "explicit profile" in server.task_list(graph="global")


def test_configured_global_profile_target_keeps_its_stamp(graphs):
    server, local, home, _ = graphs
    home.add_node("Profiled global result", content="profile target")
    home.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('kin_profile', 'outer')")
    home.conn.commit()
    local.config.profiles["outer"] = ProfileEntry(data_dir=str(home.config.data_path))

    assert "graph=global" in server.search("profile target")


def test_global_write_does_not_stamp_unstamped_profile_target(graphs):
    server, local, home, _ = graphs
    home.add_node("Unstamped seed")
    local.config.profiles["outer"] = ProfileEntry(data_dir=str(home.config.data_path))

    result = server.add("Explicit outer capture", graph="global")

    assert "Created node: global:" in result
    assert home.get_node_by_title("Explicit outer capture") is not None
    assert home.get_meta("kin_profile") is None


def test_no_home_database_is_not_created(graphs):
    server, local, home, _ = graphs
    home.close()
    home.db_path.unlink(missing_ok=True)
    assert "No results" in server.search("missing topic")
    assert "unavailable or missing" in server.search("missing topic", graph="global")
    assert "unavailable or missing" in server.task_list(graph="global")
    assert "No tasks" in server.task_list(graph="project")
    assert not home.db_path.exists()


def test_ambiguous_outer_profile_is_typed_and_project_scope_still_works(graphs):
    server, local, home, _ = graphs
    local.add_node("Local profile-safe result", content="profileambigprobe")
    local.add_node("Bare collision")
    home.add_node("Bare collision")
    local.config.profiles["one"] = ProfileEntry(data_dir=str(home.config.data_path))
    local.config.profiles["two"] = ProfileEntry(data_dir=str(home.config.data_path))

    for result in (server.search("profileambigprobe"), server.task_list(),
                   server.search("profileambigprobe", graph="global"),
                   server.task_list(graph="global")):
        assert result.startswith("Error: memory unavailable (ValueError)")
        assert "matches multiple profiles" in result
    assert "Local profile-safe result" in server.search(
        "profileambigprobe", graph="project")
    assert server.edit("Bare collision", append="unsafe").startswith(
        "Error: memory unavailable (ValueError)")
    assert local.get_node_by_title("Bare collision")["content"] == ""
    assert home.get_meta("schema_version") is not None


def test_secondary_constructor_path_failure_preserves_cause_and_cleans_up(
    graphs, monkeypatch,
):
    import kindex.store as store_module

    server, local, _, _ = graphs
    local.add_node("Local constructor-safe result", content="constructorprobe")
    closed = []

    class BrokenSecondary:
        @property
        def db_path(self):
            raise OSError("secondary path failed")

        def close(self):
            closed.append(True)
            raise RuntimeError("secondary close failed")

    monkeypatch.setattr(store_module, "Store", lambda *args, **kwargs: BrokenSecondary())
    result = server.search("constructorprobe")

    assert result.startswith("Error: memory unavailable (OSError)")
    assert closed == [True]
    assert "Local constructor-safe result" in server.search(
        "constructorprobe", graph="project")


def test_secondary_store_constructor_failure_is_typed(graphs, monkeypatch):
    import kindex.store as store_module

    server, local, _, _ = graphs
    local.add_node("Local constructor result", content="constructorfailprobe")

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("secondary constructor failed")

    monkeypatch.setattr(store_module, "Store", fail_constructor)
    assert server.search("constructorfailprobe").startswith(
        "Error: memory unavailable (RuntimeError)")
    assert "Local constructor result" in server.search(
        "constructorfailprobe", graph="project")


def test_secondary_config_constructor_failure_is_typed(graphs, monkeypatch):
    import kindex.config as config_module

    server, local, _, _ = graphs
    local.add_node("Local config result", content="configfailprobe")

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("secondary config failed")

    monkeypatch.setattr(config_module, "Config", fail_constructor)
    assert server.task_list().startswith("Error: memory unavailable (RuntimeError)")
    assert "Local config result" in server.search(
        "configfailprobe", graph="project")


def test_secondary_teardown_does_not_mask_query_failure(graphs, monkeypatch):
    import kindex.retrieve as retrieve

    server, _, _, _ = graphs
    closed = []

    class BrokenSecondary:
        def close(self):
            closed.append(True)
            raise RuntimeError("close failed")

    outer = BrokenSecondary()
    monkeypatch.setattr(server, "_global_read_store", lambda *args: outer)

    def failing_search(store, *args, **kwargs):
        if store is outer:
            raise sqlite3.OperationalError("query failed")
        return []

    monkeypatch.setattr(retrieve, "hybrid_search", failing_search)
    result = server.search("anything")

    assert result == "Error: memory unavailable (OperationalError)"
    assert closed == [True]


def test_load_config_uses_nondefault_global_data_dir(tmp_path, monkeypatch):
    from kindex.config import load_config

    project = tmp_path / "project"
    project.mkdir()
    local_dir = project / ".kin" / "local" / "kindex"
    global_dir = tmp_path / "configured-outer-graph"
    global_config = tmp_path / "kin.yaml"
    global_config.write_text(f"data_dir: {global_dir}\n")
    monkeypatch.setattr("kindex.config._GLOBAL_PATHS", [global_config])
    monkeypatch.setattr("kindex.config._git_root", lambda _: project)
    monkeypatch.delenv("KIN_PROFILE", raising=False)
    local = Store(Config(data_dir=str(local_dir)))
    local.add_node("Project seed")
    local.close()
    cfg = load_config(project_path=project)

    assert cfg.data_path == local_dir
    assert cfg._global_data_dir == str(global_dir)


def test_project_edit_policy_does_not_govern_configured_global_graph(tmp_path, monkeypatch):
    from kindex.config import load_config
    import kindex.mcp_server as server

    project = tmp_path / "project"
    local_dir = project / ".kin" / "local" / "kindex"
    local_dir.mkdir(parents=True)
    (project / ".kin" / "config").write_text(
        "edit_policy:\n  decision: editable\n")
    global_dir = tmp_path / "configured-global"
    global_config = tmp_path / "kin.yaml"
    global_config.write_text(f"data_dir: {global_dir}\n")
    monkeypatch.setattr("kindex.config._GLOBAL_PATHS", [global_config])
    monkeypatch.setattr("kindex.config._git_root", lambda _: project)
    monkeypatch.delenv("KIN_PROFILE", raising=False)

    local = Store(Config(data_dir=str(local_dir)))
    local.add_node("Project seed")
    home = Store(Config(data_dir=str(global_dir)))
    decision = home.add_node("Immutable global decision", node_type="decision",
                             content="original")
    cfg = load_config(project_path=project)
    assert cfg.edit_policy["decision"] == "editable"
    selected = Store(cfg)
    monkeypatch.setattr(server, "_store", selected)
    monkeypatch.setattr(server, "_config", cfg)
    try:
        result = server.edit(server._graph_ref("global", decision), content="overwritten")
        assert "additive" in result
        assert home.get_node(decision)["content"] == "original"
    finally:
        selected.close()
        local.close()
        home.close()
