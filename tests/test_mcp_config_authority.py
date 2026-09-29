"""Real config loading and MCP routing across repo-local and user graphs."""

import re
import subprocess

import pytest

pytest.importorskip("mcp")

from kindex.config import Config, load_config
from kindex.store import Store


@pytest.fixture
def configured_repo(tmp_path, monkeypatch):
    import kindex.config as config_module
    import kindex.mcp_server as server

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        "edit_policy:\n  decision: editable\n  constraint: editable\n")
    local_dir = repo / ".kin" / "local" / "kindex"
    user_dir = tmp_path / "user-outer-graph"
    user_yaml = tmp_path / "user-kin.yaml"
    user_yaml.write_text(
        f"data_dir: {user_dir}\n"
        "edit_policy:\n  decision: additive\n  document: additive\n"
        f"profiles:\n  home:\n    data_dir: {user_dir}\n")
    monkeypatch.setattr(config_module, "_GLOBAL_PATHS", [user_yaml])
    for key in ("KIN_PROFILE", "KIN_PROJECT", "KIN_PROJECT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(repo)
    local = Store(Config(data_dir=str(local_dir)))
    local.add_node("Existing local presence")
    local.close()
    outer = Store(Config(data_dir=str(user_dir)))
    outer.add_node("Existing outer presence")
    outer.close()
    server._reset_singletons()
    yield server, repo, local_dir, user_dir
    server._reset_singletons()


def _search_ref(output: str, source: str) -> str:
    match = re.search(rf"id=({source}:[0-9a-f]+:[0-9a-f]+)", output)
    assert match, output
    return match.group(1)


def test_presence_selected_capture_search_link_and_global_derivation(configured_repo):
    server, repo, local_dir, user_dir = configured_repo
    selected, cfg = server._get_store()
    assert cfg.data_path == local_dir
    assert cfg.active_profile is None
    assert cfg.edit_policy["decision"] == "editable"
    assert cfg.edit_policy["document"] == "additive"

    first = server.add("First local capture")
    second = server.add("Second local capture")
    first_ref = _search_ref(server.search("First local capture"), "project")
    second_ref = _search_ref(server.search("Second local capture"), "project")
    assert first_ref in first and second_ref in second
    assert "Linked:" in server.link(first_ref, second_ref)
    assert selected.get_node(first_ref.rsplit(":", 1)[1]) is not None
    assert selected.edges_from(first_ref.rsplit(":", 1)[1])

    outer_ref = _search_ref(server.search("Existing outer presence"), "global")
    derived = server.add("Derived outer capture", source_refs=outer_ref)
    derived_ref = derived.split("Created node: ", 1)[1].split(" ", 1)[0]
    assert derived_ref.startswith("global:")
    assert "Linked:" in server.link(derived_ref, outer_ref)
    outer = Store(Config(data_dir=str(user_dir)))
    try:
        assert outer.get_node(derived_ref.rsplit(":", 1)[1]) is not None
        assert outer.edges_from(derived_ref.rsplit(":", 1)[1])
    finally:
        outer.close()
    assert selected.get_node(derived_ref.rsplit(":", 1)[1]) is None

    denied = server.task_add("Mixed-source task must not exist",
                             link_to=f"{first_ref},{outer_ref}")
    assert "cross-graph" in denied
    assert selected.get_node_by_title("Mixed-source task must not exist") is None
    outer = Store(Config(data_dir=str(user_dir)))
    try:
        assert outer.get_node_by_title("Mixed-source task must not exist") is None
    finally:
        outer.close()


def test_profile_primary_uses_user_policy_and_local_retains_repo_policy(
    configured_repo, monkeypatch,
):
    server, _, local_dir, user_dir = configured_repo
    local_decision = server.add("Editable local decision", node_type="decision")
    local_constraint = server.add("Editable local constraint", node_type="constraint")
    local_decision_ref = local_decision.split("Created node: ", 1)[1].split(" ", 1)[0]
    local_constraint_ref = local_constraint.split("Created node: ", 1)[1].split(" ", 1)[0]
    assert "Edited" in server.edit(local_decision_ref, content="local rewrite")
    assert "Edited" in server.edit(local_constraint_ref, content="local constraint rewrite")

    monkeypatch.setenv("KIN_PROFILE", "home")
    server._reset_singletons()
    selected, cfg = server._get_store()
    assert cfg.active_profile == "home"
    assert cfg.data_path == user_dir
    assert cfg.edit_policy == {"decision": "additive", "document": "additive"}
    assert "edit_policy" in cfg._ignored_project_keys
    assert selected.config.data_path == user_dir
    decision = server.add("Protected profile decision", node_type="decision")
    constraint = server.add("Protected profile constraint", node_type="constraint")
    decision_ref = decision.split("Created node: ", 1)[1].split(" ", 1)[0]
    constraint_ref = constraint.split("Created node: ", 1)[1].split(" ", 1)[0]
    assert "additive" in server.edit(decision_ref, content="forbidden")
    assert "additive" in server.edit(constraint_ref, content="forbidden")
    assert selected.get_node(decision_ref)["content"] == "Protected profile decision"
    assert selected.get_node(constraint_ref)["content"] == "Protected profile constraint"
    assert local_dir != user_dir


def test_external_declared_store_cannot_apply_repo_edit_policy(tmp_path, monkeypatch):
    import kindex.config as config_module

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".kin").mkdir()
    external = tmp_path / "external-primary"
    (repo / ".kin" / "config").write_text(
        f"data_dir: {external}\nedit_policy:\n  decision: editable\n")
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("edit_policy:\n  document: additive\n")
    monkeypatch.setattr(config_module, "_GLOBAL_PATHS", [user_yaml])
    for key in ("KIN_PROFILE", "KIN_PROJECT", "KIN_PROJECT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(repo)

    cfg = load_config()

    assert cfg.data_path == external
    assert cfg.edit_policy == {"document": "additive"}
    assert "edit_policy" in cfg._ignored_project_keys


def test_declared_repo_local_custom_store_keeps_its_policy(tmp_path, monkeypatch):
    import kindex.config as config_module

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        "data_dir: .private-kindex\nedit_policy:\n  decision: editable\n")
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("edit_policy:\n  document: additive\n")
    monkeypatch.setattr(config_module, "_GLOBAL_PATHS", [user_yaml])
    for key in ("KIN_PROFILE", "KIN_PROJECT", "KIN_PROJECT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(repo)

    cfg = load_config()

    assert cfg.data_path == repo / ".private-kindex"
    assert cfg.edit_policy == {"decision": "editable", "document": "additive"}
    assert "edit_policy" not in cfg._ignored_project_keys


def test_user_global_path_inside_repo_is_not_project_policy_authority(
    tmp_path, monkeypatch,
):
    import kindex.config as config_module

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / ".kin").mkdir()
    (repo / ".kin" / "config").write_text(
        "edit_policy:\n  decision: editable\n")
    user_dir = repo / "user-owned-global"
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(f"data_dir: {user_dir}\n")
    monkeypatch.setattr(config_module, "_GLOBAL_PATHS", [user_yaml])
    for key in ("KIN_PROFILE", "KIN_PROJECT", "KIN_PROJECT_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(repo)

    cfg = load_config()

    assert cfg.data_path == user_dir
    assert cfg.edit_policy == {}
    assert "edit_policy" in cfg._ignored_project_keys


def test_bound_root_declared_global_path_cannot_acquire_repo_policy(tmp_path):
    from kindex.config import bound_root

    root = tmp_path / "bound"
    (root / ".kin").mkdir(parents=True)
    (root / ".config" / "kindex").mkdir(parents=True)
    (root / ".kin" / "config").write_text(
        "data_dir: outer-global\nedit_policy:\n  decision: editable\n")
    (root / ".config" / "kindex" / "kin.yaml").write_text(
        "data_dir: /outer-global\nedit_policy:\n  document: additive\n")

    with bound_root(root):
        cfg = load_config()

    assert cfg.data_path == root / "outer-global"
    assert cfg.edit_policy == {"document": "additive"}
    assert "edit_policy" in cfg._ignored_project_keys


def test_explicit_config_path_keeps_authorized_policy_bypass(configured_repo):
    _, repo, _, _ = configured_repo

    cfg = load_config(config_path=repo / ".kin" / "config")

    assert cfg.edit_policy == {"decision": "editable", "constraint": "editable"}
