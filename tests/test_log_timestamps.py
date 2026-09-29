"""Scheduler log lines carry the time they were written."""

import io
import re
import sys
from datetime import datetime, timedelta, timezone

import pytest

from kindex import logstamp

STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d ")


def fixed_clock(*times):
    it = iter(times)
    return lambda: next(it)


T0 = datetime(2026, 9, 29, 14, 36, 2, tzinfo=timezone(timedelta(hours=-7)))
T1 = T0 + timedelta(seconds=5)


def test_each_line_gets_the_time_its_first_character_was_written():
    raw = io.StringIO()
    stream = logstamp.TimestampedStream(raw, clock=fixed_clock(T0, T1))
    stream.write("Checked [hoo3]: ")
    stream.write("0 fired\nChecked [personal]")
    stream.write(": 0 fired\n")
    assert raw.getvalue() == (
        "2026-09-29T14:36:02-07:00 Checked [hoo3]: 0 fired\n"
        "2026-09-29T14:36:07-07:00 Checked [personal]: 0 fired\n"
    )


def test_blank_lines_and_multiline_writes_are_each_stamped():
    raw = io.StringIO()
    stream = logstamp.TimestampedStream(raw, clock=fixed_clock(T0, T0, T0))
    stream.write("a\n\nb\n")
    assert raw.getvalue().splitlines() == [
        "2026-09-29T14:36:02-07:00 a",
        "2026-09-29T14:36:02-07:00 ",
        "2026-09-29T14:36:02-07:00 b",
    ]


def test_empty_write_emits_nothing():
    raw = io.StringIO()
    stream = logstamp.TimestampedStream(raw, clock=fixed_clock())
    assert stream.write("") == 0
    assert raw.getvalue() == ""


def test_write_returns_the_callers_length_and_delegates_other_attributes():
    raw = io.StringIO()
    stream = logstamp.TimestampedStream(raw, clock=fixed_clock(T0))
    assert stream.write("hello\n") == len("hello\n")
    assert stream.getvalue() == raw.getvalue()
    stream.writelines([])
    stream.flush()


def test_install_is_a_noop_without_the_env_var(monkeypatch):
    monkeypatch.delenv(logstamp.ENV_VAR, raising=False)
    out, err = sys.stdout, sys.stderr
    logstamp.install_from_env()
    assert sys.stdout is out and sys.stderr is err


def test_install_wraps_both_streams_once(monkeypatch):
    monkeypatch.setenv(logstamp.ENV_VAR, "1")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    logstamp.install_from_env()
    logstamp.install_from_env()
    assert isinstance(sys.stdout, logstamp.TimestampedStream)
    assert not isinstance(sys.stdout._stream, logstamp.TimestampedStream)
    print("Checked [hoo3]: 0 fired, 0 auto-snoozed")
    print("Warning: VOYAGE_API_KEY not set.", file=sys.stderr)
    assert STAMP.match(sys.stdout._stream.getvalue())
    assert STAMP.match(sys.stderr._stream.getvalue())


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_install_ignores_falsey_values(monkeypatch, value):
    monkeypatch.setenv(logstamp.ENV_VAR, value)
    out = sys.stdout
    logstamp.install_from_env()
    assert sys.stdout is out


def test_cli_main_stamps_output_when_scheduled(monkeypatch, capsys):
    from kindex import cli
    monkeypatch.setenv(logstamp.ENV_VAR, "1")
    monkeypatch.setattr(sys, "argv", ["kin", "--version"])
    cli.main()
    out = capsys.readouterr().out
    assert STAMP.match(out)
    assert out.rstrip().endswith("(Kindex)")


def test_cli_main_leaves_interactive_output_alone(monkeypatch, capsys):
    from kindex import cli
    monkeypatch.delenv(logstamp.ENV_VAR, raising=False)
    monkeypatch.setattr(sys, "argv", ["kin", "--version"])
    cli.main()
    assert capsys.readouterr().out.startswith("kin ")


def test_scheduled_jobs_opt_in(monkeypatch, tmp_path):
    from kindex import setup as ksetup
    from kindex.config import Config

    config = Config(data_dir=str(tmp_path), claude_dir=str(tmp_path / "claude"),
                    project_dirs=[str(tmp_path / "projects")])
    written = {}

    def fake_run(cmd, **kw):
        if cmd[:2] == ["crontab", "-l"]:
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        written["crontab"] = kw.get("input", "")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("kindex.setup.subprocess.run", fake_run)
    monkeypatch.setattr(ksetup, "_find_kin_path", lambda: "/usr/local/bin/kin")
    monkeypatch.setattr(ksetup, "scheduler_path", lambda: "/usr/bin")
    ksetup.install_crontab(config)
    lines = [line for line in written["crontab"].splitlines() if line]
    assert len(lines) == 2
    assert all(f" {logstamp.ENV_VAR}=1 " in line for line in lines)

    assert logstamp.ENV_VAR in ksetup.scheduler_environment()


def test_launchd_jobs_opt_in(monkeypatch, tmp_path):
    from kindex import setup as ksetup

    plist = ksetup._launchd_plist(label="x", program_args=["kin", "cron"], interval=60,
                                  stdout_path="/o", stderr_path="/e",
                                  environment=ksetup.scheduler_environment())
    assert f"<key>{logstamp.ENV_VAR}</key>\n        <string>1</string>" in plist


def test_detached_dream_stamps_dream_log(monkeypatch, tmp_path):
    from kindex.config import Config
    from kindex.dream import detach_dream

    seen = {}

    class FakeProc:
        pid = 1

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.delenv(logstamp.ENV_VAR, raising=False)
    monkeypatch.setattr("kindex.setup._find_kin_path", lambda: "/tmp/kin")
    monkeypatch.setattr("kindex.dream.subprocess.Popen", fake_popen)
    detach_dream(Config(data_dir=str(tmp_path)), mode="lightweight", force=True)
    assert seen["env"][logstamp.ENV_VAR] == "1"
    assert seen["env"]["PATH"]


def test_children_do_not_inherit_the_opt_in(monkeypatch):
    # A scheduled action launches agents whose kin hooks answer in JSON on a
    # captured stdout. Only the process writing to the log may stamp.
    import os
    import subprocess

    monkeypatch.setenv(logstamp.ENV_VAR, "1")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    logstamp.install_from_env()
    assert logstamp.ENV_VAR not in os.environ
    child = subprocess.run([sys.executable, "-m", "kindex.cli", "--version"],
                           capture_output=True, text=True, timeout=60)
    assert child.stdout.startswith("kin ")
