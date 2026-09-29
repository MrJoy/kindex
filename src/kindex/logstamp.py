"""Timestamp every line a scheduled kin job writes to its log.

launchd and cron redirect a job's stdout/stderr straight into
``cron.log``, ``reminders.log`` and friends, so the log said what happened
but never when. Scheduled jobs set ``KIN_LOG_TIMESTAMPS=1``; ``kin`` then
prefixes each line with the local time its first character was written.
Interactive and piped output stay untouched because nothing sets the
variable there.

Child processes that inherit the file descriptor write around the wrapper.
A child ``kin`` inherits the variable too and stamps its own lines.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable, TextIO

ENV_VAR = "KIN_LOG_TIMESTAMPS"

_FALSEY = frozenset({"", "0", "false", "no", "off"})


def enabled(environ: "os._Environ[str] | dict[str, str] | None" = None) -> bool:
    value = (os.environ if environ is None else environ).get(ENV_VAR, "")
    return value.strip().lower() not in _FALSEY


def _now() -> datetime:
    return datetime.now().astimezone()


class TimestampedStream:
    """Text stream proxy that stamps the start of every line."""

    def __init__(self, stream: TextIO, clock: Callable[[], datetime] = _now):
        self._stream = stream
        self._clock = clock
        self._at_line_start = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        out = []
        for line in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._clock().isoformat(timespec="seconds") + " ")
            out.append(line)
            self._at_line_start = line.endswith("\n")
        self._stream.write("".join(out))
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def install_from_env() -> None:
    """Wrap sys.stdout and sys.stderr when the scheduler asked for stamps."""
    if not enabled():
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, TimestampedStream):
            setattr(sys, name, TimestampedStream(stream))
