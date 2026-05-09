"""Logging + tracing infrastructure for the harness.

Two outputs by default:
  1. Pretty console output via rich.
  2. JSONL per-event trace file at results/traces/run-<ts>.jsonl, suitable for
     post-run analysis (every chat call, retry, cache hit, error is one line).

Trace IDs are propagated through `contextvars` so async fan-out remains traceable.
A correlation ID can also be set per logical group of calls (e.g. one prompt's
full bot+judge round-trip) using `with_correlation(...)`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from rich.logging import RichHandler


_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def short_path(p: Path | str) -> str:
    """Return a project-relative or ~-relative path so absolute filesystem paths
    do not leak into log lines, traces, or shared output."""
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        pass
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return p.name


def trace_id() -> str:
    return _trace_id.get()


def correlation_id() -> str:
    return _correlation_id.get()


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:8]
    _trace_id.set(tid)
    return tid


@contextlib.contextmanager
def with_correlation(label: str | None = None) -> Iterator[str]:
    cid = (label or "") + "-" + uuid.uuid4().hex[:6] if label else uuid.uuid4().hex[:8]
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        record.correlation_id = _correlation_id.get()
        return True


class _JsonlHandler(logging.Handler):
    """Emits one JSON object per log record to a file; safe across async tasks."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            obj: dict[str, Any] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "trace_id": getattr(record, "trace_id", "-"),
                "correlation_id": getattr(record, "correlation_id", "-"),
            }
            extra = getattr(record, "event", None)
            if isinstance(extra, dict):
                obj["event"] = extra
            if record.exc_info:
                obj["exc"] = self.format(record).split("\n", 1)[-1]
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


_configured = False


def configure(level: str | None = None, jsonl_path: Path | None = None) -> Path:
    """Configure logging once. Returns the path of the JSONL trace file."""
    global _configured, _jsonl_path
    if _configured:
        return _jsonl_path

    level_name = (level or os.environ.get("PFB_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    if jsonl_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        jsonl_path = (
            Path(__file__).resolve().parents[2] / "results" / "traces" / f"run-{ts}.jsonl"
        )

    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = RichHandler(
        level=log_level,
        rich_tracebacks=True,
        show_path=False,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    console.addFilter(_ContextFilter())
    console.setFormatter(logging.Formatter("[trace=%(trace_id)s] %(message)s"))
    root.addHandler(console)

    jsonl = _JsonlHandler(jsonl_path)
    jsonl.addFilter(_ContextFilter())
    jsonl.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(jsonl)

    # Quieter for noisy libs.
    for noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _jsonl_path = jsonl_path
    _configured = True

    logging.getLogger("pfb").info(
        "logging configured",
        extra={"event": {"level": level_name, "jsonl": short_path(jsonl_path)}},
    )
    return jsonl_path


_jsonl_path: Path = Path("/dev/null")


def get(name: str = "pfb") -> logging.Logger:
    if not _configured:
        configure()
    return logging.getLogger(name)


def die(message: str, code: int = 1) -> None:
    print(f"\n[error] {message}\n", file=sys.stderr)
    raise SystemExit(code)
