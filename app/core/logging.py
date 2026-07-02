"""
Structured logging setup.

Every module should use `get_logger(__name__)` rather than the stdlib
`logging` module directly, so log format stays consistent and we can
change output format (e.g. to JSON for a log aggregator) in one place.
"""

import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, label: str):
    """
    Usage:
        with log_duration(logger, "hybrid_retrieval"):
            candidates = retrieve(...)

    Logs elapsed ms on exit. Used to watch the 30s-timeout budget during
    development — every stage that could be slow (retrieval, cross-encoder,
    LLM call) should be wrapped with this.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{label} took {elapsed_ms:.1f}ms")
