# SPDX-License-Identifier: AGPL-3.0-or-later
"""SSE (Server-Sent Events) progressive result delivery for SearXNG.

Kubernetes-hardened fork: two-phase engine dispatch + SSE streaming.
Fast engines (<= 2s timeout) render in initial HTML response.
Slow engines stream results via SSE as they complete.

Usage:
  GET /search?q=...              → initial HTML with fast results + SSE div
  GET /search/stream?q=...       → SSE stream of slow engine results

The SSE endpoint returns HTML fragments for each result, compatible
with HTMX sse-swap for client-side DOM updates.
"""

import json
import logging
import time
import threading
import uuid
from collections import defaultdict

import flask

logger = logging.getLogger("searx.stream")

# In-memory session store for pending slow-engine results
# Session ID -> {results: [], complete: bool, engines_pending: int}
_SESSIONS: dict[str, dict] = {}
_SESSION_LOCK = threading.Lock()
_SESSION_TTL = 30  # seconds before cleanup

# Engine timeout threshold: engines with timeout <= this are "fast"
FAST_TIMEOUT_THRESHOLD = 2.0


def create_session(search_id: str, slow_engine_count: int) -> str:
    """Create a session for tracking slow engine results."""
    session_id = str(uuid.uuid4())[:8]
    with _SESSION_LOCK:
        _SESSIONS[session_id] = {
            "search_id": search_id,
            "results": [],
            "complete": False,
            "engines_pending": slow_engine_count,
            "engines_done": 0,
            "created_at": time.monotonic(),
        }
    return session_id


def add_result(session_id: str, result_html: str, engine_name: str):
    """Add a result to a session (called from engine thread)."""
    with _SESSION_LOCK:
        session = _SESSIONS.get(session_id)
        if session:
            session["results"].append({
                "html": result_html,
                "engine": engine_name,
                "time": time.monotonic(),
            })


def mark_engine_done(session_id: str):
    """Mark an engine as completed."""
    with _SESSION_LOCK:
        session = _SESSIONS.get(session_id)
        if session:
            session["engines_done"] += 1
            if session["engines_done"] >= session["engines_pending"]:
                session["complete"] = True


def get_session(session_id: str) -> dict | None:
    """Get session data."""
    with _SESSION_LOCK:
        return _SESSIONS.get(session_id)


def cleanup_sessions():
    """Remove expired sessions."""
    now = time.monotonic()
    with _SESSION_LOCK:
        expired = [
            sid for sid, s in _SESSIONS.items()
            if now - s["created_at"] > _SESSION_TTL
        ]
        for sid in expired:
            del _SESSIONS[sid]


def stream_results(session_id: str):
    """Generator that yields SSE events as slow engine results arrive."""
    start = time.monotonic()
    last_idx = 0

    while True:
        session = get_session(session_id)
        if session is None:
            yield "event: error\ndata: session not found\n\n"
            return

        # Yield any new results
        with _SESSION_LOCK:
            new_results = session["results"][last_idx:]
            last_idx = len(session["results"])
            is_complete = session["complete"]
            engines_done = session["engines_done"]
            engines_pending = session["engines_pending"]

        for result in new_results:
            yield f"event: new-result\ndata: {json.dumps(result)}\n\n"

        # Send progress update
        if engines_pending > 0:
            progress = {
                "done": engines_done,
                "total": engines_pending,
                "elapsed": round(time.monotonic() - start, 1),
            }
            yield f"event: progress\ndata: {json.dumps(progress)}\n\n"

        if is_complete:
            final = {
                "total_results": last_idx,
                "engines_completed": engines_done,
                "elapsed": round(time.monotonic() - start, 1),
            }
            yield f"event: complete\ndata: {json.dumps(final)}\n\n"
            return

        # Timeout after 15 seconds
        if time.monotonic() - start > 15:
            yield f"event: timeout\ndata: {json.dumps({'elapsed': 15})}\n\n"
            return

        time.sleep(0.2)  # Poll interval


def partition_engines(requests, processors):
    """Split engine requests into fast and slow tiers based on timeout."""
    fast = []
    slow = []
    for engine_name, query, request_params in requests:
        proc = processors.get(engine_name)
        if proc and hasattr(proc, 'engine') and hasattr(proc.engine, 'timeout'):
            if proc.engine.timeout <= FAST_TIMEOUT_THRESHOLD:
                fast.append((engine_name, query, request_params))
            else:
                slow.append((engine_name, query, request_params))
        else:
            fast.append((engine_name, query, request_params))  # default to fast
    return fast, slow
