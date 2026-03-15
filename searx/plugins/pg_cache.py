# SPDX-License-Identifier: AGPL-3.0-or-later
"""PostgreSQL result cache plugin for SearXNG.

Caches search results in PostgreSQL for persistent, cross-pod result sharing.
Phase 1: Exact query hash match. Phase 2: pgvector semantic deduplication.

Configure via environment variable:
    SEARXNG_PG_CACHE_URL=postgresql://user:pass@host:5432/db

Schema (apply before enabling):
    CREATE TABLE IF NOT EXISTS searxng_cache (
        query_hash TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        categories TEXT NOT NULL DEFAULT 'general',
        language TEXT NOT NULL DEFAULT 'en',
        pageno INTEGER NOT NULL DEFAULT 1,
        result_count INTEGER NOT NULL DEFAULT 0,
        results_json TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '5 minutes'
    );
    CREATE INDEX idx_searxng_cache_expires ON searxng_cache (expires_at);
"""

import hashlib
import json
import logging
import os
import time
import typing

from searx.plugins import Plugin, PluginInfo

if typing.TYPE_CHECKING:
    from searx.search import SearchWithPlugins
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg

log = logging.getLogger("searx.plugins.pg_cache")

# Connection pool (lazy-initialized)
_POOL = None
_PG_URL = None
_CACHE_TTL = 300  # 5 minutes default


def _get_pool():
    """Lazy-initialize the PG connection pool."""
    global _POOL, _PG_URL
    if _POOL is not None:
        return _POOL

    _PG_URL = os.environ.get("SEARXNG_PG_CACHE_URL", "")
    if not _PG_URL:
        log.warning("SEARXNG_PG_CACHE_URL not set, PG cache disabled")
        return None

    try:
        import psycopg_pool  # psycopg 3 pool
        _POOL = psycopg_pool.ConnectionPool(
            _PG_URL,
            min_size=1,
            max_size=5,
            open=True,
            kwargs={"autocommit": True},
        )
        log.info("PG cache pool initialized: %s connections", _POOL.max_size)
        return _POOL
    except ImportError:
        log.error("psycopg or psycopg_pool not installed, PG cache disabled")
        return None
    except Exception as e:
        log.error("PG cache pool init failed: %s", e)
        return None


def _cache_key(query: str, categories: str, language: str, pageno: int) -> str:
    """Generate a deterministic cache key from search parameters."""
    raw = f"{query.strip().lower()}|{categories}|{language}|{pageno}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class SXNGPlugin(Plugin):
    """PostgreSQL result cache — persistent cross-pod search caching."""

    id = "pg_cache"
    active = False  # Opt-in via settings.yml enabled_plugins

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="PostgreSQL Cache",
            description="Caches search results in PostgreSQL for persistent cross-pod sharing.",
            preference_section="general",
        )

    def init(self, app) -> bool:
        """Initialize PG connection pool."""
        pool = _get_pool()
        if pool is None:
            log.warning("PG cache plugin inactive: no database connection")
            return False
        # Ensure table exists
        try:
            with pool.connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS searxng_cache (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        categories TEXT NOT NULL DEFAULT 'general',
                        language TEXT NOT NULL DEFAULT 'en',
                        pageno INTEGER NOT NULL DEFAULT 1,
                        result_count INTEGER NOT NULL DEFAULT 0,
                        results_json TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '5 minutes'
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_searxng_cache_expires
                    ON searxng_cache (expires_at)
                """)
            log.info("PG cache table verified")
        except Exception as e:
            log.error("PG cache table creation failed: %s", e)
            return False
        return True

    def pre_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> bool:
        """Check PG cache before dispatching engines."""
        pool = _get_pool()
        if pool is None:
            return True  # Continue search normally

        query = search.search_query.query
        categories = ",".join(sorted(search.search_query.categories))
        language = search.search_query.lang
        pageno = search.search_query.pageno

        key = _cache_key(query, categories, language, pageno)

        try:
            with pool.connection() as conn:
                row = conn.execute(
                    """SELECT results_json, result_count
                       FROM searxng_cache
                       WHERE query_hash = %s AND expires_at > NOW()""",
                    (key,)
                ).fetchone()

            if row:
                results_json, result_count = row
                results = json.loads(results_json)

                # Inject cached results into the search result container
                for result in results:
                    search.result_container.add_result(result)

                log.debug("PG cache HIT: %s (%d results)", query[:50], result_count)
                search._pg_cache_hit = True
                return False  # Stop search — results served from cache

        except Exception as e:
            log.warning("PG cache read error: %s", e)

        search._pg_cache_hit = False
        return True  # Continue search normally

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> None:
        """Store results in PG cache after search completes."""
        if getattr(search, '_pg_cache_hit', False):
            return  # Don't re-cache a cache hit

        pool = _get_pool()
        if pool is None:
            return

        query = search.search_query.query
        categories = ",".join(sorted(search.search_query.categories))
        language = search.search_query.lang
        pageno = search.search_query.pageno

        key = _cache_key(query, categories, language, pageno)

        try:
            ordered = search.result_container.get_ordered_results()
            results = []
            for r in ordered[:50]:  # Cap at 50 results per cache entry
                # Serialize only JSON-safe fields
                result = {}
                for k, v in r.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        result[k] = v
                    elif isinstance(v, list):
                        result[k] = [str(x) for x in v]
                results.append(result)

            if not results:
                return

            results_json = json.dumps(results, ensure_ascii=False)
            result_count = len(results)

            with pool.connection() as conn:
                conn.execute(
                    """INSERT INTO searxng_cache
                       (query_hash, query, categories, language, pageno,
                        result_count, results_json, expires_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, NOW() + INTERVAL '%s seconds')
                       ON CONFLICT (query_hash) DO UPDATE SET
                         results_json = EXCLUDED.results_json,
                         result_count = EXCLUDED.result_count,
                         expires_at = NOW() + INTERVAL '%s seconds'
                    """,
                    (key, query[:500], categories, language, pageno,
                     result_count, results_json, _CACHE_TTL, _CACHE_TTL)
                )

            log.debug("PG cache STORE: %s (%d results)", query[:50], result_count)

        except Exception as e:
            log.warning("PG cache write error: %s", e)
