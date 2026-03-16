# SPDX-License-Identifier: AGPL-3.0-or-later
"""PostgreSQL result cache plugin for SearXNG with pgvector semantic dedup.

Caches search results in PostgreSQL for persistent, cross-pod result sharing.
Supports exact query hash match AND semantic similarity via pgvector embeddings.

Configure via environment variables:
    SEARXNG_PG_CACHE_URL=postgresql://user:pass@host:5432/db
    SEARXNG_PG_SEMANTIC_THRESHOLD=0.10  (cosine distance, lower = stricter)
    SEARXNG_PG_CACHE_TTL=300  (seconds)
    SEARXNG_ONNX_MODEL_DIR=/usr/local/searxng/models  (all-MiniLM-L6-v2 ONNX)
"""

import hashlib
import json
import logging
import os
import time
import typing
import numpy as np

from searx.plugins import Plugin, PluginInfo

if typing.TYPE_CHECKING:
    from searx.search import SearchWithPlugins
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg

log = logging.getLogger("searx.plugins.pg_cache")

# Category-based cache TTLs (seconds)
# Stable content cached longer, time-sensitive content shorter
CATEGORY_TTLS = {
    "science": 86400,      # 24 hours — academic papers rarely change
    "images": 43200,       # 12 hours — image results are stable
    "videos": 21600,       # 6 hours
    "music": 21600,        # 6 hours — music catalog is stable
    "files": 10800,        # 3 hours — torrent/file results moderately stable
    "it": 600,             # 10 minutes — package registries update frequently
    "map": 3600,           # 1 hour — map data is stable
    "general": 300,        # 5 minutes — default
    "news": 300,           # 5 minutes — news is time-sensitive
}
DEFAULT_TTL = 300

# Connection pool (lazy-initialized)
_POOL = None
_PG_URL = None
_CACHE_TTL = int(os.environ.get("SEARXNG_PG_CACHE_TTL", "300"))
_SEMANTIC_THRESHOLD = float(os.environ.get("SEARXNG_PG_SEMANTIC_THRESHOLD", "0.10"))

# ONNX embedding model (lazy-initialized)
_ONNX_SESSION = None
_TOKENIZER = None
_MODEL_DIR = os.environ.get("SEARXNG_ONNX_MODEL_DIR", "/usr/local/searxng/models")

# Prevent threading issues with ONNX + Granian
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
        import psycopg_pool
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


def _get_embedder():
    """Lazy-initialize the ONNX embedding model."""
    global _ONNX_SESSION, _TOKENIZER
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION, _TOKENIZER

    model_path = os.path.join(_MODEL_DIR, "all-MiniLM-L6-v2.onnx")
    tokenizer_path = os.path.join(_MODEL_DIR, "tokenizer.json")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        log.warning("ONNX model not found at %s, semantic dedup disabled", _MODEL_DIR)
        return None, None

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        _ONNX_SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=_ort_options(),
        )
        _TOKENIZER = Tokenizer.from_file(tokenizer_path)
        _TOKENIZER.enable_truncation(max_length=128)
        _TOKENIZER.enable_padding(length=128)
        log.info("ONNX embedding model loaded: all-MiniLM-L6-v2 (384 dims)")
        return _ONNX_SESSION, _TOKENIZER
    except ImportError:
        log.warning("onnxruntime or tokenizers not installed, semantic dedup disabled")
        return None, None
    except Exception as e:
        log.error("ONNX model load failed: %s", e)
        return None, None


def _ort_options():
    """ONNX Runtime session options tuned for sidecar use."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _embed_query(query: str) -> list[float] | None:
    """Generate 384-dim embedding for a search query."""
    session, tokenizer = _get_embedder()
    if session is None or tokenizer is None:
        return None

    try:
        encoded = tokenizer.encode(query)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # Mean pooling over token embeddings
        token_embeddings = outputs[0]  # (1, seq_len, 384)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counted = np.sum(mask_expanded, axis=1)
        mean_pooled = summed / counted
        # L2 normalize
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = (mean_pooled / norm)[0]
        return normalized.tolist()
    except Exception as e:
        log.warning("Embedding generation failed: %s", e)
        return None


def _cache_key(query: str, categories: str, language: str, pageno: int, safesearch: int = 0) -> str:
    """Generate a deterministic cache key from search parameters."""
    raw = f"{query.strip().lower()}|{categories}|{language}|{pageno}|{safesearch}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class SXNGPlugin(Plugin):
    """PostgreSQL result cache with pgvector semantic deduplication."""

    id = "pg_cache"
    active = True  # Active by default; gracefully no-ops without SEARXNG_PG_CACHE_URL

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="PostgreSQL Cache",
            description="Caches search results in PostgreSQL with pgvector semantic deduplication.",
            preference_section="general",
        )

    def init(self, app) -> bool:
        """Initialize PG connection pool and verify schema."""
        pool = _get_pool()
        if pool is None:
            log.warning("PG cache plugin inactive: no database connection")
            return False
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
                        results_json JSONB NOT NULL,
                        query_embedding vector(384),
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

        # Pre-warm ONNX model (optional, non-blocking)
        session, tokenizer = _get_embedder()
        if session:
            log.info("Semantic dedup enabled (threshold: cosine distance < %.2f)", _SEMANTIC_THRESHOLD)
        else:
            log.info("Semantic dedup disabled (ONNX model not available)")

        return True

    def pre_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> bool:
        """Check PG cache before dispatching engines."""
        pool = _get_pool()
        if pool is None:
            return True

        query = search.search_query.query
        categories = ",".join(sorted(search.search_query.categories))
        language = search.search_query.lang
        pageno = search.search_query.pageno
        safesearch = search.search_query.safesearch

        key = _cache_key(query, categories, language, pageno, safesearch)

        try:
            with pool.connection() as conn:
                # Phase 1: Exact hash match
                row = conn.execute(
                    """SELECT results_json, result_count
                       FROM searxng_cache
                       WHERE query_hash = %s AND expires_at > NOW()""",
                    (key,)
                ).fetchone()

                if row:
                    results_json, result_count = row
                    results = json.loads(results_json) if isinstance(results_json, str) else results_json
                    for result in results:
                        search.result_container.add_result(result)
                    log.debug("PG cache EXACT HIT: %s (%d results)", query[:50], result_count)
                    # Increment hit_count for LFU-aware eviction (column may not exist yet)
                    try:
                        conn.execute(
                            "UPDATE searxng_cache SET hit_count = COALESCE(hit_count, 0) + 1 WHERE query_hash = %s",
                            (key,),
                        )
                    except Exception:
                        pass
                    search._pg_cache_hit = True
                    return False

                # Phase 2: Semantic similarity match (if ONNX available)
                embedding = _embed_query(query)
                if embedding is not None:
                    search._pg_query_embedding = embedding
                    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    row = conn.execute(
                        """SELECT results_json, result_count, query,
                                  query_embedding <-> %s::vector AS distance
                           FROM searxng_cache
                           WHERE query_embedding IS NOT NULL
                             AND expires_at > NOW()
                             AND query_embedding <-> %s::vector < %s
                           ORDER BY query_embedding <-> %s::vector
                           LIMIT 1""",
                        (vec_str, vec_str, _SEMANTIC_THRESHOLD, vec_str)
                    ).fetchone()

                    if row:
                        results_json, result_count, cached_query, distance = row
                        results = json.loads(results_json) if isinstance(results_json, str) else results_json
                        for result in results:
                            search.result_container.add_result(result)
                        log.debug(
                            "PG cache SEMANTIC HIT: '%s' ~ '%s' (distance=%.4f, %d results)",
                            query[:30], cached_query[:30], distance, result_count
                        )
                        # Increment hit_count for LFU-aware eviction (column may not exist yet)
                        try:
                            conn.execute(
                                "UPDATE searxng_cache SET hit_count = COALESCE(hit_count, 0) + 1 WHERE query_hash = %s",
                                (key,),
                            )
                        except Exception:
                            pass
                        search._pg_cache_hit = True
                        return False

        except Exception as e:
            log.warning("PG cache read error: %s", e)

        search._pg_cache_hit = False
        return True

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> None:
        """Store results in PG cache after search completes."""
        if getattr(search, '_pg_cache_hit', False):
            return

        pool = _get_pool()
        if pool is None:
            return

        query = search.search_query.query
        categories = ",".join(sorted(search.search_query.categories))
        language = search.search_query.lang
        pageno = search.search_query.pageno
        safesearch = search.search_query.safesearch

        key = _cache_key(query, categories, language, pageno, safesearch)

        # Determine TTL based on primary category (highest TTL wins for multi-category queries)
        categories_list = categories.split(",")
        ttl = max(CATEGORY_TTLS.get(cat.strip(), DEFAULT_TTL) for cat in categories_list) if categories_list else DEFAULT_TTL

        try:
            ordered = search.result_container.get_ordered_results()
            results = []
            for r in ordered[:50]:
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

            # Get or compute embedding
            embedding = getattr(search, '_pg_query_embedding', None)
            if embedding is None:
                embedding = _embed_query(query)

            with pool.connection() as conn:
                if embedding is not None:
                    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    conn.execute(
                        """INSERT INTO searxng_cache
                           (query_hash, query, categories, language, pageno,
                            result_count, results_json, query_embedding, expires_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector,
                                   NOW() + make_interval(secs => %s))
                           ON CONFLICT (query_hash) DO UPDATE SET
                             results_json = EXCLUDED.results_json,
                             result_count = EXCLUDED.result_count,
                             query_embedding = EXCLUDED.query_embedding,
                             expires_at = NOW() + make_interval(secs => %s)
                        """,
                        (key, query[:500], categories, language, pageno,
                         result_count, results_json, vec_str, ttl, ttl)
                    )
                else:
                    conn.execute(
                        """INSERT INTO searxng_cache
                           (query_hash, query, categories, language, pageno,
                            result_count, results_json, expires_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb,
                                   NOW() + make_interval(secs => %s))
                           ON CONFLICT (query_hash) DO UPDATE SET
                             results_json = EXCLUDED.results_json,
                             result_count = EXCLUDED.result_count,
                             expires_at = NOW() + make_interval(secs => %s)
                        """,
                        (key, query[:500], categories, language, pageno,
                         result_count, results_json, ttl, ttl)
                    )

            log.debug("PG cache STORE: %s (%d results, ttl=%ds, embedding=%s)",
                      query[:50], result_count, ttl, embedding is not None)

        except Exception as e:
            log.warning("PG cache write error: %s", e)
