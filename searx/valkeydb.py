# SPDX-License-Identifier: AGPL-3.0-or-later
"""Implementation of the valkey client (valkey-py_) with reconnection support.

Kubernetes-hardened fork: adds retry logic for sidecar container startup races
and transient network failures. The upstream version sets _CLIENT=None on first
failure with no recovery path.

.. _valkey-py: https://github.com/valkey-io/valkey-py
"""

import os
import pwd
import logging
import time

import valkey
from searx import get_setting

_CLIENT: valkey.Valkey | None = None
_VALKEY_URL: str | None = None
_MAX_RETRIES = 5
_RETRY_DELAY = 2  # seconds

logger = logging.getLogger(__name__)


def client() -> valkey.Valkey | None:
    """Returns SearXNG's global Valkey DB connector. Attempts reconnection if
    the client was lost (e.g., sidecar restart)."""
    global _CLIENT  # pylint: disable=global-statement
    if _CLIENT is not None:
        try:
            _CLIENT.ping()
            return _CLIENT
        except (valkey.exceptions.ValkeyError, ConnectionError):
            logger.warning("Valkey connection lost, attempting reconnect...")
            _CLIENT = None

    if _VALKEY_URL:
        _connect(_VALKEY_URL)
    return _CLIENT


def _connect(valkey_url: str) -> bool:
    """Attempt to connect to Valkey with retry logic."""
    global _CLIENT  # pylint: disable=global-statement
    try:
        _CLIENT = valkey.Valkey.from_url(valkey_url)
        _CLIENT.ping()
        kwargs = _CLIENT.get_connection_kwargs().copy()
        kwargs.pop('password', None)
        kwargs_str = ' '.join([f'{k}={v!r}' for k, v in kwargs.items()])
        logger.info("connected to Valkey %s", kwargs_str)
        return True
    except valkey.exceptions.ValkeyError:
        _CLIENT = None
        return False


def initialize():
    global _CLIENT, _VALKEY_URL  # pylint: disable=global-statement
    import warnings

    if get_setting('redis.url'):
        warnings.warn("setting redis.url is deprecated, use valkey.url", DeprecationWarning)
    _VALKEY_URL = get_setting('valkey.url') or get_setting('redis.url')
    if not _VALKEY_URL:
        return False

    # Retry loop for sidecar startup race condition
    for attempt in range(1, _MAX_RETRIES + 1):
        if _connect(_VALKEY_URL):
            return True
        if attempt < _MAX_RETRIES:
            logger.warning(
                "Valkey not ready (attempt %d/%d), retrying in %ds...",
                attempt, _MAX_RETRIES, _RETRY_DELAY,
            )
            time.sleep(_RETRY_DELAY)

    _pw = pwd.getpwuid(os.getuid())
    logger.error(
        "[%s (%s)] can't connect Valkey DB after %d attempts",
        _pw.pw_name, _pw.pw_uid, _MAX_RETRIES,
    )
    return False
