# SPDX-License-Identifier: AGPL-3.0-or-later
"""Providing a Valkey database for the botdetection methods.

Kubernetes-hardened fork: graceful degradation when Valkey is unavailable
instead of raising ValueError. Returns None so callers can fail-open.
"""

import logging

import valkey

__all__ = ["set_valkey_client", "get_valkey_client"]

logger = logging.getLogger(__name__)

CLIENT: valkey.Valkey | None = None
"""Global Valkey DB connection (Valkey client object)."""


def set_valkey_client(valkey_client: valkey.Valkey):
    global CLIENT  # pylint: disable=global-statement
    CLIENT = valkey_client


def get_valkey_client() -> valkey.Valkey | None:
    """Returns the Valkey client, or None if unavailable.

    Upstream raises ValueError when CLIENT is None, which crashes the
    ip_limit filter. This fork returns None so the limiter can degrade
    gracefully (skip rate limiting rather than 500).
    """
    if CLIENT is None:
        logger.debug("No Valkey connection available for botdetection")
        return None
    try:
        CLIENT.ping()
        return CLIENT
    except (valkey.exceptions.ValkeyError, ConnectionError):
        logger.warning("Valkey ping failed in botdetection, returning None")
        return None
