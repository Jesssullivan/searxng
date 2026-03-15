# SPDX-License-Identifier: AGPL-3.0-or-later
""".. _botdetection.ip_limit:

Method ``ip_limit``
-------------------

Kubernetes-hardened fork: rate limit constants are configurable via limiter.toml.

Add to limiter.toml to override defaults:

.. code:: toml

   [botdetection.ip_limit]
   burst_window = 20
   burst_max = 15
   burst_max_suspicious = 2
   long_window = 600
   long_max = 150
   long_max_suspicious = 10
   api_window = 3600
   api_max = 4
   suspicious_ip_window = 2592000
   suspicious_ip_max = 3

"""

from ipaddress import (
    IPv4Network,
    IPv6Network,
)

import flask
import werkzeug

from searx.valkeylib import incr_sliding_window, drop_counter

from . import link_token
from . import config
from . import valkeydb
from ._helpers import (
    too_many_requests,
    logger,
)


logger = logger.getChild('ip_limit')

# Defaults — overridable via limiter.toml [botdetection.ip_limit]
BURST_WINDOW = 20
BURST_MAX = 15
BURST_MAX_SUSPICIOUS = 2
LONG_WINDOW = 600
LONG_MAX = 150
LONG_MAX_SUSPICIOUS = 10
API_WINDOW = 3600
API_MAX = 4
SUSPICIOUS_IP_WINDOW = 3600 * 24 * 30
SUSPICIOUS_IP_MAX = 3


def _get_limit(cfg: config.Config, key: str, default: int) -> int:
    """Read a rate limit value from config, falling back to module default."""
    try:
        val = cfg.get(f'botdetection.ip_limit.{key}')
        if val is not None:
            return int(val)
    except (KeyError, TypeError, ValueError):
        pass
    return default


def filter_request(
    network: IPv4Network | IPv6Network,
    request: flask.Request,
    cfg: config.Config,
) -> werkzeug.Response | None:

    # pylint: disable=too-many-return-statements
    valkey_client = valkeydb.get_valkey_client()

    if network.is_link_local and not cfg['botdetection.ip_limit.filter_link_local']:
        logger.debug("network %s is link-local -> not monitored by ip_limit method", network.compressed)
        return None

    # Read configurable limits
    burst_window = _get_limit(cfg, 'burst_window', BURST_WINDOW)
    burst_max = _get_limit(cfg, 'burst_max', BURST_MAX)
    burst_max_suspicious = _get_limit(cfg, 'burst_max_suspicious', BURST_MAX_SUSPICIOUS)
    long_window = _get_limit(cfg, 'long_window', LONG_WINDOW)
    long_max = _get_limit(cfg, 'long_max', LONG_MAX)
    long_max_suspicious = _get_limit(cfg, 'long_max_suspicious', LONG_MAX_SUSPICIOUS)
    api_window = _get_limit(cfg, 'api_window', API_WINDOW)
    api_max = _get_limit(cfg, 'api_max', API_MAX)
    suspicious_ip_window = _get_limit(cfg, 'suspicious_ip_window', SUSPICIOUS_IP_WINDOW)
    suspicious_ip_max = _get_limit(cfg, 'suspicious_ip_max', SUSPICIOUS_IP_MAX)

    if request.args.get('format', 'html') != 'html':
        c = incr_sliding_window(valkey_client, 'ip_limit.API_WINDOW:' + network.compressed, api_window)
        if c > api_max:
            return too_many_requests(network, "too many request in API_WINDOW")

    if cfg['botdetection.ip_limit.link_token']:

        suspicious = link_token.is_suspicious(network, request, True)

        if not suspicious:
            # this IP is no longer suspicious: release ip again / delete the counter of this IP
            drop_counter(valkey_client, 'ip_limit.SUSPICIOUS_IP_WINDOW' + network.compressed)
            return None

        # this IP is suspicious: count requests from this IP
        c = incr_sliding_window(
            valkey_client, 'ip_limit.SUSPICIOUS_IP_WINDOW' + network.compressed, suspicious_ip_window
        )
        if c > suspicious_ip_max:
            logger.error("BLOCK: too many request from %s in SUSPICIOUS_IP_WINDOW (redirect to /)", network)
            response = flask.redirect(flask.url_for('index'), code=302)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

        c = incr_sliding_window(valkey_client, 'ip_limit.BURST_WINDOW' + network.compressed, burst_window)
        if c > burst_max_suspicious:
            return too_many_requests(network, "too many request in BURST_WINDOW (BURST_MAX_SUSPICIOUS)")

        c = incr_sliding_window(valkey_client, 'ip_limit.LONG_WINDOW' + network.compressed, long_window)
        if c > long_max_suspicious:
            return too_many_requests(network, "too many request in LONG_WINDOW (LONG_MAX_SUSPICIOUS)")

        return None

    # vanilla limiter without extensions counts burst_max and long_max
    c = incr_sliding_window(valkey_client, 'ip_limit.BURST_WINDOW' + network.compressed, burst_window)
    if c > burst_max:
        return too_many_requests(network, "too many request in BURST_WINDOW (BURST_MAX)")

    c = incr_sliding_window(valkey_client, 'ip_limit.LONG_WINDOW' + network.compressed, long_window)
    if c > long_max:
        return too_many_requests(network, "too many request in LONG_WINDOW (LONG_MAX)")

    return None
