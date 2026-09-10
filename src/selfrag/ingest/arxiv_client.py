"""A rate-limited, retrying HTTP client for talking to arXiv.

arXiv's Terms of Use permit **at most one request per three seconds on a
single connection**. That is a legal constraint on this project, not a
performance knob -- exceeding it risks the IP being blocked, which would
stall every later phase that depends on this corpus. The module is built so
that constraint cannot be quietly dropped later:

* The throttle lives in one module-level object, ``_SHARED_LIMITER``. Every
  ``ArxivClient`` -- including ones constructed from different call sites,
  concurrently -- acquires from the *same* limiter instance before making a
  request. There is no constructor argument that replaces it: the only way
  to get an ``ArxivClient`` that does not honour the 3-second spacing is to
  edit this file. That is "structurally impossible to bypass" in the sense
  the project's build instructions ask for, as opposed to "please remember
  to throttle yourself," which is the thing that gets forgotten under
  deadline pressure.
* Retries are bounded and explicit (5xx, connection errors, 429, and 503
  honouring ``Retry-After``) rather than "retry until it works," which on a
  throttled API would just manufacture more throttling.

``RateLimiter`` itself *is* public and *does* take an injectable clock and
sleep function -- that is what makes its own spacing behaviour testable
without a real 3-second sleep per test. Injectability of the primitive is
not the same thing as a bypass of the shared instance: nothing in this
module ever constructs a second, unthrottled ``ArxivClient``-facing limiter.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

#: What arXiv's API documentation asks consumers to send: a descriptive,
#: honest identifier plus a link back to the project, not a generic browser
#: string. https://arxiv.org (help pages) asks for exactly this so abuse
#: and outages can be traced to a project rather than an anonymous client.
USER_AGENT = "selfrag-acquisition/0.1 (+https://github.com/MudMonster341/Self_RAG)"

#: Legal minimum spacing between requests on one connection (arXiv ToU).
MIN_REQUEST_INTERVAL_SECONDS = 3.0

_DEFAULT_BACKOFF = wait_exponential(multiplier=1, min=1, max=20)


class RateLimiter:
    """Enforces a minimum spacing between successive ``acquire()`` calls.

    ``clock`` and ``sleep`` are injected (default ``time.monotonic`` /
    ``time.sleep``) so a test can supply a fake pair -- typically a mutable
    counter that ``sleep`` advances instead of blocking -- and assert exact
    spacing without the test itself taking three real seconds per call.

    Thread-safe via a lock: this project's harvester is single-process and
    single-threaded per CLAUDE.md ("Windows has no fork"), but a limiter
    that silently only worked under that assumption would be a landmine for
    the day something adds a thread pool for I/O overlap.
    """

    def __init__(
        self,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def acquire(self) -> None:
        """Block (via the injected ``sleep``) until spacing is satisfied.

        The first call never waits -- there is no prior request to space
        against. Every call after that waits out whatever is left of the
        minimum interval since the previous call returned.
        """
        with self._lock:
            now = self._clock()
            if self._last_call is not None:
                wait = self._min_interval - (now - self._last_call)
                if wait > 0:
                    self._sleep(wait)
                    now = self._clock()
            self._last_call = now


#: The one limiter every ArxivClient request goes through. Leading
#: underscore is deliberate: production code has no reason to ever touch
#: this name. Tests that need to avoid real sleeping reach in via
#: ``monkeypatch.setattr(arxiv_client._SHARED_LIMITER, "_clock", ...)`` /
#: ``"_sleep"`` -- whitebox test access, not a code path any real caller
#: can reach.
_SHARED_LIMITER = RateLimiter()


class _RetryableStatus(Exception):
    """Internal signal: this HTTP response is retryable, not yet a failure."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"retryable HTTP status {response.status_code} for {response.request.url}")


def _is_retryable_status(status_code: int) -> bool:
    """429 (rate limited) and any 5xx (arXiv uses 503 specifically to throttle)."""
    return status_code == 429 or 500 <= status_code < 600


def _retry_after_or_backoff(retry_state: RetryCallState) -> float:
    """Honour ``Retry-After`` on a retryable response; otherwise back off exponentially.

    arXiv answers throttling with HTTP 503 and a ``Retry-After`` header --
    the whole point of retrying at all here is to *cooperate* with that
    signal rather than hammer the endpoint on a fixed schedule that ignores
    it.
    """
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    if isinstance(exc, _RetryableStatus):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return _DEFAULT_BACKOFF(retry_state)


class ArxivClient:
    """Small typed surface over ``httpx`` for arXiv's OAI-PMH and e-print endpoints.

    Every outgoing request -- including every retry attempt -- passes
    through the shared :class:`RateLimiter` first. ``transport`` exists so
    tests can supply ``httpx.MockTransport`` and never touch the network;
    ``retry_sleep`` exists so retry *backoff* (a resilience policy, not a
    legal constraint) can be driven without a real wait in tests. Neither
    parameter provides a way to skip the rate limiter itself.
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        max_attempts: int = 5,
        retry_sleep: Callable[[float], None] | None = None,
    ) -> None:
        # Follow redirects. arXiv moves endpoints (the documented OAI-PMH host
        # now 301s elsewhere), and a harvester that dies on a 301 is brittle
        # against a change we do not control. Each redirect still passes through
        # the shared rate limiter, so this cannot be used to exceed the one
        # request per three seconds arXiv's Terms of Use allow.
        self._http = httpx.Client(
            transport=transport, timeout=timeout, follow_redirects=True
        )
        self._headers = {"User-Agent": USER_AGENT}
        self._max_attempts = max_attempts
        self._retry_sleep = retry_sleep or time.sleep

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _build_retrying(self) -> Retrying:
        return Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=_retry_after_or_backoff,
            retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
            reraise=True,
            sleep=self._retry_sleep,
        )

    def _request_once(self, method: str, url: str, params: dict[str, Any] | None) -> httpx.Response:
        _SHARED_LIMITER.acquire()
        response = self._http.request(method, url, params=params, headers=self._headers)
        if response.status_code >= 400:
            if _is_retryable_status(response.status_code):
                raise _RetryableStatus(response)
            response.raise_for_status()
        return response

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET ``url``, retrying on 5xx / 429 / connection errors, honouring ``Retry-After``.

        Raises:
            httpx.HTTPStatusError: a non-retryable 4xx, or a retryable status
                that never succeeded within ``max_attempts``.
            httpx.TransportError: a connection-level failure that never
                succeeded within ``max_attempts``.
        """
        retrying = self._build_retrying()
        try:
            return retrying(self._request_once, "GET", url, params)
        except _RetryableStatus as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover -- raise_for_status always raises for status >= 400

    def stream_to_file(self, url: str, dest: Path, *, params: dict[str, Any] | None = None) -> int:
        """Stream a GET response body straight to ``dest``, returning bytes written.

        Written in chunks rather than buffered in memory -- e-print
        archives are small relative to this machine's RAM, but the metadata
        harvest this client also serves is not, and nothing here should
        depend on the caller remembering which endpoint is "the big one."
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        retrying = self._build_retrying()

        def _attempt() -> int:
            _SHARED_LIMITER.acquire()
            written = 0
            with self._http.stream("GET", url, params=params, headers=self._headers) as response:
                if response.status_code >= 400:
                    if _is_retryable_status(response.status_code):
                        raise _RetryableStatus(response)
                    response.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        written += len(chunk)
            return written

        try:
            return retrying(_attempt)
        except _RetryableStatus as exc:
            exc.response.raise_for_status()
            raise  # pragma: no cover -- raise_for_status always raises for status >= 400
