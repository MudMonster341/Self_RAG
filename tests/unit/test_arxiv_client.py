"""Tests for selfrag.ingest.arxiv_client.

Covers the two guarantees this module exists to make structural rather
than conventional:

1. the shared rate limiter enforces >=3s spacing -- proven with an
   injected clock and sleep, so this file never actually sleeps for the
   limiter's sake even though several tests simulate tens of seconds of
   spacing across many requests;
2. the retry policy honours arXiv's own throttling signal (503 +
   ``Retry-After``, and bare 429) while never retrying an ordinary 4xx.

Every test that touches ``ArxivClient`` uses ``httpx.MockTransport`` --
nothing here reaches the network -- and the module-level ``_SHARED_LIMITER``
is monkeypatched to a fake clock/sleep pair by an autouse fixture, since it
is a singleton shared across the whole test session and would otherwise
carry real timestamps from one test into the next.
"""

from __future__ import annotations

import time as time_module

import httpx
import pytest

from selfrag.ingest import arxiv_client
from selfrag.ingest.arxiv_client import USER_AGENT, ArxivClient, RateLimiter


class FakeClock:
    """A controllable clock: ``sleep()`` advances it instead of blocking."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleep_calls: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def fast_shared_limiter(monkeypatch):
    """Give every test in this file a fresh fake clock on the shared limiter.

    ``_SHARED_LIMITER`` is a module-level singleton, so without this a
    test's spacing state would leak into the next test (and, worse, would
    make every ``ArxivClient.get``/``stream_to_file`` call in this file
    actually sleep for real between requests).
    """
    clock = FakeClock()
    monkeypatch.setattr(arxiv_client._SHARED_LIMITER, "_clock", clock.clock)
    monkeypatch.setattr(arxiv_client._SHARED_LIMITER, "_sleep", clock.sleep)
    monkeypatch.setattr(arxiv_client._SHARED_LIMITER, "_last_call", None)
    return clock


def _make_client(handler, **kwargs) -> ArxivClient:
    transport = httpx.MockTransport(handler)
    kwargs.setdefault("retry_sleep", lambda _seconds: None)
    return ArxivClient(transport=transport, **kwargs)


class TestRateLimiter:
    """The primitive itself, in isolation, with its own injected clock."""

    def test_first_acquire_never_waits(self):
        clock = FakeClock()
        limiter = RateLimiter(3.0, clock=clock.clock, sleep=clock.sleep)
        limiter.acquire()
        assert clock.sleep_calls == []

    def test_second_acquire_waits_out_the_remaining_interval(self):
        clock = FakeClock()
        limiter = RateLimiter(3.0, clock=clock.clock, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 1.0  # only 1s of "real" time passed on its own
        limiter.acquire()
        assert clock.sleep_calls == [2.0]  # tops up to the full 3s

    def test_acquire_does_not_wait_if_interval_already_elapsed(self):
        clock = FakeClock()
        limiter = RateLimiter(3.0, clock=clock.clock, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 5.0  # more than the minimum interval passed on its own
        limiter.acquire()
        assert clock.sleep_calls == []

    def test_ten_consecutive_acquires_are_spaced_by_exactly_the_minimum_interval(self):
        clock = FakeClock()
        limiter = RateLimiter(3.0, clock=clock.clock, sleep=clock.sleep)
        timestamps = []
        for _ in range(10):
            limiter.acquire()
            timestamps.append(clock.now)
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
        assert all(gap >= 3.0 for gap in gaps)
        assert gaps == [3.0] * 9  # exact with a fake clock -- no real-world slack

    def test_never_sleeps_for_real(self):
        """The point of injection: this simulates 27s of spacing in well under a second."""
        clock = FakeClock()
        limiter = RateLimiter(3.0, clock=clock.clock, sleep=clock.sleep)
        started = time_module.monotonic()
        for _ in range(10):
            limiter.acquire()
        assert time_module.monotonic() - started < 1.0


class TestUserAgent:
    def test_sends_a_descriptive_user_agent_naming_the_repo(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, text="ok")

        client = _make_client(handler)
        client.get("http://example.test/x")
        assert seen["user-agent"] == USER_AGENT
        assert "github.com/MudMonster341/Self_RAG" in seen["user-agent"]


class TestGet:
    def test_returns_response_body_on_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="hello")

        client = _make_client(handler)
        response = client.get("http://example.test/x")
        assert response.status_code == 200
        assert response.text == "hello"

    def test_passes_query_params_through(self):
        seen_params = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.update(dict(request.url.params))
            return httpx.Response(200, text="ok")

        client = _make_client(handler)
        client.get("http://example.test/x", params={"verb": "ListRecords", "set": "cs"})
        assert seen_params == {"verb": "ListRecords", "set": "cs"}

    def test_404_is_not_retried_and_raises_http_status_error(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(404, text="not found")

        client = _make_client(handler)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.get("http://example.test/x")
        assert exc_info.value.response.status_code == 404
        assert attempts["n"] == 1  # no retry at all

    def test_400_is_not_retried(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, text="bad request")

        client = _make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            client.get("http://example.test/x")
        assert attempts["n"] == 1

    def test_429_is_retried_and_eventually_succeeds(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(429, headers={"Retry-After": "1"})
            return httpx.Response(200, text="ok")

        client = _make_client(handler, max_attempts=5)
        response = client.get("http://example.test/x")
        assert response.text == "ok"
        assert attempts["n"] == 3

    def test_503_retry_after_is_honoured_as_the_wait_duration(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, headers={"Retry-After": "7"})
            return httpx.Response(200, text="ok")

        recorded_sleeps: list[float] = []
        client = _make_client(handler, retry_sleep=recorded_sleeps.append)
        response = client.get("http://example.test/x")
        assert response.text == "ok"
        assert 7.0 in recorded_sleeps

    def test_5xx_without_retry_after_backs_off_exponentially(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, text="ok")

        recorded_sleeps: list[float] = []
        client = _make_client(handler, retry_sleep=recorded_sleeps.append)
        client.get("http://example.test/x")
        assert len(recorded_sleeps) == 1
        assert recorded_sleeps[0] > 0  # exponential backoff, no Retry-After to honour

    def test_connection_error_is_retried(self):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(200, text="ok")

        client = _make_client(handler)
        response = client.get("http://example.test/x")
        assert response.text == "ok"
        assert attempts["n"] == 2

    def test_exhausted_retries_on_5xx_raises_http_status_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="still broken")

        client = _make_client(handler, max_attempts=3)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.get("http://example.test/x")
        assert exc_info.value.response.status_code == 500

    def test_exhausted_retries_on_connection_error_reraises_it(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("still refused")

        client = _make_client(handler, max_attempts=3)
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.test/x")

    def test_shared_limiter_enforces_spacing_across_successive_get_calls(self, fast_shared_limiter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="ok")

        client = _make_client(handler)
        client.get("http://example.test/a")
        t1 = fast_shared_limiter.now
        client.get("http://example.test/b")
        t2 = fast_shared_limiter.now
        assert t2 - t1 >= 3.0

    def test_shared_limiter_also_spaces_out_retry_attempts(self, fast_shared_limiter):
        """Each retry is itself a request on the connection -- it must be
        throttled too, not just the first attempt."""
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(500)
            return httpx.Response(200, text="ok")

        client = _make_client(handler)
        client.get("http://example.test/x")
        # 3 attempts => at least 2 waits of >=3s each were paid.
        assert fast_shared_limiter.now >= 6.0


class TestStreamToFile:
    def test_writes_response_body_to_disk(self, tmp_path):
        body = b"x" * 10_000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        client = _make_client(handler)
        dest = tmp_path / "out.bin"
        written = client.stream_to_file("http://example.test/big", dest)
        assert written == len(body)
        assert dest.read_bytes() == body

    def test_creates_parent_directories(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"data")

        client = _make_client(handler)
        dest = tmp_path / "nested" / "dir" / "out.bin"
        client.stream_to_file("http://example.test/big", dest)
        assert dest.read_bytes() == b"data"

    def test_retries_on_503_with_retry_after(self, tmp_path):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, headers={"Retry-After": "2"})
            return httpx.Response(200, content=b"final-content")

        recorded_sleeps: list[float] = []
        client = _make_client(handler, retry_sleep=recorded_sleeps.append)
        dest = tmp_path / "out.bin"
        written = client.stream_to_file("http://example.test/big", dest)
        assert written == len(b"final-content")
        assert dest.read_bytes() == b"final-content"
        assert 2.0 in recorded_sleeps

    def test_404_is_not_retried(self, tmp_path):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(404)

        client = _make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            client.stream_to_file("http://example.test/missing", tmp_path / "out.bin")
        assert attempts["n"] == 1


class TestContextManager:
    def test_can_be_used_as_a_context_manager_and_closed_twice_safely(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="ok")

        with _make_client(handler) as client:
            client.get("http://example.test/x")
        client.close()  # closing an already-closed client must not raise
