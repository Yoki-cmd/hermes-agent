"""Stale-while-revalidate behaviour of ``cached_provider_model_ids``.

Regression context: the desktop model tab (``hermes:api`` →
``/api/model/options``) serially re-fetches every authenticated provider's
live ``/v1/models`` list whenever the 1h disk cache expires. On networks
where those endpoints are slow or unreachable, a single picker open blocked
15-70s and tripped the desktop's 15s IPC timeout
(``Error invoking remote method 'hermes:api': Error: Timed out connecting
to Hermes backend after 15000ms``).

These tests pin the fix: a stale entry is served immediately and refreshed
off-thread, and a failed live fetch is negative-cached so the picker never
re-blocks on a dead network.
"""

import json
import threading
import time
from unittest.mock import patch

import pytest

import hermes_cli.models as models_mod
from hermes_cli.models import cached_provider_model_ids


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    """Point the provider-models disk cache at a temp file.

    The credential fingerprint is pinned so test entries match whatever the
    real environment would compute.
    """
    path = tmp_path / "provider_models_cache.json"
    monkeypatch.setattr(models_mod, "_provider_models_cache_path", lambda: path)
    monkeypatch.setattr(models_mod, "_credential_fingerprint", lambda provider: "fp-x")
    return path


def _write_cache(cache_file, provider, *, fp="fp-x", at=None, models, negative=False):
    data = {}
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    entry = {
        "fp": fp,
        "at": at if at is not None else time.time(),
        "models": list(models),
    }
    if negative:
        entry["negative"] = True
    data[provider] = entry
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")


def _read_cache(cache_file, provider):
    if not cache_file.exists():
        return {}
    return json.loads(cache_file.read_text(encoding="utf-8")).get(provider, {})


class TestStaleWhileRevalidate:
    def test_stale_entry_served_without_blocking(self, cache_file, monkeypatch):
        """A stale entry must be returned instantly, with no synchronous fetch."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["glm-old"])
        fetch_threads = []

        def live(provider, *, force_refresh=False):
            fetch_threads.append(threading.current_thread())
            return ["glm-new"]

        monkeypatch.setattr(models_mod, "provider_model_ids", live)

        t0 = time.time()
        ids = cached_provider_model_ids("zai")

        assert ids == ["glm-old"]
        assert time.time() - t0 < 1.0
        # Any live fetch that already ran must NOT have been on the caller's
        # (main) thread — a synchronous fetch here is exactly the bug.
        assert all(not t is threading.main_thread() for t in fetch_threads)

    def test_stale_entry_refreshed_in_background(self, cache_file, monkeypatch):
        """After serving stale data, a background thread repopulates the cache."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["glm-old"])
        started = time.time()
        monkeypatch.setattr(
            models_mod,
            "provider_model_ids",
            lambda p, *, force_refresh=False: ["glm-new"],
        )

        ids = cached_provider_model_ids("zai")
        assert ids == ["glm-old"]

        deadline = time.time() + 10
        entry = {}
        while time.time() < deadline:
            entry = _read_cache(cache_file, "zai")
            if entry.get("models") == ["glm-new"] and entry.get("at", 0) >= started:
                break
            time.sleep(0.05)

        assert entry.get("models") == ["glm-new"]
        assert entry.get("at", 0) >= started

    def test_inflight_refresh_not_duplicated(self, cache_file, monkeypatch):
        """A provider already being refreshed in the background must not spawn a second thread."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["glm-old"])
        calls = []

        def live(provider, *, force_refresh=False):
            calls.append(provider)
            return ["glm-new"]

        monkeypatch.setattr(models_mod, "provider_model_ids", live)
        models_mod._provider_models_refresh_inflight.add("zai")
        try:
            ids = cached_provider_model_ids("zai")
            assert ids == ["glm-old"]
            time.sleep(0.2)
            assert calls == []
        finally:
            models_mod._provider_models_refresh_inflight.discard("zai")

    def test_dead_network_keeps_serving_stale(self, cache_file, monkeypatch):
        """When the background refresh fails, the next open still serves stale instantly."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["glm-old"])
        calls = []
        first_refresh_done = threading.Event()

        def live(provider, *, force_refresh=False):
            calls.append(provider)
            first_refresh_done.set()
            return []

        monkeypatch.setattr(models_mod, "provider_model_ids", live)

        assert cached_provider_model_ids("zai") == ["glm-old"]
        # Deterministically wait for the background refresh (a negative-cache
        # write) to finish, then re-open the picker.
        assert first_refresh_done.wait(timeout=5)
        # Background refresh failed → stale entry untouched on disk.
        assert _read_cache(cache_file, "zai")["models"] == ["glm-old"]
        assert cached_provider_model_ids("zai") == ["glm-old"]
        # Second open must NOT have triggered another synchronous fetch
        # (in-flight guard still held or negative-cached by the background
        # thread) — at most the one background call.
        assert len(calls) <= 2

    def test_fresh_cache_still_served(self, cache_file, monkeypatch):
        """A fresh entry is returned as before — no fetch, no background work."""
        _write_cache(cache_file, "zai", at=time.time(), models=["glm-fresh"])

        def fail(*_a, **_k):
            raise AssertionError("provider_model_ids must not be called")

        monkeypatch.setattr(models_mod, "provider_model_ids", fail)
        assert cached_provider_model_ids("zai") == ["glm-fresh"]

    def test_rotated_credentials_ignore_stale_entry(self, cache_file, monkeypatch):
        """A stale entry under a different credential fingerprint is not served."""
        _write_cache(
            cache_file, "zai", fp="fp-rotated-away", at=time.time() - 10, models=["old-cred"]
        )
        monkeypatch.setattr(
            models_mod, "provider_model_ids", lambda p, *, force_refresh=False: ["new-cred"]
        )
        assert cached_provider_model_ids("zai") == ["new-cred"]


class TestNegativeCache:
    def test_empty_live_fetch_is_negative_cached(self, cache_file, monkeypatch):
        """A failed live fetch must not re-block every subsequent picker open."""
        calls = []

        def live(provider, *, force_refresh=False):
            calls.append(provider)
            return []

        monkeypatch.setattr(models_mod, "provider_model_ids", live)

        assert cached_provider_model_ids("zai") == []
        assert len(calls) == 1
        assert cached_provider_model_ids("zai") == []
        assert len(calls) == 1

        entry = _read_cache(cache_file, "zai")
        assert entry.get("negative") is True
        assert entry.get("models") == []

    def test_negative_cache_expires_and_retries(self, cache_file, monkeypatch):
        """After the negative TTL the live fetch is attempted again."""
        _write_cache(
            cache_file, "zai", at=time.time() - 600, models=[], negative=True
        )
        calls = []

        def live(provider, *, force_refresh=False):
            calls.append(provider)
            return ["glm-new"]

        monkeypatch.setattr(models_mod, "provider_model_ids", live)
        assert cached_provider_model_ids("zai") == ["glm-new"]
        assert len(calls) == 1

    def test_failed_background_refresh_keeps_stale_models(self, cache_file, monkeypatch):
        """A failed background refresh must not wipe the stale model list from disk."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["glm-old"])
        refresh_done = threading.Event()

        def live(provider, *, force_refresh=False):
            refresh_done.set()
            return []  # network dead

        monkeypatch.setattr(models_mod, "provider_model_ids", live)

        assert cached_provider_model_ids("zai") == ["glm-old"]
        # set() fires inside live() before the background thread writes —
        # poll the file until the negative entry lands.
        deadline = time.time() + 10
        entry = {}
        while time.time() < deadline:
            entry = _read_cache(cache_file, "zai")
            if entry.get("negative"):
                break
            time.sleep(0.05)

        assert entry.get("negative") is True
        assert entry.get("models") == ["glm-old"]  # stale fallback preserved

        # And the next open still serves the stale list within the negative window.
        assert cached_provider_model_ids("zai") == ["glm-old"]

    def test_negative_cache_respects_fingerprint(self, cache_file, monkeypatch):
        """Rotating credentials invalidates a negative entry immediately."""
        _write_cache(
            cache_file, "zai", fp="fp-old", at=time.time(), models=[], negative=True
        )
        monkeypatch.setattr(
            models_mod, "provider_model_ids", lambda p, *, force_refresh=False: ["glm-new"]
        )
        assert cached_provider_model_ids("zai") == ["glm-new"]


class TestUnchangedPaths:
    def test_cold_fetch_persists_cache(self, cache_file, monkeypatch):
        """Cold miss + successful live fetch still persists a positive entry."""
        monkeypatch.setattr(
            models_mod, "provider_model_ids", lambda p, *, force_refresh=False: ["m1"]
        )
        assert cached_provider_model_ids("zai") == ["m1"]
        entry = _read_cache(cache_file, "zai")
        assert entry.get("models") == ["m1"]
        assert "negative" not in entry

    def test_force_refresh_fetches_synchronously(self, cache_file, monkeypatch):
        """Explicit refresh keeps its synchronous contract and overwrites the cache."""
        _write_cache(cache_file, "zai", models=["old"])
        monkeypatch.setattr(
            models_mod,
            "provider_model_ids",
            lambda p, *, force_refresh=False: ["new"],
        )
        assert cached_provider_model_ids("zai", force_refresh=True) == ["new"]
        assert _read_cache(cache_file, "zai")["models"] == ["new"]

    def test_force_refresh_failure_falls_back_to_stale(self, cache_file, monkeypatch):
        """A failed explicit refresh keeps serving the stale entry (same fingerprint)."""
        _write_cache(cache_file, "zai", at=time.time() - 7200, models=["old"])
        monkeypatch.setattr(
            models_mod, "provider_model_ids", lambda p, *, force_refresh=False: []
        )
        assert cached_provider_model_ids("zai", force_refresh=True) == ["old"]
        # Failed refresh must not overwrite the stale entry with a negative one.
        entry = _read_cache(cache_file, "zai")
        assert entry.get("models") == ["old"]
