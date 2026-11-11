

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_genre_leak(monkeypatch):
    """Production sets VIDEO_GENRE for the run; tests start without it."""
    import os
    monkeypatch.delenv("VIDEO_GENRE", raising=False)
    yield
    os.environ.pop("VIDEO_GENRE", None)


@_pytest.fixture(autouse=True)
def _no_live_gemini(monkeypatch):
    """Tests never reach the real Gemini API, even when a developer has a key set."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_MIN_INTERVAL", "RECAP_WRITER"):
        monkeypatch.delenv(name, raising=False)
