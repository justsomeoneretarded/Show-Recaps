

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
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_MIN_INTERVAL", "RECAP_WRITER"):
        monkeypatch.delenv(name, raising=False)
    from recap_pipeline import gemini
    gemini.reset_run_state()


@_pytest.fixture(autouse=True)
def _offline_youtube_defaults(monkeypatch):
    """No pacing sleeps in tests; image-fallback tests opt into stills explicitly by default."""
    monkeypatch.setenv("RECAP_YOUTUBE_PAUSE", "off")
    monkeypatch.setenv("RECAP_ALLOW_STILLS", "on")
    monkeypatch.setenv("RECAP_FANDOM", "off")
    from movie_pipeline import download_access
    download_access.reset()
