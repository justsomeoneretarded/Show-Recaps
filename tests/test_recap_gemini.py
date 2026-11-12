import json

import pytest

from movie_pipeline.core import PipelineError
from recap_pipeline import gemini
from recap_pipeline.gemini import GeminiClient, GeminiUnavailable
from recap_pipeline.writing import RecapWriter, SYSTEM


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self.payload, self.headers = status, payload or {}, headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def ok(value, **candidate):
    return Response(payload={"candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [{"text": "thinking...", "thought": True},
                              {"text": json.dumps(value)}]}, **candidate}]})


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def client(*responses, model="gemini-3-flash-preview"):
    sleeps = []
    session = Session(*responses)
    return GeminiClient("secret-key", model, session=session, sleep=sleeps.append), session, sleeps


def test_request_shape_keeps_key_out_of_url_and_skips_thought_parts():
    c, session, _ = client(ok({"paragraphs": []}))
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert c.generate_json("system", "prompt", schema, temperature=0.2) == {"paragraphs": []}
    url, kwargs = session.calls[0]
    assert "secret-key" not in url and url.endswith("/gemini-3-flash-preview:generateContent")
    assert kwargs["headers"] == {"x-goog-api-key": "secret-key"}
    config = kwargs["json"]["generationConfig"]
    assert config["responseJsonSchema"] == schema and config["responseMimeType"] == "application/json"
    assert "temperature" not in config  # Gemini 3 keeps its tuned default
    assert kwargs["json"]["systemInstruction"]["parts"][0]["text"] == "system"


def test_older_models_receive_the_requested_temperature():
    c, session, _ = client(ok({}), model="gemini-2.5-flash")
    c.generate_json("s", "p", {"type": "object"}, temperature=0.15)
    assert session.calls[0][1]["json"]["generationConfig"]["temperature"] == 0.15


def test_rate_limit_waits_for_the_server_retry_delay_then_succeeds():
    limited = Response(429, {"error": {"details": [{"retryDelay": "17s"}]}})
    c, session, sleeps = client(limited, ok({"ok": True}))
    assert c.generate_json("s", "p", {"type": "object"}) == {"ok": True}
    assert sleeps == [18.0] and len(session.calls) == 2


def test_truncated_or_invalid_output_is_retried():
    truncated = ok({"x": 1}, finishReason="MAX_TOKENS")
    broken = Response(payload={"candidates": [{"content": {"parts": [{"text": "{not json"}]}}]})
    c, session, _ = client(truncated, broken, ok({"ok": 1}))
    assert c.generate_json("s", "p", {"type": "object"}) == {"ok": 1}
    assert len(session.calls) == 3


def test_persistent_quota_failure_is_reported_as_unavailable():
    c, session, sleeps = client(*[Response(429) for _ in range(gemini.MAX_ATTEMPTS)])
    with pytest.raises(GeminiUnavailable, match="HTTP 429"):
        c.generate_json("s", "p", {"type": "object"})
    assert len(sleeps) == gemini.MAX_ATTEMPTS - 1 and max(sleeps) <= gemini.MAX_BACKOFF_SECONDS


def test_bad_key_fails_fast_without_retrying_or_falling_back():
    c, session, sleeps = client(Response(403))
    with pytest.raises(PipelineError, match="Check GEMINI_API_KEY") as error:
        c.generate_json("s", "p", {"type": "object"})
    assert not isinstance(error.value, GeminiUnavailable)
    assert sleeps == [] and "secret-key" not in str(error.value)


class FakeGemini:
    model = "gemini-test"

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def generate_json(self, system, prompt, schema, *, temperature=None):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.result


def local_post(monkeypatch, calls):
    class Local:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"local": true}'}}
    monkeypatch.setattr("recap_pipeline.writing.requests.post",
                        lambda url, **kwargs: calls.append((url, kwargs)) or Local())


def test_writer_uses_gemini_with_the_recap_system_prompt(monkeypatch):
    calls = []
    local_post(monkeypatch, calls)
    fake = FakeGemini({"gemini": True})
    writer = RecapWriter(gemini=fake)
    assert writer.model == "gemini-test"
    assert writer.ask("task", {"type": "object"}) == {"gemini": True}
    assert fake.calls[0][0] == SYSTEM and calls == []


def test_auto_mode_switches_to_the_local_model_for_the_rest_of_the_run(monkeypatch, capsys):
    calls = []
    local_post(monkeypatch, calls)
    fake = FakeGemini(error=GeminiUnavailable("quota"))
    writer = RecapWriter(gemini=fake)
    assert writer.ask("one", {"type": "object"}) == {"local": True}
    assert writer.ask("two", {"type": "object"}) == {"local": True}
    assert len(fake.calls) == 1 and len(calls) == 2
    assert writer.fell_back and writer.model == "gemma4:12b"
    assert "continuing with local model" in capsys.readouterr().out


def test_forced_gemini_mode_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("RECAP_WRITER", "gemini")
    writer = RecapWriter(gemini=FakeGemini(error=GeminiUnavailable("quota")))
    with pytest.raises(GeminiUnavailable):
        writer.ask("task", {"type": "object"})


def test_provider_selection_from_environment(monkeypatch):
    assert RecapWriter().gemini is None  # no key: local model as before
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert RecapWriter().model == gemini.DEFAULT_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    assert RecapWriter().model == "gemini-custom"
    monkeypatch.setenv("RECAP_WRITER", "local")
    assert RecapWriter().model == "gemma4:12b"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("RECAP_WRITER", "gemini")
    with pytest.raises(PipelineError, match="needs the GEMINI_API_KEY"):
        RecapWriter()
    monkeypatch.setenv("RECAP_WRITER", "cloud")
    with pytest.raises(PipelineError, match="auto, gemini or local"):
        RecapWriter()


def test_after_an_exhausted_retry_cycle_no_further_calls_are_made():
    c, session, sleeps = client(*[Response(429) for _ in range(gemini.MAX_ATTEMPTS)])
    with pytest.raises(GeminiUnavailable):
        c.generate_json("s", "p", {"type": "object"})
    calls = len(session.calls)
    with pytest.raises(GeminiUnavailable, match="earlier in this run"):
        c.generate_json("s", "p", {"type": "object"})
    assert len(session.calls) == calls
