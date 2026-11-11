"""Completion reminders are optional and never print token-bearing failures."""
from types import SimpleNamespace

import pytest
import requests

from recap_pipeline.notify import notify


def _env(**changes):
    return {"SHOW_TITLE": "Example <Show>", "UPCOMING_SEASON": "3",
            "GITHUB_REPOSITORY": "owner/Show-Recaps", "GITHUB_RUN_ID": "12345",
            "TELEGRAM_BOT_TOKEN": "private-token", "TELEGRAM_CHAT_ID": "private-chat", **changes}


class Transport:
    def __init__(self, *, fail=None, status=200, body=None):
        self.calls = []
        self.fail, self.status = fail, status
        self.body = body if body is not None else {"ok": True, "result": {"message_id": 42}}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail:
            raise self.fail
        return SimpleNamespace(status_code=self.status, json=lambda: self.body)


def test_success_mentions_both_packages_prior_season_run_link_and_48_hours():
    transport = Transport()
    assert notify(env=_env(), session=transport)
    assert len(transport.calls) == 2
    url, request = transport.calls[0]
    text = request["json"]["text"]
    assert url.endswith("/sendMessage") and request["timeout"] == 20
    assert "Both show recaps are ready" in text
    assert "Example &lt;Show&gt;" in text
    assert "Season 2 Recap and The Story So Far" in text
    assert "https://github.com/owner/Show-Recaps/actions/runs/12345" in text
    assert "48 hours" in text
    assert text.count("\n" * 6) == 3
    assert "private-token" not in text and "private-chat" not in text
    assert transport.calls[1][0].endswith("/pinChatMessage")
    assert transport.calls[1][1]["json"]["message_id"] == 42


@pytest.mark.parametrize("changes", [{"TELEGRAM_BOT_TOKEN": ""}, {"TELEGRAM_CHAT_ID": ""},
                                     {"GITHUB_RUN_ID": ""}, {"GITHUB_REPOSITORY": "https://evil.invalid"}])
def test_missing_secrets_or_run_link_skip_without_network(changes, capsys):
    transport = Transport()
    assert not notify(env=_env(**changes), session=transport)
    assert transport.calls == []
    assert "private-token" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", [requests.RequestException("https://api.telegram.org/botprivate-token/sendMessage"),
                                      ValueError("private-token"), RuntimeError("private-token")])
def test_transport_and_response_failure_never_leak_secret(failure, capsys):
    transport = Transport(fail=failure)
    assert not notify(env=_env(), session=transport)
    output = capsys.readouterr()
    assert "private-token" not in output.out + output.err
    assert "api.telegram.org" not in output.out + output.err


@pytest.mark.parametrize("status,body", [(403, {"description": "private-token"}),
                                         (200, {"ok": False, "description": "private-token"}),
                                         (200, []), (200, None)])
def test_provider_rejection_is_nonfatal_and_does_not_echo_body(status, body, capsys):
    transport = Transport(status=status, body=body)
    # Explicit None exercises malformed JSON rather than the transport default.
    transport.body = body
    assert not notify(env=_env(), session=transport)
    assert len(transport.calls) == 1
    assert "private-token" not in capsys.readouterr().out


def test_pin_can_be_disabled_or_omitted_without_failing_delivery():
    for env, body in [(_env(TELEGRAM_PIN="false"), {"ok": True, "result": {"message_id": 42}}),
                      (_env(), {"ok": True, "result": {}}),
                      (_env(), {"ok": True, "result": []})]:
        transport = Transport(body=body)
        assert notify(env=env, session=transport)
        assert len(transport.calls) == 1


def test_optional_pin_failure_does_not_reverse_delivery(capsys):
    transport = Transport()
    original = transport.post

    def post(url, **kwargs):
        if url.endswith("/pinChatMessage"):
            raise requests.RequestException("https://api.telegram.org/botprivate-token/pinChatMessage")
        return original(url, **kwargs)

    transport.post = post
    assert notify(env=_env(), session=transport)
    assert "private-token" not in capsys.readouterr().out


@pytest.mark.parametrize("incoming", ["1", "101", "bad", "2; secret"])
def test_malformed_season_input_cannot_inject_message_or_raise(incoming):
    transport = Transport()
    assert notify(env=_env(UPCOMING_SEASON=incoming, TELEGRAM_PIN="false"), session=transport)
    assert "Season previous Recap" in transport.calls[0][1]["json"]["text"]


def test_season_one_only_run_announces_a_single_package():
    transport = Transport()
    assert notify(env=_env(UPCOMING_SEASON="2"), session=transport)
    text = transport.calls[0][1]["json"]["text"]
    assert "Your show recap is ready" in text and "One package:</b> Season 1 Recap" in text
    assert "Both" not in text and "The Story So Far" not in text
