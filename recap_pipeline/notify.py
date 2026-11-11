"""Best-effort completion reminder, called only after every upload succeeds."""
from __future__ import annotations

from html import escape
import os
import re

import requests


def _run_url(env):
    repository, run_id = env.get("GITHUB_REPOSITORY", ""), env.get("GITHUB_RUN_ID", "")
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            or not re.fullmatch(r"[0-9]+", run_id)):
        return None
    # A fixed trusted host prevents environment values from adding credentials
    # or a hostile URL to the reminder.
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def notify(*, env=None, session=None):
    env = os.environ if env is None else env
    session = requests if session is None else session
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram recap reminder skipped: bot/chat secrets are not configured.", flush=True)
        return False
    run_url = _run_url(env)
    if not run_url:
        print("Telegram recap reminder skipped: no valid GitHub run link.", flush=True)
        return False
    title = " ".join(env.get("SHOW_TITLE", "Your show").split())[:300] or "Your show"
    incoming = env.get("UPCOMING_SEASON", "").strip()
    season = str(int(incoming) - 1) if re.fullmatch(r"(?:[2-9]|[1-9][0-9]|100)", incoming) else "previous"
    gap = "\n" * 6
    if season == "1":
        lines = ["✅ <b>Your show recap is ready!</b>", f"<b>Show:</b> {escape(title)}",
                 "<b>One package:</b> Season 1 Recap"]
        closing = "\n\nOpen Artifacts to download the package. Available for 48 hours."
    else:
        lines = ["✅ <b>Both show recaps are ready!</b>", f"<b>Show:</b> {escape(title)}",
                 f"<b>Two packages:</b> Season {season} Recap and The Story So Far"]
        closing = "\n\nOpen Artifacts to download both packages. Available for 48 hours."
    message = gap.join([*lines, f'<b>Download:</b> <a href="{escape(run_url, quote=True)}">GitHub run link</a>']) + closing
    try:
        response = session.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
            "chat_id": chat_id, "text": message, "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }, timeout=20)
        if response.status_code != 200:
            print("Telegram recap reminder was not delivered. Both packages remain in GitHub Artifacts.", flush=True)
            return False
        data = response.json()
        if not isinstance(data, dict) or not data.get("ok"):
            print("Telegram did not accept the recap reminder. Both packages remain in Artifacts.", flush=True)
            return False
    except Exception:
        # Transport exceptions include the URL, which embeds the token.
        print("Telegram recap reminder could not be confirmed. Both packages remain in Artifacts.", flush=True)
        return False
    print("Telegram recap completion reminder delivered.", flush=True)
    result = data.get("result")
    message_id = result.get("message_id") if isinstance(result, dict) else None
    if env.get("TELEGRAM_PIN", "true").strip().lower() == "false" or type(message_id) is not int:
        return True
    try:
        pinned = session.post(f"https://api.telegram.org/bot{token}/pinChatMessage", json={
            "chat_id": chat_id, "message_id": message_id, "disable_notification": False,
        }, timeout=20)
        if pinned.status_code == 200 and pinned.json().get("ok"):
            print("Telegram recap reminder pinned.", flush=True)
        else:
            print("Reminder sent; optional pinning was unavailable.", flush=True)
    except Exception:
        print("Reminder sent; optional pinning could not be confirmed.", flush=True)
    return True


if __name__ == "__main__":
    notify()
