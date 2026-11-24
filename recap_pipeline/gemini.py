"""Gemini API transport for recap writing, with bounded retries for the free tier.

The API key travels in a request header, never in the URL, so request errors
cannot echo it into Actions logs.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

from movie_pipeline.core import PipelineError

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FILES_ROOT = "https://generativelanguage.googleapis.com/v1beta/files"
UPLOAD_ROOT = "https://generativelanguage.googleapis.com/upload/v1beta/files"
DEFAULT_MODEL = "gemini-3-flash-preview"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
MAX_BACKOFF_SECONDS = 90
# A per-minute quota refills within a minute or two: waiting is far better than handing the
# writing to the slow, weaker local model. Up to this many waits per request (~12 minutes).
MAX_QUOTA_WAITS = 8
# Long generations finish well within this; a hung connection must not cost ten minutes.
REQUEST_TIMEOUT = 300
# Free-tier quotas are counted per model, so when the main model's daily quota is spent on
# every key, these models (each with its own quota) carry on before the local model does.
# A model this API no longer offers is skipped.
FALLBACK_MODELS = "gemini-2.5-flash,gemini-flash-latest,gemini-flash-lite-latest"
# Story recaps legitimately describe violence, death and self-harm. Block only
# high-probability harm so ordinary drama plots are not refused.
SAFETY = [{"category": category, "threshold": "BLOCK_ONLY_HIGH"} for category in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


# Time a run may lose to Gemini (quota waits, backoff, failed or hung requests), shared by every
# client in the process. Past it Gemini is off for the rest of the run, so a spent free tier can
# never stall preparation for hours: the local model and cached results carry on.
DEFAULT_WAIT_BUDGET_MINUTES = 40
_RUN = {"lost": 0.0, "off": ""}


def reset_run_state():
    _RUN.update(lost=0.0, off="")


def _wait_budget():
    try:
        return max(0.0, float(os.getenv("GEMINI_WAIT_BUDGET_MINUTES", DEFAULT_WAIT_BUDGET_MINUTES))) * 60
    except ValueError:
        return DEFAULT_WAIT_BUDGET_MINUTES * 60


def _switch_off(reason):
    if not _RUN["off"]:
        _RUN["off"] = reason
        print(f"Gemini switched off for the rest of this run: {reason}.", flush=True)


class GeminiUnavailable(PipelineError):
    """Quota, outage or unusable output: a local model may take over."""


class NotGrounded(PipelineError):
    """A search-grounded answer cited no web source; the caller discards it (no retry)."""


KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GOOGLE_API_KEY")


def configured_keys(env=None):
    """Every configured key, primary first; spares take over when one hits its limit."""
    env = os.environ if env is None else env
    keys = [str(env.get(name) or "").strip() for name in KEY_NAMES]
    return list(dict.fromkeys(key for key in keys if key))


def configured_key(env=None):
    keys = configured_keys(env)
    return keys[0] if keys else ""


def _retry_delay(response, attempt):
    header = response.headers.get("Retry-After", "") if response is not None else ""
    if header.isdigit():
        return min(int(header), MAX_BACKOFF_SECONDS)
    try:
        for detail in response.json().get("error", {}).get("details", []):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                return min(float(match.group(1)) + 1, MAX_BACKOFF_SECONDS)
    except (AttributeError, ValueError):
        pass
    return min(2 ** (attempt + 1), MAX_BACKOFF_SECONDS)


def _quota(response):
    """(scope, quota ids) named by a 429: scope is "day", "minute" or "" when not stated."""
    ids = []
    try:
        for detail in response.json().get("error", {}).get("details", []):
            for violation in (detail.get("violations") or []) if isinstance(detail, dict) else []:
                if isinstance(violation, dict) and violation.get("quotaId"):
                    ids.append(str(violation["quotaId"]))
    except (AttributeError, ValueError):
        pass
    text = " ".join(ids)
    scope = "day" if re.search(r"per\s*day", text, re.I) else "minute" if re.search(r"per\s*minute", text, re.I) else ""
    return scope, ", ".join(dict.fromkeys(ids))


def _text(payload):
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not candidates:
        reason = (payload.get("promptFeedback", {}) or {}).get("blockReason") if isinstance(payload, dict) else None
        raise GeminiUnavailable(f"Gemini returned no candidate{f' ({reason})' if reason else ''}.")
    candidate = candidates[0]
    finish = candidate.get("finishReason", "STOP")
    if finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        raise GeminiUnavailable(f"Gemini stopped early: {finish}.")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts
                   if isinstance(part, dict) and not part.get("thought"))
    if not text.strip():
        raise GeminiUnavailable("Gemini returned an empty response.")
    return text


class GeminiClient:
    def __init__(self, key=None, model=None, *, session=None, sleep=time.sleep, keys=None):
        self.keys = list(keys) if keys is not None else ([key] if key is not None else configured_keys())
        self.keys = [k for k in self.keys if k]
        self.all_keys = list(self.keys)
        self.index = 0
        self.model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", FALLBACK_MODELS if model is None else "")
        self.fallbacks = [m.strip() for m in fallbacks.split(",") if m.strip() and m.strip() != self.model]
        self._reported = set()
        self.session = session or requests.Session()
        self.sleep = sleep
        self.min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", "0") or 0)
        self._last = 0.0
        # After one exhausted retry cycle (daily quota, outage) stop asking for the rest of the
        # run, so optional calls cannot add hours of backoff.
        self.unavailable = False

    @property
    def key(self):
        return self.keys[self.index] if self.keys else ""

    def _next_key(self, tried):
        """Switch to a key not yet tried in this request; False when all were tried."""
        for offset in range(1, len(self.keys)):
            candidate = (self.index + offset) % len(self.keys)
            if candidate not in tried:
                self.index = candidate
                print(f"Gemini key {len(tried)} hit its limit; switching to spare key {candidate + 1}.", flush=True)
                return True
        return False

    def generate_json(self, system, prompt, schema, *, temperature=None, max_tokens=32768,
                      media=None, media_resolution=None):
        """Structured JSON answer. ``media`` holds extra request parts such as
        ``{"fileData": {...}}`` or ``{"inlineData": {...}}`` placed before the prompt."""
        if not self.key:
            raise PipelineError("GEMINI_API_KEY is not configured.")
        self._check_run()
        config = {"responseMimeType": "application/json", "responseJsonSchema": schema,
                  "maxOutputTokens": max_tokens}
        if media_resolution:
            config["mediaResolution"] = media_resolution
        # Gemini 3 models are tuned for their default temperature; lowering it
        # can cause looping, so only older models receive the requested value.
        if temperature is not None and not self.model.startswith("gemini-3"):
            config["temperature"] = temperature
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [*(media or []), {"text": prompt}]}],
                "generationConfig": config, "safetySettings": SAFETY}
        # An uploaded file belongs to the key (project) that uploaded it: such a request stays
        # on that key and model rather than rotating to a spare that cannot read the file.
        pinned = any(isinstance(part, dict) and "fileData" in part for part in media or [])
        return self._request(body, lambda payload: json.loads(_text(payload)), pinned=pinned)

    def generate_grounded(self, system, prompt, *, max_tokens=32768):
        """Answer researched with Google Search: (text, [{"uri", "title"}] of the web sources used).

        Raises NotGrounded when the answer cited no web source.
        """
        if not self.key:
            raise PipelineError("GEMINI_API_KEY is not configured.")
        self._check_run()
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],
                "generationConfig": {"maxOutputTokens": max_tokens}, "safetySettings": SAFETY}

        def parse(payload):
            text = _text(payload)
            metadata = (payload.get("candidates") or [{}])[0].get("groundingMetadata") or {}
            sources = []
            for chunk in metadata.get("groundingChunks") or []:
                web = chunk.get("web") if isinstance(chunk, dict) else None
                if isinstance(web, dict) and web.get("uri"):
                    sources.append({"uri": str(web["uri"]), "title": str(web.get("title") or "")})
            if not sources:
                raise NotGrounded("Gemini's answer was not grounded in any web source.")
            return text, sources
        return self._request(body, parse, patient=False)

    def _check_run(self):
        if self.unavailable:
            raise GeminiUnavailable("Gemini was unavailable earlier in this run.")
        if _RUN["off"]:
            raise GeminiUnavailable(f"Gemini is off for this run ({_RUN['off']}).")

    def _lose(self, seconds, why=""):
        """Account time lost to Gemini; past the run budget switch Gemini off instead of waiting."""
        seconds = max(0.0, float(seconds))
        budget = _wait_budget()
        if _RUN["lost"] + seconds > budget:
            _switch_off(f"{budget / 60:.0f} minutes already lost to quota waits and failed requests")
            raise GeminiUnavailable(f"Gemini is off for this run ({_RUN['off']}).")
        _RUN["lost"] += seconds
        if why and seconds >= 1:
            print(f"Gemini {why}; waiting {seconds:.0f} s "
                  f"({_RUN['lost'] / 60:.1f} of {budget / 60:.0f} min budget used).", flush=True)

    def _wait(self, seconds, why):
        self._lose(seconds, why)
        self.sleep(seconds)

    def _switch_model(self):
        """Every key has spent the current model's daily quota: move to the next fallback model."""
        while self.fallbacks:
            previous, self.model = self.model, self.fallbacks.pop(0)
            self.keys, self.index = list(self.all_keys), 0
            print(f"Gemini {previous} has used its daily quota on every key; continuing with {self.model}.",
                  flush=True)
            return True
        return False

    def _request(self, body, parse, *, patient=True, pinned=False):
        """POST with key rotation. ``patient`` requests wait out per-minute quotas and move to a
        fallback model when a daily quota is spent; impatient ones (optional extras) give up."""
        last_problem = "no attempt was made"
        tried = {self.index}
        attempt = waits = 0
        on_fallback = False
        while attempt < MAX_ATTEMPTS:
            url = f"{API_ROOT}/{self.model}:generateContent"
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            response = None
            sent = time.monotonic()
            try:
                response = self.session.post(url, json=body, timeout=REQUEST_TIMEOUT,
                                             headers={"x-goog-api-key": self.key})
            except requests.RequestException as exc:
                last_problem = type(exc).__name__
                self._lose(time.monotonic() - sent)
                print(f"Gemini request failed ({last_problem}).", flush=True)
            if response is not None:
                if response.status_code in (401, 403) and pinned:
                    raise GeminiUnavailable(f"Gemini could not read the uploaded file (HTTP {response.status_code}).")
                if response.status_code in (401, 403):
                    # A revoked or wrong key is dropped; the spares carry on.
                    self.keys.pop(self.index)
                    self.all_keys = [k for k in self.all_keys if k in self.keys]
                    self.index = 0
                    tried = {0}
                    if not self.keys:
                        raise PipelineError(f"Gemini rejected every configured key (HTTP {response.status_code}). "
                                            "Check the GEMINI_API_KEY secrets.")
                    print(f"A Gemini key was rejected (HTTP {response.status_code}); using a spare key.", flush=True)
                    continue
                if response.status_code in (400, 404):
                    if on_fallback and patient:
                        # This fallback model is not offered (any more): try the next one.
                        print(f"Gemini model {self.model} is not available (HTTP {response.status_code}).", flush=True)
                        if self._switch_model():
                            tried = {self.index}
                            continue
                        break
                    raise PipelineError(
                        f"Gemini rejected the request (HTTP {response.status_code}). Check GEMINI_API_KEY "
                        f"and that model {self.model} is available to this key.")
                if response.status_code not in RETRY_STATUSES:
                    try:
                        response.raise_for_status()
                        return parse(response.json())
                    except GeminiUnavailable as exc:
                        last_problem = str(exc)
                    except (ValueError, requests.HTTPError) as exc:
                        last_problem = f"unusable response ({type(exc).__name__})"
                else:
                    last_problem = f"HTTP {response.status_code}"
                    self._lose(time.monotonic() - sent)
                    if response.status_code == 429:
                        scope, ids = _quota(response)
                        if ids and (self.model, ids) not in self._reported:
                            self._reported.add((self.model, ids))
                            print(f"Gemini quota reached on {self.model}: {ids}.", flush=True)
                        if scope == "day" and pinned:
                            # The next upload goes to a key that still has quota today.
                            if len(self.keys) > 1:
                                self.keys.pop(self.index)
                                self.index = 0
                            raise GeminiUnavailable(f"Gemini's daily quota is spent ({ids}).")
                        if scope == "day" and patient:
                            # Spent until tomorrow: never ask this key for this model again in this run.
                            self.keys.pop(self.index)
                            self.index = 0
                            tried = {0}
                            if self.keys:
                                continue
                            if self._switch_model():
                                on_fallback = True
                                tried = {self.index}
                                continue
                            self.keys = list(self.all_keys)
                            _switch_off("the daily quota is spent on every key and model")
                            break
                        if not pinned and self._next_key(tried):
                            tried.add(self.index)
                            continue  # a spare key with its own quota: no waiting
                        if not patient:
                            break
                        if scope == "minute" or waits < MAX_QUOTA_WAITS // 2:
                            # Every key is limited this minute: wait for the refill without
                            # spending one of the request's attempts.
                            waits += 1
                            if waits <= MAX_QUOTA_WAITS:
                                tried = {self.index}
                                self._wait(_retry_delay(response, min(waits, 6)),
                                           f"per-minute limit on every key ({self.model})")
                                continue
            attempt += 1
            tried = {self.index}
            if attempt < MAX_ATTEMPTS:
                self._wait(_retry_delay(response, attempt - 1), f"retry after {last_problem}")
        if patient and not pinned:
            self.unavailable = True
        raise GeminiUnavailable(f"Gemini was unavailable after {attempt + waits} attempts: {last_problem}.")

    def upload_file(self, path, mime_type, *, display_name="recap-source", wait_seconds=300):
        """Upload a local file with the resumable File API and wait until it can be used.

        Returns the file resource (``name``, ``uri``, ``mimeType``). Upload and
        processing problems raise GeminiUnavailable so callers can skip the file.
        """
        from pathlib import Path
        if not self.key:
            raise PipelineError("GEMINI_API_KEY is not configured.")
        data = Path(path).read_bytes()
        headers = {"x-goog-api-key": self.key, "X-Goog-Upload-Protocol": "resumable",
                   "X-Goog-Upload-Command": "start", "X-Goog-Upload-Header-Content-Length": str(len(data)),
                   "X-Goog-Upload-Header-Content-Type": mime_type, "Content-Type": "application/json"}
        try:
            start = self.session.post(UPLOAD_ROOT, headers=headers, timeout=60,
                                      json={"file": {"display_name": display_name}})
            upload_url = start.headers.get("X-Goog-Upload-URL") or start.headers.get("x-goog-upload-url")
            if start.status_code >= 400 or not upload_url:
                raise GeminiUnavailable(f"Gemini file upload could not start (HTTP {start.status_code}).")
            done = self.session.post(upload_url, data=data, timeout=600, headers={
                "x-goog-api-key": self.key, "Content-Length": str(len(data)),
                "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
            if done.status_code >= 400:
                raise GeminiUnavailable(f"Gemini file upload failed (HTTP {done.status_code}).")
            resource = done.json().get("file") or {}
        except (requests.RequestException, ValueError) as exc:
            raise GeminiUnavailable(f"Gemini file upload failed ({type(exc).__name__}).") from exc
        waited = 0.0
        while resource.get("state") == "PROCESSING" and waited < wait_seconds:
            self.sleep(3)
            waited += 3
            try:
                resource = self.session.get(f"{FILES_ROOT}/{resource['name'].split('/')[-1]}", timeout=30,
                                            headers={"x-goog-api-key": self.key}).json()
            except (requests.RequestException, ValueError, KeyError) as exc:
                raise GeminiUnavailable(f"Gemini file status was unavailable ({type(exc).__name__}).") from exc
        if resource.get("state") not in (None, "ACTIVE") or not resource.get("uri"):
            raise GeminiUnavailable(f"Gemini could not process the uploaded file ({resource.get('state')}).")
        return resource

    def delete_file(self, resource):
        """Best-effort cleanup; uploaded files also expire on their own."""
        try:
            self.session.delete(f"{FILES_ROOT}/{str(resource.get('name', '')).split('/')[-1]}",
                                timeout=30, headers={"x-goog-api-key": self.key})
        except requests.RequestException:
            pass


def shared_client():
    """A client when a key is configured and the writer has not been forced local, else None."""
    if os.getenv("RECAP_WRITER", "auto").strip().lower() == "local" or not configured_key():
        return None
    return GeminiClient()
