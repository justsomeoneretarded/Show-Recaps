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
# Story recaps legitimately describe violence, death and self-harm. Block only
# high-probability harm so ordinary drama plots are not refused.
SAFETY = [{"category": category, "threshold": "BLOCK_ONLY_HIGH"} for category in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


class GeminiUnavailable(PipelineError):
    """Quota, outage or unusable output: a local model may take over."""


def configured_key(env=None):
    env = os.environ if env is None else env
    return (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip()


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
    def __init__(self, key=None, model=None, *, session=None, sleep=time.sleep):
        self.key = key if key is not None else configured_key()
        self.model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
        self.session = session or requests.Session()
        self.sleep = sleep
        self.min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", "0") or 0)
        self._last = 0.0
        # After one exhausted retry cycle (daily quota, outage) stop asking for the rest of the
        # run, so optional calls cannot add hours of backoff.
        self.unavailable = False

    def generate_json(self, system, prompt, schema, *, temperature=None, max_tokens=32768,
                      media=None, media_resolution=None):
        """Structured JSON answer. ``media`` holds extra request parts such as
        ``{"fileData": {...}}`` or ``{"inlineData": {...}}`` placed before the prompt."""
        if not self.key:
            raise PipelineError("GEMINI_API_KEY is not configured.")
        if self.unavailable:
            raise GeminiUnavailable("Gemini was unavailable earlier in this run.")
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
        url = f"{API_ROOT}/{self.model}:generateContent"
        last_problem = "no attempt was made"
        for attempt in range(MAX_ATTEMPTS):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            response = None
            try:
                response = self.session.post(url, json=body, timeout=600,
                                             headers={"x-goog-api-key": self.key})
            except requests.RequestException as exc:
                last_problem = type(exc).__name__
            if response is not None:
                if response.status_code in (400, 401, 403, 404):
                    raise PipelineError(
                        f"Gemini rejected the request (HTTP {response.status_code}). Check GEMINI_API_KEY "
                        f"and that model {self.model} is available to this key.")
                if response.status_code not in RETRY_STATUSES:
                    try:
                        response.raise_for_status()
                        return json.loads(_text(response.json()))
                    except GeminiUnavailable as exc:
                        last_problem = str(exc)
                    except (ValueError, requests.HTTPError) as exc:
                        last_problem = f"unusable response ({type(exc).__name__})"
                else:
                    last_problem = f"HTTP {response.status_code}"
            if attempt + 1 < MAX_ATTEMPTS:
                self.sleep(_retry_delay(response, attempt))
        self.unavailable = True
        raise GeminiUnavailable(f"Gemini was unavailable after {MAX_ATTEMPTS} attempts: {last_problem}.")

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
