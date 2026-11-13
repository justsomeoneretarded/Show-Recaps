"""Conservative, season-scoped recap visuals with bounded optional downloads.

A video must establish both the requested show and one prior season in provider
metadata. Search rank never supplies evidence. No model chooses footage identity:
Gemini may suggest extra searches and log what each accepted clip shows, but the
metadata checks below alone decide whether a clip belongs to a season.
"""
from __future__ import annotations

import hashlib
import heapq
import io
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from movie_pipeline import download_access
from movie_pipeline.core import PipelineError, normalized, probe
from movie_pipeline.video_settings import require_disk

CACHE_VERSION = 3
MAX_VIDEOS = 14
MAX_CANDIDATES = 60
MAX_DOWNLOAD_ATTEMPTS = 30
SEARCH_RESULTS = 8
MAX_VIDEO_BYTES = 400_000_000
MAX_RUN_BYTES = 9_000_000_000
MAX_IMAGE_BYTES = 12_000_000
MAX_IMAGE_FAILURES = 6
MAX_IMAGE_ATTEMPTS = 48
MAX_IMAGES = 32
REJECT = re.compile(
    r"\b(concept|fan.?made|ai.?generated|deepfake|remake|reaction|review|"
    r"recap|explained|analysis|full\s+(?:episode|season|series|movie)|"
    r"complete\s+(?:episode|season|series)|commentary|reactions?|reviews?|recaps?|fan\s+edit|"
    # Dubbed or subtitled foreign re-uploads carry burned-in captions and channel logos.
    r"legendad[oa]s?|dublad[oa]s?|doblad[oa]s?|subtitulad[oa]s?|sottotitolat[oi]|vostfr|vost|"
    r"deutsch|espa[nñ]ol|portugu[eê]s|fran[cç]ais|italiano|hindi|türkçe|dubbed|subbed)\b", re.I)
PROMO = re.compile(r"\b(trailer|teaser|scene|scenes|clip|clips|preview|moment|moments)\b", re.I)
SEASON = re.compile(r"\bseasons?\s*0*(\d{1,2})(?!\d)|\bs0*(\d{1,2})(?:e\d{1,3})?\b", re.I)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def _hash(*values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _season_numbers(text: str) -> set[int]:
    result = {int(a or b) for a, b in SEASON.findall(text)}
    for word, number in ORDINALS.items():
        if re.search(rf"\b{word}\s+season\b", text, re.I):
            result.add(number)
    # Ranges and combined-season compilations must not pass as one season.
    for first, last in re.findall(r"\bseasons?\s*(\d{1,2})\s*(?:[-–&/,]|and|to|through|thru)\s*(\d{1,2})\b", text, re.I):
        result.update((int(first), int(last)))
    for first, last in re.findall(r"\bs0*(\d{1,2})\s*[-–&/,]\s*s?0*(\d{1,2})\b", text, re.I):
        result.update((int(first), int(last)))
    return result


def _show_named(title: str, show_title: str) -> bool:
    """Contiguous title, with no unrecognised name extension (Dark != Dark Matter)."""
    title, name = normalized(title), normalized(show_title)
    if not name:
        return False
    match = re.search(rf"(?:^| ){re.escape(name)}(?: |$)", title)
    if not match:
        return False
    suffix = title[match.end():].strip()
    if not suffix:
        return True
    first = suffix.split()[0]
    return (first in {"season", "seasons", "official", "trailer", "teaser", "clip", "clips",
                      "scene", "scenes", "best", "preview", "moment", "moments", "hd", "4k"}
            or first in ORDINALS
            or bool(re.fullmatch(r"s\d{1,2}(?:e\d{1,3})?", first))
            or bool(re.fullmatch(r"(?:19|20)\d{2}", first)))


def _upload_in(info: dict, window) -> bool:
    if not window:
        return False
    raw = str(info.get("upload_date") or "")
    try:
        uploaded = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return False
    low, high = window
    return low <= uploaded <= high


def candidate_ok(info: dict, show: dict, season: int, upcoming_season: int,
                 season_year: int | None = None, window=None) -> bool:
    """Reject ambiguity, incoming seasons, other shows and commentary before download.

    A clip naming no season at all is accepted only when ``window`` (this season's
    airing window, ending well before any later season) contains its upload date.
    """
    title = str(info.get("title") or "")
    description = str(info.get("description") or "")
    try:
        seconds = float(info.get("duration") or 0)
    except (ValueError, TypeError):
        return False
    if (not 0 < season < upcoming_season or not math.isfinite(seconds)
            or not 8 <= seconds <= 480 or info.get("is_live") or info.get("was_live")
            or REJECT.search(title) or not PROMO.search(title)
            or not _show_named(title, str(show.get("title") or ""))):
        return False
    # A concept/fan reconstruction can advertise an otherwise convincing title.
    if re.search(r"\b(?:fan.?made|ai.?generated|concept\s+trailer|deepfake|remake)\b", description, re.I):
        return False
    declared_series = info.get("series")
    if declared_series and normalized(str(declared_series)) != normalized(str(show.get("title") or "")):
        return False
    refs = _season_numbers(title + " " + description)
    if refs != {season} and not (not refs and _upload_in(info, window)):
        return False
    explicit_season = info.get("season_number")
    if explicit_season is not None and str(explicit_season).lstrip("0") != str(season):
        return False
    title_years = set(re.findall(r"\b(?:19|20)\d{2}\b", title))
    title_years -= set(re.findall(r"\b(?:19|20)\d{2}\b", str(show.get("title") or "")))
    allowed_years = {str(value) for value in (show.get("year"), season_year) if value}
    if title_years and not title_years.issubset(allowed_years):
        return False
    identity_years = set(re.findall(r"\b((?:19|20)\d{2})\s+(?:television\s+|tv\s+)?series\b", description, re.I))
    if identity_years and identity_years != {str(show.get("year"))}:
        return False
    # Same-title adaptations need independent positive identity evidence when
    # research found ambiguity. A query's year never supplies that evidence.
    for field in ("imdb_id", "tvdb_id"):
        if info.get(field) and show.get(field) and str(info[field]) != str(show[field]):
            return False
    return (_identity_evidence(info, show) != "exact-title-and-season"
            or not bool(show.get("ambiguous_title")))


def _identity_evidence(info: dict, show: dict) -> str:
    if any(info.get(field) and show.get(field) and str(info[field]) == str(show[field])
           for field in ("imdb_id", "tvdb_id")):
        return "provider-id"
    premiere = str(show.get("year") or "")
    title = str(info.get("title") or "")
    description = str(info.get("description") or "")
    title_years = set(re.findall(r"\b(?:19|20)\d{2}\b", title))
    title_years -= set(re.findall(r"\b(?:19|20)\d{2}\b", str(show.get("title") or "")))
    identity_years = set(re.findall(r"\b((?:19|20)\d{2})\s+(?:television\s+|tv\s+)?series\b", description, re.I))
    year_identity = (
        premiere in title_years
        or identity_years == {premiere}
        or bool(re.search(rf"(?:^| ){re.escape(normalized(str(show.get('title') or '')))} {re.escape(premiere)}(?: |$)", normalized(description)))
        or bool(re.search(rf"\b(?:premiered|debuted)\s+(?:in\s+)?{re.escape(premiere)}\b", description, re.I))
    )
    return "premiere-year" if year_identity else "exact-title-and-season"

def search_queries(show: dict, season: int) -> list[str]:
    prefix = f'"{show["title"]}" {show.get("year") or ""} "season {season}"'
    genres = " ".join(map(str, show.get("genres") or [])).casefold()
    scene = ("funniest scenes" if "comedy" in genres else
             "suspense scene" if any(g in genres for g in ("horror", "thriller", "mystery")) else
             "action scene" if any(g in genres for g in ("action", "adventure", "science-fiction")) else
             "dramatic scene")
    return [f"{prefix} official trailer", f"{prefix} best scenes", f"{prefix} {scene}"]


def extra_queries(show: dict, season: dict) -> list[str]:
    """More official-footage searches: clips, sneak peeks, teasers and episode titles."""
    number = int(season["number"])
    prefix = f'"{show["title"]}" "season {number}"'
    queries = [f"{prefix} official clip", f"{prefix} sneak peek", f"{prefix} teaser",
               f"{prefix} scene", f'{show["title"]} season {number} clip']
    titles = [str(e.get("title") or "").strip() for e in season.get("episodes") or []]
    queries += [f'{prefix} "{title}" clip' for title in titles if title][:8]
    return queries


def _all_queries(show: dict, season: dict, client=None) -> list[str]:
    from .gemini_media import suggest_queries
    number = int(season["number"])
    titles = [str(e.get("title") or "") for e in season.get("episodes") or []]
    queries = [*search_queries(show, number), *suggest_queries(client, show, number, titles),
               *extra_queries(show, season)]
    return list(dict.fromkeys(queries))


def _url(entry: dict) -> str | None:
    identifier = str(entry.get("id") or "")
    if re.fullmatch(r"[\w-]{11}", identifier):
        return "https://www.youtube.com/watch?v=" + identifier
    url = str(entry.get("webpage_url") or entry.get("url") or "")
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname in {"www.youtube.com", "youtube.com", "youtu.be", "vimeo.com", "www.vimeo.com"}:
        return url
    return None


def _blocked(exc: Exception) -> bool:
    text = str(exc).casefold().replace("’", "'")
    return any(phrase in text for phrase in ("confirm you're not a bot", "sign in", "login required", "http error 429"))


class _Budget:
    def __init__(self):
        self.used = 0

    def can_fit(self, amount: int) -> bool:
        return self.used + max(0, amount) <= MAX_RUN_BYTES

    def add(self, amount: int):
        if not self.can_fit(amount):
            self.used = MAX_RUN_BYTES
            raise PipelineError("Recap source download byte budget exhausted.")
        self.used += max(0, amount)


def _video_info(path: Path) -> dict:
    metadata = probe(path)
    stream = next((row for row in metadata.get("streams", []) if row.get("codec_type") == "video"), {})
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    seconds = float(metadata.get("format", {}).get("duration") or 0)
    if width <= 0 or height <= 0 or not math.isfinite(seconds) or not 8 <= seconds <= 480:
        raise PipelineError("Recap footage has invalid dimensions or duration.")
    if path.stat().st_size > MAX_VIDEO_BYTES:
        raise PipelineError("Recap footage exceeds the per-source byte budget.")
    return {"width": width, "height": height, "duration": seconds}


def season_windows(seasons: list[dict]) -> dict[int, tuple]:
    """Upload windows for clips that name no season: from 90 days before a season's
    premiere until 150 days after its finale, and always before the next season."""
    from datetime import timedelta
    spans = {}
    for row in seasons:
        days = []
        for episode in row.get("episodes") or []:
            try:
                days.append(date.fromisoformat(str(episode.get("airdate") or "")[:10]))
            except ValueError:
                continue
        if days:
            spans[int(row["number"])] = (min(days), max(days))
    result = {}
    for number, (first, last) in spans.items():
        high = last + timedelta(days=150)
        following = [spans[n][0] for n in spans if n > number]
        if following:
            high = min(high, min(following) - timedelta(days=200))
        low = first - timedelta(days=90)
        if high > low:
            result[number] = (low, high)
    return result


def _clip_limit() -> int:
    try:
        limit = int(os.environ.get("RECAP_MAX_CLIPS_PER_SEASON", str(MAX_VIDEOS)))
    except (TypeError, ValueError) as exc:
        raise PipelineError("RECAP_MAX_CLIPS_PER_SEASON must be an integer from 1 to 30.") from exc
    if not 1 <= limit <= 30:
        raise PipelineError("RECAP_MAX_CLIPS_PER_SEASON must be an integer from 1 to 30.")
    return limit


def _discover_videos(show: dict, season: dict, incoming: int, root: Path,
                     ceiling: int, budget: _Budget, warnings: list[str], client=None,
                     window=None) -> list[dict]:
    import yt_dlp

    max_videos = _clip_limit()
    max_attempts = MAX_DOWNLOAD_ATTEMPTS

    number = int(season["number"])
    seen: set[str] = set()
    accepted: list[dict] = []
    attempts = 0
    base = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "socket_timeout": 20, "retries": 1, "fragment_retries": 1,
            "js_runtimes": {"node": {}}, "cachedir": False}
    for route in download_access.routes()[:2]:
        common = download_access.options(base, route)
        blocked = False
        for query in _all_queries(show, season, client):
            if blocked or len(seen) >= MAX_CANDIDATES or attempts >= max_attempts or len(accepted) >= max_videos:
                break
            try:
                with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
                    results = ydl.extract_info(f"ytsearch{SEARCH_RESULTS}:{query}", download=False) or {}
            except (yt_dlp.utils.DownloadError, OSError, ValueError) as exc:
                blocked = _blocked(exc)
                continue
            for entry in (results.get("entries") or [])[:SEARCH_RESULTS]:
                if (blocked or len(seen) >= MAX_CANDIDATES or attempts >= max_attempts
                        or len(accepted) >= max_videos):
                    break
                if not isinstance(entry, dict):
                    continue
                url = _url(entry)
                if not url or url in seen:
                    continue
                # A blocked route may retry this URL on the next configured route.
                try:
                    with yt_dlp.YoutubeDL(common) as ydl:
                        info = ydl.extract_info(url, download=False) or {}
                except (yt_dlp.utils.DownloadError, OSError, ValueError) as exc:
                    blocked = _blocked(exc)
                    if not blocked:
                        seen.add(url)
                    continue
                seen.add(url)
                if not candidate_ok(info, show, number, incoming, season.get("year"), window):
                    continue
                attempts += 1
                if budget.used + MAX_VIDEO_BYTES > MAX_RUN_BYTES:
                    warnings.append(f"Season {number}: footage search stopped at the shared source byte budget.")
                    return accepted
                require_disk(root)
                progress: dict[str, int] = {}
                source_bytes = 0

                def guard(data):
                    nonlocal source_bytes
                    filename = str(data.get("filename") or "download")
                    downloaded = int(data.get("downloaded_bytes") or 0)
                    delta = max(0, downloaded - progress.get(filename, 0))
                    progress[filename] = max(progress.get(filename, 0), downloaded)
                    source_bytes += delta
                    budget.add(delta)
                    if source_bytes > MAX_VIDEO_BYTES:
                        raise PipelineError("Recap source exceeded its download byte budget.")

                with tempfile.TemporaryDirectory(prefix="footage-", dir=root) as temporary:
                    attempt = Path(temporary) / "source.mp4"
                    options = {**common, "format": f"bv[height<={ceiling}]+ba/b[height<={ceiling}]",
                               "format_sort": ["res", "vcodec:h264"], "merge_output_format": "mp4",
                               "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
                               "outtmpl": str(attempt.with_suffix(".%(ext)s")),
                               "max_filesize": MAX_VIDEO_BYTES, "progress_hooks": [guard]}
                    try:
                        with yt_dlp.YoutubeDL(options) as ydl:
                            ydl.download([url])
                        if not attempt.is_file():
                            raise PipelineError("No video file was produced.")
                        metadata = _video_info(attempt)
                        # Include bytes even when a mocked/extractor path omits progress hooks.
                        budget.add(max(0, attempt.stat().st_size - source_bytes))
                        destination = root / f"video-{_hash(url)}.mp4"
                        attempt.replace(destination)
                        evidence = {key: info.get(key) for key in
                                    ("title", "description", "duration", "series", "season_number", "imdb_id", "tvdb_id",
                                     "upload_date")}
                        accepted.append({"path": str(destination.resolve()), "source_url": url,
                                         "title": str(info.get("title") or ""), "season": number,
                                         "kind": "video", **metadata, "episode_ids": [],
                                         "scope": "season", "illustrative": False, "metadata": evidence,
                                         "identity_evidence": _identity_evidence(info, show)})
                        download_access.record_route_success(route)
                    except (yt_dlp.utils.DownloadError, OSError, ValueError, PipelineError) as exc:
                        blocked = _blocked(exc)
                        if blocked:
                            # This metadata-verified candidate remains retryable via a configured route.
                            seen.discard(url)
        if len(accepted) >= max_videos or not blocked:
            break
        warnings.append(f"Season {number}: optional video provider blocked the {route} route; trying available fallback sources.")
    return accepted


def _image_url(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("original") or value.get("medium") or value.get("url")
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"https", "http"} and parsed.hostname else None


def _fetch_image(url: str, destination: Path, budget: _Budget) -> dict:
    if not budget.can_fit(MAX_IMAGE_BYTES):
        raise PipelineError("Recap source download byte budget exhausted.")
    require_disk(destination.parent)
    data = io.BytesIO()
    with requests.get(url, stream=True, timeout=(10, 25)) as response:
        response.raise_for_status()
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_IMAGE_BYTES:
            raise PipelineError("Recap still exceeds the image byte budget.")
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            budget.add(len(chunk))
            if data.tell() + len(chunk) > MAX_IMAGE_BYTES:
                raise PipelineError("Recap still exceeds the image byte budget.")
            data.write(chunk)
    data.seek(0)
    with Image.open(data) as original:
        if original.width < 160 or original.height < 90 or original.width * original.height > 30_000_000:
            raise PipelineError("Recap still has unusable image dimensions.")
        image = original.convert("RGB")
        image.load()
    image.save(destination, format="JPEG", quality=93)
    return {"width": image.width, "height": image.height}


def _episode_rows(season: dict) -> list[dict]:
    result = []
    for episode in season.get("episodes") or []:
        try:
            valid = int(episode.get("season", -1)) == int(season["number"])
            aired = str(episode.get("airdate") or "")[:10]
            if not aired or date.fromisoformat(aired) > date.today():
                valid = False
        except (TypeError, ValueError):
            valid = False
        if valid and episode.get("id") is not None:
            result.append(episode)
    return result


def _image_limit() -> int:
    try:
        limit = int(os.environ.get("RECAP_MAX_IMAGES_PER_SEASON", str(MAX_IMAGES)))
    except (TypeError, ValueError) as exc:
        raise PipelineError("RECAP_MAX_IMAGES_PER_SEASON must be an integer from 1 to 48.") from exc
    if not 1 <= limit <= MAX_IMAGE_ATTEMPTS:
        raise PipelineError("RECAP_MAX_IMAGES_PER_SEASON must be an integer from 1 to 48.")
    return limit


def _episode_order(episode: dict) -> tuple:
    try:
        number = int(episode.get("number") or 0)
    except (TypeError, ValueError):
        number = 0
    return (number if number > 0 else math.inf,
            str(episode.get("airdate") or ""), str(episode["id"]))


def _image_candidates(season: dict) -> list[dict]:
    """Finale, premiere, then the middle of the largest uncovered interval."""
    rows = sorted((episode for episode in _episode_rows(season)
                   if _image_url(episode.get("image"))), key=_episode_order)
    if not rows:
        return []
    indices = [len(rows) - 1]
    if len(rows) > 1:
        indices.append(0)
    gaps = [(-(len(rows) - 1), 0, len(rows) - 1)]
    while gaps:
        _, left, right = heapq.heappop(gaps)
        if right - left <= 1:
            continue
        middle = (left + right) // 2
        indices.append(middle)
        heapq.heappush(gaps, (-(middle - left), left, middle))
        heapq.heappush(gaps, (-(right - middle), middle, right))
    result, seen = [], set()
    for index in indices:
        url = _image_url(rows[index].get("image"))
        if url not in seen:
            seen.add(url)
            result.append(rows[index])
    return result


def _fallback_images(show: dict, season: dict, root: Path, budget: _Budget,
                     warnings: list[str], cached: list[dict] | None = None,
                     limit: int | None = None) -> list[dict]:
    number = int(season["number"])
    limit = _image_limit() if limit is None else limit
    candidates = _image_candidates(season)
    cached = cached or []
    available = {(str(identifier), asset.get("source_url")): asset
                 for asset in cached if asset.get("scope") == "episode"
                 for identifier in asset.get("episode_ids") or []}
    records = []
    failures = attempts = reused = 0
    stopped = set()
    for episode in candidates:
        if len(records) >= limit:
            stopped.add("the per-season image cap")
            break
        url = _image_url(episode.get("image"))
        saved = available.get((str(episode["id"]), url))
        if saved is not None:
            size = Path(saved["path"]).stat().st_size
            if budget.can_fit(size):
                budget.add(size)
                records.append(saved)
                reused += 1
            else:
                stopped.add("the shared source byte budget")
            continue
        # Keep scanning cached entries after a provider/budget stop. Previously
        # verified stills remain useful even when missing URLs are unreachable.
        if failures >= MAX_IMAGE_FAILURES:
            stopped.add("the failed-image attempt limit")
            continue
        if attempts >= MAX_IMAGE_ATTEMPTS:
            stopped.add("the total image attempt limit")
            continue
        if not budget.can_fit(MAX_IMAGE_BYTES):
            stopped.add("the shared source byte budget")
            continue
        attempts += 1
        destination = root / f"episode-{_hash(str(episode['id']), url)}.jpg"
        try:
            metadata = _fetch_image(url, destination, budget)
        except (requests.RequestException, OSError, ValueError, PipelineError, Image.DecompressionBombError):
            destination.unlink(missing_ok=True)
            failures += 1
            continue
        records.append({"path": str(destination.resolve()), "source_url": url,
                        "title": f"{show['title']} — S{number:02}E{int(episode.get('number') or 0):02}: {episode.get('title') or ''}",
                        "season": number, "kind": "image", **metadata,
                        "episode_ids": [str(episode["id"])], "scope": "episode", "illustrative": False})
    if records:
        order = {str(episode["id"]): _episode_order(episode) for episode in candidates}
        records.sort(key=lambda asset: min(order[str(identifier)] for identifier in asset["episode_ids"]
                                          if str(identifier) in order))
        if reused:
            warnings.append(f"Season {number}: using cached verified prior-episode stills, enriching the pool where possible.")
        else:
            warnings.append(f"Season {number}: no scoped footage was available; using verified prior-episode stills.")
        if len(records) < len(candidates):
            reasons = ", ".join(sorted(stopped)) or "unreachable image URLs"
            warnings.append(f"Season {number}: verified still coverage is {len(records)} of {len(candidates)} available episode images; limited by {reasons}.")
        return records
    # A current generic poster may depict the incoming season. It is safe only
    # with explicit prior-season provenance in the research metadata.
    poster = show.get("image")
    provenance = poster.get("season") if isinstance(poster, dict) else show.get("image_season")
    if str(provenance) == str(number) and (url := _image_url(poster)):
        saved = next((asset for asset in cached if asset.get("scope") == "season-poster"), None)
        if saved is not None and budget.can_fit(Path(saved["path"]).stat().st_size):
            budget.add(Path(saved["path"]).stat().st_size)
            warnings.append(f"Season {number}: using a cached illustrative season promotional image, not exact event evidence.")
            return [saved]
        if (failures < MAX_IMAGE_FAILURES and attempts < MAX_IMAGE_ATTEMPTS
                and budget.can_fit(MAX_IMAGE_BYTES)):
            destination = root / f"poster-{_hash(url)}.jpg"
            try:
                metadata = _fetch_image(url, destination, budget)
                records.append({"path": str(destination.resolve()), "source_url": url,
                                "title": f"{show['title']} — season {number} promotional image",
                                "season": number, "kind": "image", **metadata, "episode_ids": [],
                                "scope": "season-poster", "illustrative": True})
                warnings.append(f"Season {number}: a verified season promotional image is illustrative and does not establish any exact event.")
                return records
            except (requests.RequestException, OSError, ValueError, PipelineError, Image.DecompressionBombError):
                destination.unlink(missing_ok=True)
    if stopped:
        warnings.append(f"Season {number}: image collection stopped at {', '.join(sorted(stopped))}.")
    warnings.append(f"Season {number}: no verified season footage or imagery was reachable; "
                    "footage from another covered season will be used instead of a blank plate.")
    return []


def _fingerprint(show: dict, season: dict) -> str:
    return _hash(show.get("id"), show.get("title"), show.get("year"), show.get("ambiguous_title"), season.get("year"),
                 [(str(e.get("id")), e.get("season"), e.get("airdate"), e.get("image"))
                  for e in _episode_rows(season)], show.get("image"), show.get("image_season"))


def _cached(manifest: Path, show: dict, season: dict, incoming: int, window=None) -> list[dict] | None:
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        # Legacy sparse pools must pass the same identity/provenance checks before enrichment.
        if record.get("version") not in {1, CACHE_VERSION} or record.get("fingerprint") != _fingerprint(show, season):
            return None
        assets = record["assets"]
        if not isinstance(assets, list) or not assets:
            return None
        episode_images = {str(e["id"]): _image_url(e.get("image")) for e in _episode_rows(season)}
        for asset in assets:
            path = Path(asset["path"])
            if (asset.get("season") != int(season["number"]) or not path.is_file()
                    or path.parent.resolve() != manifest.parent.resolve()):
                return None
            if asset.get("kind") == "video":
                if not candidate_ok(asset.get("metadata") or {}, show, int(season["number"]), incoming,
                                    season.get("year"), window):
                    return None
                asset.update(_video_info(path))
                asset["identity_evidence"] = _identity_evidence(asset.get("metadata") or {}, show)
            elif asset.get("kind") == "image":
                with Image.open(path) as image:
                    image.load()
                    if image.width < 160 or image.height < 90:
                        return None
                    asset.update(width=image.width, height=image.height)
                if asset.get("scope") == "episode":
                    ids = asset.get("episode_ids") or []
                    if not ids or any(episode_images.get(str(i)) != asset.get("source_url") for i in ids):
                        return None
                elif asset.get("scope") == "season-poster":
                    poster = show.get("image")
                    provenance = poster.get("season") if isinstance(poster, dict) else show.get("image_season")
                    if str(provenance) != str(season["number"]) or asset.get("source_url") != _image_url(poster):
                        return None
                else:
                    # Old neutral title plates are never reused: rediscover real footage.
                    return None
            else:
                return None
        return assets
    except (OSError, KeyError, ValueError, TypeError, PipelineError, Image.DecompressionBombError):
        return None


def _letterbox_crop(path: Path) -> str | None:
    """FFmpeg crop filter that removes baked-in black bars, or None when there are none."""
    import subprocess
    try:
        info = _video_info(path)
    except PipelineError:
        return None
    width, height, seconds = info["width"], info["height"], info["duration"]
    found = []
    for fraction in (0.25, 0.5, 0.75):
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-ss", f"{seconds * fraction:.2f}", "-i", str(path), "-t", "3",
                 "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        rows = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
        if rows:
            found.append(tuple(map(int, rows[-1])))
    if len(found) < 2:
        return None
    # The bars must be consistent; a dark scene alone is not a letterbox.
    visible = max(row[1] for row in found)
    top = min(row[3] for row in found)
    if visible >= height * 0.92 or visible < height * 0.5:
        return None
    crop_width = min(width, int(visible * 16 / 9) // 2 * 2)
    left = (width - crop_width) // 2 // 2 * 2
    return f"crop={crop_width}:{visible}:{left}:{top},scale={width}:{height}:flags=lanczos,setsar=1"


def _remove_letterbox(asset: dict, warnings: list[str]) -> None:
    """Re-encode a letterboxed clip so the picture fills the 16:9 frame. Timing is unchanged."""
    path = Path(asset["path"])
    if asset.get("letterbox_removed"):
        return
    crop = _letterbox_crop(path)
    if not crop:
        return
    target = path.with_name(path.stem + "-filled.mp4")
    try:
        from movie_pipeline.core import ffmpeg
        ffmpeg("-i", path, "-vf", crop, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "copy", "-movflags", "+faststart", target)
        target.replace(path)
        asset["letterbox_removed"] = True
        asset.update(_video_info(path))
    except (OSError, PipelineError):
        target.unlink(missing_ok=True)
        warnings.append(f"Letterbox removal was skipped for {path.name}.")


def _log_clips(assets: list[dict], show: dict, season: dict, root: Path, client, warnings: list[str]) -> None:
    """Gemini watches each clip and logs clean scenes; skipped silently without a key."""
    from .gemini_media import analyse_clip
    titles = [str(e.get("title") or "") for e in season.get("episodes") or []]
    logged = 0
    for asset in assets:
        if asset.get("kind") != "video":
            continue
        _remove_letterbox(asset, warnings)
        result = analyse_clip(client, Path(asset["path"]), float(asset.get("duration") or 0), show,
                              int(season["number"]), titles, root / "gemini")
        if result and result.get("scenes"):
            asset["shots"] = result["scenes"]
            asset["thumbnail_times"] = result.get("thumbnail_times", [])
            logged += 1
    if client is not None and assets:
        warnings.append(f"Season {season['number']}: Gemini logged scenes in {logged} of "
                        f"{sum(a.get('kind') == 'video' for a in assets)} clips.")


def collect_assets(bundle: dict, spec, cache_dir: Path) -> dict:
    """Return local video/image paths shared by season and series recaps.

    Every returned season precedes spec.upcoming_season. Generic show posters
    without prior-season provenance never enter the visual pool. A season may
    come back empty; the renderer then borrows footage from another covered
    season rather than showing a blank plate.
    """
    from .gemini import shared_client
    show = bundle.get("show") or {}
    incoming = int(spec.upcoming_season)
    if (not show.get("id") or not show.get("title")
            or normalized(show["title"]) != normalized(spec.show_title)
            or int(show.get("year") or 0) != int(spec.premiere_year)):
        raise PipelineError("Recap visual identity does not match the researched show and premiere year.")
    root = Path(cache_dir) / f"show-{_hash(show['id'], show['title'], show['year'])}"
    root.mkdir(parents=True, exist_ok=True)
    result = {"seasons": {}, "sources": [], "warnings": []}
    budget = _Budget()
    image_limit = _image_limit()
    client = shared_client()
    prior = [row for row in bundle.get("seasons") or [] if 0 < int(row["number"]) < incoming]
    windows = season_windows(prior)
    for season in sorted(prior, key=lambda row: int(row["number"])):
        number = int(season["number"])
        season_root = root / f"season-{number:02}"
        season_root.mkdir(parents=True, exist_ok=True)
        manifest = season_root / "assets.json"
        window = windows.get(number)
        cached = _cached(manifest, show, season, incoming, window)
        if cached is None:
            assets = _discover_videos(show, season, incoming, season_root,
                                      2160 if spec.try_4k else 1080, budget, result["warnings"],
                                      client, window)
            if not assets:
                assets = _fallback_images(show, season, season_root, budget, result["warnings"],
                                          limit=image_limit)
        elif all(asset.get("kind") == "video" for asset in cached):
            assets = []
            for asset in cached:
                size = Path(asset["path"]).stat().st_size
                if budget.can_fit(size):
                    budget.add(size)
                    assets.append(asset)
                else:
                    result["warnings"].append(f"Season {number}: cached footage exceeds the shared source byte budget.")
            if not assets:
                assets = _fallback_images(show, season, season_root, budget, result["warnings"],
                                          limit=image_limit)
        else:
            assets = _fallback_images(show, season, season_root, budget, result["warnings"],
                                      cached=cached, limit=image_limit)
        _log_clips(assets, show, season, season_root, client, result["warnings"])
        payload = {"version": CACHE_VERSION, "image_limit": image_limit,
                   "fingerprint": _fingerprint(show, season), "assets": assets}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", dir=season_root, delete=False) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temp_path = Path(temporary.name)
        temp_path.replace(manifest)
        result["seasons"][str(number)] = assets
        result["sources"].extend({"url": asset["source_url"], "title": asset["title"], "season": number}
                                 for asset in assets if asset.get("source_url"))
    if not result["seasons"]:
        raise PipelineError("No researched prior seasons are available for recap visuals.")
    if not any(result["seasons"].values()):
        raise PipelineError("No footage or imagery could be downloaded for any covered season. "
                            "Add or refresh the YOUTUBE_COOKIES secret and run again; no blank-plate video is made.")
    return result


def link_footage(collected: dict, ledger: dict, client=None) -> dict[str, float]:
    """Tag each logged scene with the sourced story event it shows.

    Links stay inside the clip's own season. Returns seconds of linked footage per
    event id, which tells the writer which moments can be shown on screen.
    """
    from .gemini_media import link_scenes
    by_season = {}
    for event in ledger.get("events", []) if isinstance(ledger, dict) else []:
        if isinstance(event, dict) and event.get("id") and event.get("season") is not None:
            by_season.setdefault(int(event["season"]), []).append(event)
    footage: dict[str, float] = {}
    linked = scenes = 0
    for key, rows in (collected.get("seasons") or {}).items():
        events = by_season.get(int(key), [])
        for row in rows:
            shots = row.get("shots") if row.get("kind") == "video" else None
            if not isinstance(shots, list) or not any(s.get("description") for s in shots):
                continue
            already = {i: s for i, s in enumerate(shots) if s.get("event_id")}
            links = already and {i: {"id": s["event_id"], "episode_id": s.get("episode_id"),
                                     "summary": s.get("event_summary", "")} for i, s in already.items()}
            if not links:
                links = link_scenes(client, shots, events)
            scenes += len(shots)
            for index, event in links.items():
                shot = shots[index]
                shot.update(event_id=str(event["id"]), episode_id=str(event.get("episode_id") or ""),
                            event_summary=str(event.get("summary", ""))[:300])
                footage[str(event["id"])] = round(footage.get(str(event["id"]), 0.0)
                                                  + float(shot["end"]) - float(shot["start"]), 2)
                linked += 1
    if scenes:
        collected.setdefault("warnings", []).append(
            f"Gemini linked {linked} of {scenes} logged scenes to {len(footage)} sourced story events.")
    return footage
