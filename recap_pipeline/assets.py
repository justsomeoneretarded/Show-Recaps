"""Conservative, season-scoped recap visuals with bounded optional downloads.

A video must establish both the requested show and one prior season in provider
metadata. Search rank never supplies evidence. No model chooses footage identity.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw, ImageFont

from movie_pipeline import download_access
from movie_pipeline.core import PipelineError, normalized, probe
from movie_pipeline.video_settings import require_disk

CACHE_VERSION = 1
MAX_VIDEOS = 2
MAX_CANDIDATES = 8
MAX_DOWNLOAD_ATTEMPTS = 4
MAX_VIDEO_BYTES = 400_000_000
MAX_RUN_BYTES = 2_400_000_000
MAX_IMAGE_BYTES = 12_000_000
MAX_IMAGE_ATTEMPTS = 6
MAX_IMAGES = 3
REJECT = re.compile(
    r"\b(concept|fan.?made|ai.?generated|deepfake|remake|reaction|review|"
    r"recap|explained|analysis|full\s+(?:episode|season|series|movie)|"
    r"complete\s+(?:episode|season|series)|commentary|reactions?|reviews?|recaps?|fan\s+edit)\b", re.I)
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


def candidate_ok(info: dict, show: dict, season: int, upcoming_season: int,
                 season_year: int | None = None) -> bool:
    """Reject ambiguity, incoming seasons, other shows and commentary before download."""
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
    if refs != {season}:
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

    def add(self, amount: int):
        self.used += max(0, amount)
        if self.used > MAX_RUN_BYTES:
            raise PipelineError("Recap source download byte budget exhausted.")


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


def _discover_videos(show: dict, season: dict, incoming: int, root: Path,
                     ceiling: int, budget: _Budget, warnings: list[str]) -> list[dict]:
    import yt_dlp

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
        for query in search_queries(show, number):
            if blocked or len(seen) >= MAX_CANDIDATES or attempts >= MAX_DOWNLOAD_ATTEMPTS or len(accepted) >= MAX_VIDEOS:
                break
            try:
                with yt_dlp.YoutubeDL({**common, "extract_flat": True}) as ydl:
                    results = ydl.extract_info(f"ytsearch4:{query}", download=False) or {}
            except (yt_dlp.utils.DownloadError, OSError, ValueError) as exc:
                blocked = _blocked(exc)
                continue
            for entry in (results.get("entries") or [])[:4]:
                if (blocked or len(seen) >= MAX_CANDIDATES or attempts >= MAX_DOWNLOAD_ATTEMPTS
                        or len(accepted) >= MAX_VIDEOS):
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
                if not candidate_ok(info, show, number, incoming, season.get("year")):
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
                                    ("title", "description", "duration", "series", "season_number", "imdb_id", "tvdb_id")}
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
        if len(accepted) >= MAX_VIDEOS or not blocked:
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


def _fallback_images(show: dict, season: dict, root: Path, budget: _Budget,
                     warnings: list[str]) -> list[dict]:
    number = int(season["number"])
    records = []
    seen = set()
    attempts = 0
    for episode in _episode_rows(season):
        if attempts >= MAX_IMAGE_ATTEMPTS or len(records) >= MAX_IMAGES:
            break
        url = _image_url(episode.get("image"))
        if not url or url in seen:
            continue
        seen.add(url)
        attempts += 1
        destination = root / f"episode-{_hash(str(episode['id']), url)}.jpg"
        try:
            metadata = _fetch_image(url, destination, budget)
        except (requests.RequestException, OSError, ValueError, PipelineError, Image.DecompressionBombError):
            destination.unlink(missing_ok=True)
            continue
        records.append({"path": str(destination.resolve()), "source_url": url,
                        "title": f"{show['title']} — S{number:02}E{int(episode.get('number') or 0):02}: {episode.get('title') or ''}",
                        "season": number, "kind": "image", **metadata,
                        "episode_ids": [str(episode["id"])], "scope": "episode", "illustrative": False})
    if records:
        warnings.append(f"Season {number}: no scoped footage was available; using verified prior-episode stills.")
        return records
    # A current generic poster may depict the incoming season. It is safe only
    # with explicit prior-season provenance in the research metadata.
    poster = show.get("image")
    provenance = poster.get("season") if isinstance(poster, dict) else show.get("image_season")
    if str(provenance) == str(number) and (url := _image_url(poster)):
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
    destination = root / "neutral-title-card.jpg"
    _neutral_card(show["title"], number, destination)
    warnings.append(f"Season {number}: no verified season imagery was reachable. Using a neutral illustrative title plate with no actors or plot imagery; the generic current show poster was excluded.")
    return [{"path": str(destination.resolve()), "source_url": "",
             "title": f"{show['title']} — season {number} neutral title plate",
             "season": number, "kind": "image", "width": 1280, "height": 720,
             "episode_ids": [], "scope": "neutral-title-card", "illustrative": True}]


def _neutral_card(title: str, season: int, destination: Path):
    image = Image.new("RGB", (1280, 720), "#111923")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 1269, 709), outline="#72b7b4", width=3)
    def font(size):
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        return ImageFont.load_default()
    headline = " ".join(str(title).split())[:150]
    size = 76
    while size > 18 and draw.textbbox((0, 0), headline, font=font(size))[2] > 1120:
        size -= 2
    draw.text((80, 240), headline, fill="white", font=font(size))
    draw.text((80, 350), f"Season {season} Recap", fill="#72b7b4", font=font(55))
    draw.text((80, 590), "ILLUSTRATIVE TITLE PLATE · SEASON IMAGERY UNAVAILABLE",
              fill="#b4bec9", font=font(22))
    image.save(destination, format="JPEG", quality=94)


def _fingerprint(show: dict, season: dict) -> str:
    return _hash(show.get("id"), show.get("title"), show.get("year"), show.get("ambiguous_title"), season.get("year"),
                 [(str(e.get("id")), e.get("season"), e.get("airdate"), e.get("image"))
                  for e in _episode_rows(season)], show.get("image"), show.get("image_season"))


def _cached(manifest: Path, show: dict, season: dict, incoming: int) -> list[dict] | None:
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("version") != CACHE_VERSION or record.get("fingerprint") != _fingerprint(show, season):
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
                if not candidate_ok(asset.get("metadata") or {}, show, int(season["number"]), incoming, season.get("year")):
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
                elif (asset.get("scope") != "neutral-title-card" or not asset.get("illustrative")
                      or asset.get("source_url") or asset.get("episode_ids")):
                    return None
            else:
                return None
        return assets
    except (OSError, KeyError, ValueError, TypeError, PipelineError, Image.DecompressionBombError):
        return None


def collect_assets(bundle: dict, spec, cache_dir: Path) -> dict:
    """Return local video/image paths shared by season and series recaps.

    Every returned season precedes spec.upcoming_season. Generic show posters
    without prior-season provenance never enter the visual pool.
    """
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
    for season in sorted(bundle.get("seasons") or [], key=lambda row: int(row["number"])):
        number = int(season["number"])
        if not 0 < number < incoming:
            continue
        season_root = root / f"season-{number:02}"
        season_root.mkdir(parents=True, exist_ok=True)
        manifest = season_root / "assets.json"
        assets = _cached(manifest, show, season, incoming)
        if assets is None:
            assets = _discover_videos(show, season, incoming, season_root,
                                      2160 if spec.try_4k else 1080, budget, result["warnings"])
            if not assets:
                assets = _fallback_images(show, season, season_root, budget, result["warnings"])
            payload = {"version": CACHE_VERSION, "fingerprint": _fingerprint(show, season), "assets": assets}
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", dir=season_root, delete=False) as temporary:
                json.dump(payload, temporary, ensure_ascii=False, indent=2)
                temp_path = Path(temporary.name)
            temp_path.replace(manifest)
        else:
            for scope, message in (
                ("episode", "using cached verified prior-episode stills."),
                ("season-poster", "using a cached illustrative season promotional image, not exact event evidence."),
                ("neutral-title-card", "using a cached neutral illustrative title plate; verified season imagery was unavailable.")):
                if any(asset.get("scope") == scope for asset in assets):
                    result["warnings"].append(f"Season {number}: {message}")
        result["seasons"][str(number)] = assets
        result["sources"].extend({"url": asset["source_url"], "title": asset["title"], "season": number}
                                 for asset in assets if asset.get("source_url"))
    if not result["seasons"]:
        raise PipelineError("No researched prior seasons are available for recap visuals.")
    return result
