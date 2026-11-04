"""Conservative season/episode editing with shared source freshness.

Scene analysis supplies timing and picture quality, never episode identity. The
asset manifest and researched episode map alone determine source scope.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from movie_pipeline.core import PipelineError

MAX_ANALYSED_SOURCES = 6
MAX_NATIVE_SOURCES = 4
MOTIONS = ("zoom_in", "zoom_out")
STOPWORDS = {"the", "and", "that", "this", "with", "from", "into", "their", "they", "them",
             "then", "have", "has", "was", "were", "are", "for", "but", "not", "his", "her",
             "she", "you", "your", "our", "will", "would", "could", "should", "must", "can",
             "what", "when", "where", "who", "how", "all", "one", "just", "now", "been",
             "had", "more", "some", "out", "too", "its", "about", "there", "here"}


def _number(value, default=0.0):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return default
    return number if math.isfinite(number) else default


def _integer(value):
    number = _number(value, -1)
    return int(number) if number >= 1 and number.is_integer() else None


def _episode_ids(value):
    return list(dict.fromkeys(str(item) for item in value
                              if isinstance(item, (str, int)) and str(item))) if isinstance(value, list) else []


def _terms(text):
    return {word for word in re.findall(r"[a-z0-9]{3,}", str(text).casefold()) if word not in STOPWORDS}


def _relevance(text, description):
    query, found = _terms(text), _terms(description)
    return len(query & found) / max(1, len(query))


@dataclass
class UsageTracker:
    """One instance per recap; keep it across every paragraph and native insert."""
    upcoming_season: int
    allowed_seasons: set[int] | None = None
    episode_seasons: dict[str, int] | None = None
    warnings: list[str] = field(default_factory=list)
    intervals: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    image_uses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    shots: dict[str, list[dict]] = field(default_factory=dict)
    analysed_sources: set[str] = field(default_factory=set)
    native_sources: set[str] = field(default_factory=set)
    native_cues: dict[str, list[dict]] = field(default_factory=dict)
    native_sections: set[str] = field(default_factory=set)
    last_source: str | None = None
    beat: int = 0

    def __post_init__(self):
        incoming = _integer(self.upcoming_season)
        if incoming is None or incoming < 2:
            raise PipelineError("Recap editing needs an incoming season greater than one.")
        self.upcoming_season = incoming
        self.allowed_seasons = (set(range(1, incoming)) if self.allowed_seasons is None
                                else set(self.allowed_seasons))
        if any(_integer(season) is None or not 0 < season < incoming for season in self.allowed_seasons):
            raise PipelineError("Recap editing scope includes an incoming or invalid season.")
        if self.episode_seasons is not None:
            self.episode_seasons = {str(k): v for k, v in self.episode_seasons.items()}

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def reserve(self, source, start, seconds):
        if seconds > 0:
            self.intervals[str(source)].append((float(start), float(start + seconds)))

    def fresh(self, source, start, seconds):
        end = start + seconds
        overlap = sum(max(0, min(end, high) - max(start, low))
                      for low, high in self.intervals.get(str(source), []))
        return max(0.0, 1 - overlap / max(seconds, .001))


def _rows(assets):
    seasons = assets.get("seasons", {}) if isinstance(assets, dict) else {}
    for key, rows in seasons.items() if isinstance(seasons, dict) else []:
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield key, row


def _valid_asset(row, season, usage):
    actual = _integer(row.get("season"))
    if (actual != season or season not in usage.allowed_seasons
            or not 0 < season < usage.upcoming_season
            or not isinstance(row.get("path"), (str, Path)) or not str(row["path"])):
        return False
    ids = _episode_ids(row.get("episode_ids", []))
    if row.get("scope") == "episode" and not ids:
        return False
    if usage.episode_seasons is not None and any(
            usage.episode_seasons.get(episode_id) != season for episode_id in ids):
        return False
    return row.get("kind") in {"video", "image"}


def _current_season(section, usage):
    value = section.get("season")
    if value is None:
        return None
    season = _integer(value)
    if season not in usage.allowed_seasons or season is None or season >= usage.upcoming_season:
        raise PipelineError("A recap section refers to an incoming or out-of-scope season.")
    return season


def _requested_episodes(section, paragraph, explicit, season, usage):
    value = explicit if explicit is not None else (
        paragraph.get("episode_ids", section.get("episode_ids", []))
        if isinstance(paragraph, dict) else section.get("episode_ids", []))
    ids = _episode_ids(value)
    if usage.episode_seasons is not None and any(
            usage.episode_seasons.get(episode_id) != season for episode_id in ids):
        raise PipelineError("A recap paragraph's episode does not belong to its current season.")
    return ids


def _pool(section, paragraph, assets, usage, explicit):
    season = _current_season(section, usage)
    if season is None:
        return None, [], [], []
    ids = _requested_episodes(section, paragraph, explicit, season, usage)
    valid = [row for key, row in _rows(assets)
             if _integer(key) == season and _valid_asset(row, season, usage)]
    exact = [row for row in valid if ids and _episode_ids(row.get("episode_ids")) and
             set(_episode_ids(row.get("episode_ids"))).issubset(set(ids))]
    generic = [row for row in valid if not _episode_ids(row.get("episode_ids"))]
    # A different episode's still never becomes generic fallback footage.
    if ids and not exact:
        usage.warn(f"Season {season}: exact current-episode imagery was unavailable; using season-level imagery or an explicit neutral plate.")
    return season, ids, exact, generic


def ensure_neutral_plate(work: Path, show_title: str, season: int | None) -> dict:
    """An explicit illustrative fallback with no inferred actors, locations or plot."""
    from PIL import Image, ImageDraw, ImageFont
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256((str(show_title) + ":" + str(season)).encode()).hexdigest()[:16]
    path = work / f"recap-neutral-{name}.jpg"
    if not path.is_file():
        image = Image.new("RGB", (1280, 720), "#101b26")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 1249, 689), outline="#80cbc4", width=3)

        def face(size):
            for font in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
                try:
                    return ImageFont.truetype(font, size)
                except OSError:
                    pass
            return ImageFont.load_default()

        title = " ".join(str(show_title).split())[:160]
        size = 72
        while size > 18 and draw.textbbox((0, 0), title, font=face(size))[2] > 1100:
            size -= 2
        draw.text((85, 240), title, font=face(size), fill="white")
        draw.text((85, 350), f"Season {season}" if season else "The story so far",
                  font=face(48), fill="#80cbc4")
        draw.text((85, 595), "ILLUSTRATIVE TITLE PLATE", font=face(22), fill="#b9c8d7")
        image.save(path, "JPEG", quality=94)
    return {"path": str(path.resolve()), "kind": "image", "width": 1280, "height": 720,
            "season": season, "episode_ids": [], "scope": "neutral-title-card",
            "illustrative": True, "source_url": "", "title": "Illustrative recap title plate"}


def _neutral(assets, season):
    row = assets.get("_neutral") if isinstance(assets, dict) else None
    if (not isinstance(row, dict) or row.get("kind") != "image"
            or row.get("scope") != "neutral-title-card" or not row.get("illustrative")
            or not row.get("path") or row.get("season") != season):
        raise PipelineError("A scoped explicit neutral plate is required when no safe visual is available.")
    return row


def _clean_shots(rows, length):
    result = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("text_card"):
            continue
        start, end = max(0.0, _number(row.get("start"), -1)), _number(row.get("end"), -1)
        if start < end <= length + .05 and end - start >= .3:
            result.append({**row, "start": start, "end": min(end, length)})
    return result


def analyse_assets(assets: dict, work: Path, usage: UsageTracker, *, analyser=None) -> dict:
    """Analyse at most six prior-season videos, without downloading matching models.

    The local cut/motion analyser is sufficient. Optional matching descriptions in
    cached shots are soft preferences; an absent/weak match is never a failure.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for key, row in _rows(assets):
        season = _integer(key)
        if season is None or not _valid_asset(row, season, usage) or row.get("kind") != "video":
            continue
        path = str(row["path"])
        if path in usage.analysed_sources:
            continue
        if len(usage.analysed_sources) >= MAX_ANALYSED_SOURCES:
            usage.warn("Visual analysis stopped at six sources; remaining footage uses conservative timing or neutral plates.")
            break
        usage.analysed_sources.add(path)
        try:
            length = _number(row.get("duration"))
            raw = row.get("shots")
            if not isinstance(raw, list):
                if analyser is None:
                    from movie_pipeline.shots import analyse
                    analyser = analyse
                token = hashlib.sha256(path.encode()).hexdigest()[:16]
                raw = analyser(Path(path), work / f"recap-shots-{token}.json")
            usage.shots[path] = _clean_shots(raw, length)
            if not usage.shots[path]:
                usage.warn(f"Season {season}: no coherent moving shots were found in {Path(path).name}; using moving stills or neutral plates.")
        except Exception as exc:  # Optional scene analysis must not fail an otherwise valid recap.
            usage.shots[path] = []
            usage.warn(f"Season {season}: optional scene analysis was unavailable ({type(exc).__name__}); using moving stills or neutral plates.")
    return dict(usage.shots)


def _free_windows(source, start, end, usage):
    windows = [(start, end)]
    for low, high in sorted(usage.intervals.get(source, [])):
        next_windows = []
        for left, right in windows:
            if high <= left or low >= right:
                next_windows.append((left, right))
            else:
                if low - .08 > left:
                    next_windows.append((left, low - .08))
                if high + .08 < right:
                    next_windows.append((high + .08, right))
        windows = next_windows
    return windows


def _visual_options(rows, paragraph, seconds, usage, exact):
    result = []
    text = paragraph.get("text", "") if isinstance(paragraph, dict) else paragraph
    for row in rows:
        source = str(row["path"])
        match = _relevance(text, str(row.get("title", "")) + " " + str(row.get("description", "")))
        if row["kind"] == "image":
            uses = usage.image_uses[source]
            result.append((100 * exact + 12 - uses * 2 + match,
                           {"row": row, "start": 0.0, "still": True, "image": True}))
            continue
        length = _number(row.get("duration"))
        raw = usage.shots.get(source, _clean_shots(row.get("shots"), length))
        # Unanalysed clips are not guessed to be one coherent scene.
        for shot in raw:
            if _number(shot.get("motion")) > 30:
                continue  # A fast or unstable source becomes a gentle moving frame instead.
            for start, end in _free_windows(source, shot["start"], shot["end"], usage):
                if end - start + .001 < seconds:
                    continue
                motion = max(0.0, _number(shot.get("motion")))
                description = str(shot.get("description", "")) + " " + str(shot.get("text", ""))
                score = (100 * exact + 20 + match + _relevance(text, description)
                         - min(motion, 100) / 50 - (2 if usage.last_source == source else 0))
                result.append((score, {"row": row, "start": start, "still": False, "image": False}))
        # A short coherent shot can provide one gently moving frame. Never freeze
        # an arbitrary trailer end or loop its last shot to fill narration.
        if not any(value[1]["row"] is row for value in result):
            for shot in raw:
                middle = (shot["start"] + shot["end"]) / 2
                marker = source + f"@{middle:.3f}"
                if usage.image_uses[marker] == 0:
                    result.append((100 * exact + 5 + match,
                                   {"row": row, "start": middle, "still": True,
                                    "image": False, "frame_key": marker}))
    return result


def plan_visuals(section: dict, paragraph, seconds: float, assets: dict, usage: UsageTracker,
                 *, episode_ids=None) -> list[dict]:
    """Fill all narration, normally with coherent 6–10 second cuts or moving stills."""
    seconds = _number(seconds)
    if seconds <= 0:
        raise PipelineError("A recap paragraph must have a positive measured duration.")
    season, ids, exact, generic = _pool(section, paragraph, assets, usage, episode_ids)
    count = max(1, math.ceil(seconds / 10.0))
    beat_seconds = seconds / count
    plan = []
    for index in range(count):
        wanted = seconds - sum(shot["duration"] for shot in plan) if index == count - 1 else beat_seconds
        options = _visual_options(exact, paragraph, wanted, usage, True)
        if not options:
            options = _visual_options(generic, paragraph, wanted, usage, False)
        if options:
            _, chosen = max(options, key=lambda item: item[0])
            row = chosen["row"]
        else:
            row = _neutral(assets, season)
            chosen = {"row": row, "start": 0.0, "still": True, "image": True}
            if season is not None:
                usage.warn(f"Season {season}: safe fresh imagery was exhausted or unavailable; an explicit neutral title plate fills the remaining narration.")
        source = str(row["path"])
        row_ids = _episode_ids(row.get("episode_ids"))
        if row_ids and (not ids or not set(row_ids).issubset(set(ids))):
            raise PipelineError("An episode-specific asset escaped the current paragraph's scope.")
        shot = {"source": source, "path": source, "start": chosen["start"], "duration": wanted,
                "still": chosen["still"], "image": chosen["image"], "freeze": 0.0,
                "still_motion": MOTIONS[usage.beat % len(MOTIONS)], "season": season,
                "episode_id": row_ids[0] if len(row_ids) == 1 else None, "episode_ids": row_ids,
                "scope": row.get("scope", "season"), "illustrative": bool(row.get("illustrative")),
                "source_url": row.get("source_url", ""),
                "selection": "current-episode" if row_ids else
                             "neutral-plate" if row.get("scope") == "neutral-title-card" else "same-season-generic"}
        plan.append(shot)
        usage.beat += 1
        usage.last_source = source
        if shot["still"]:
            if usage.image_uses[chosen.get("frame_key", source)] and row.get("scope") != "neutral-title-card":
                usage.warn(f"Season {season}: limited verified imagery requires a repeated moving still.")
            usage.image_uses[chosen.get("frame_key", source)] += 1
            if not chosen["image"]:
                usage.reserve(source, max(0.0, chosen["start"] - .1), .2)
        else:
            usage.reserve(source, shot["start"], wanted)
    return plan


def choose_resolution(assets: dict, try_4k: bool) -> tuple[int, int, list[str]]:
    """Use actual scoped source detail; report any enlargement and quality limits."""
    ceiling = 2160 if try_4k else 1080
    rows = [row for _, row in _rows(assets) if row.get("scope") != "neutral-title-card"]
    dimensions = [(int(_number(row.get("width"))), int(_number(row.get("height")))) for row in rows]
    dimensions = [(width, height) for width, height in dimensions if width > 0 and height > 0]
    if not dimensions:
        return 1280, 720, ["No verified photographic source detail was available; rendering neutral plates at 720p."]
    sizes = ((3840, 2160), (2560, 1440), (1920, 1080), (1280, 720),
             (854, 480), (640, 360), (426, 240), (256, 144))
    selected = None
    for width, height in sizes:
        if height > ceiling:
            continue
        native = sum(sw >= width and sh >= height for sw, sh in dimensions)
        if native == len(dimensions) or (
                native / len(dimensions) >= .6 and
                all(height / min(sh, sw * 9 / 16) <= 1.5 for sw, sh in dimensions)):
            selected = width, height
            break
    if selected is None:
        detail = min(min(sw / 16, sh / 9) for sw, sh in dimensions)
        selected = max(2, int(detail * 16) // 2 * 2), max(2, int(detail * 9) // 2 * 2)
    width, height = selected
    warnings = []
    if any(sw < width or sh < height for sw, sh in dimensions):
        warnings.append(f"Some recap sources are enlarged to the selected {width}x{height} canvas; additional detail is not invented.")
    if height < min(ceiling, 1080):
        warnings.append(f"Source quality limits this recap to {width}x{height}.")
    elif try_4k and height < 2160:
        warnings.append(f"Verified source detail supports {width}x{height}; 4K was not forced.")
    return width, height, warnings


def _phrases(cues):
    clean = []
    expanded = []
    for cue in cues if isinstance(cues, list) else []:
        if not isinstance(cue, dict):
            continue
        words = cue.get("words")
        if isinstance(words, list) and words:
            expanded.extend({**word, "text": word.get("word", word.get("text", ""))}
                            for word in words if isinstance(word, dict))
        else:
            expanded.append(cue)
    for cue in expanded:
        start, end = _number(cue.get("start"), -1), _number(cue.get("end"), -1)
        text = str(cue.get("text", "")).strip()
        if start >= 0 and end > start and text:
            clean.append({**cue, "start": start, "end": end, "text": text})
    clean.sort(key=lambda cue: cue["start"])
    group = []
    for cue in clean:
        if group and cue["start"] - group[-1]["end"] >= .8:
            group = []  # An unfinished phrase before a pause is not a complete beat.
        group.append(cue)
        if re.search(r"[.!?][\"'’”)]?$", cue["text"]):
            yield group
            group = []


def plan_source_audio(section: dict, paragraph, assets: dict, usage: UsageTracker, *,
                      episode_ids=None, analyser=None) -> dict | None:
    """Optional complete utterances, restricted to verified current-episode clips.

    At most four unique sources receive ASR in one recap. No recognizer, complete
    phrase, fresh coherent scene, or lexical connection means no native insert.
    """
    season, ids, exact, _ = _pool(section, paragraph, assets, usage, episode_ids)
    section_id = str(section.get("id", ""))
    if (season is None or len(ids) != 1 or not callable(analyser)
            or section.get("kind") not in {"story", "ending"}
            or section_id in usage.native_sections):
        return None
    query = paragraph.get("text", "") if isinstance(paragraph, dict) else str(paragraph)
    for row in exact:
        if row.get("kind") != "video" or _episode_ids(row.get("episode_ids")) != ids:
            continue
        source = str(row["path"])
        shots = usage.shots.get(source, _clean_shots(row.get("shots"), _number(row.get("duration"))))
        if not shots:
            continue
        if source not in usage.native_sources:
            if len(usage.native_sources) >= MAX_NATIVE_SOURCES:
                return None
            usage.native_sources.add(source)
            try:
                usage.native_cues[source] = analyser(Path(source)) or []
            except Exception as exc:  # Source audio is optional, including ASR failures.
                usage.native_cues[source] = []
                usage.warn(f"Season {season}: optional source-audio analysis was skipped ({type(exc).__name__}).")
        for group in _phrases(usage.native_cues.get(source, [])):
            text = " ".join(cue["text"] for cue in group)
            shared = _terms(query) & _terms(text)
            if len(shared) < 2 or len(shared) / max(1, len(_terms(text))) < .25:
                continue
            start, end = group[0]["start"] - .14, group[-1]["end"] + .20
            if start < 0 or end > _number(row.get("duration")) or not 3 <= end - start <= 12:
                continue
            if not any(shot["start"] <= start and shot["end"] >= end for shot in shots):
                continue
            if usage.fresh(source, start, end - start) < .98:
                continue
            row_ids = _episode_ids(row.get("episode_ids"))
            clip = {"source": source, "path": source, "start": start, "end": end, "duration": end - start,
                    "text": text, "cues": [dict(cue) for cue in group], "season": season,
                    "episode_ids": row_ids, "episode_id": row_ids[0] if len(row_ids) == 1 else None,
                    "scope": row.get("scope", "episode"), "illustrative": False,
                    "source_url": row.get("source_url", ""), "native_verified": True, "muted": False}
            usage.reserve(source, start, end - start)
            usage.native_sections.add(section_id)
            return clip
    return None
