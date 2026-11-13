"""Conservative season/episode editing with shared source freshness.

Scene analysis (local or Gemini) supplies timing, descriptions and picture
quality, never episode identity. The asset manifest and researched episode map
alone determine source scope. Narration is always covered by real footage: when
a season's fresh footage runs out, the editor reuses its least-used shots, then
borrows another covered season's footage. It never shows a blank title plate.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from movie_pipeline.core import PipelineError

MAX_ANALYSED_SOURCES = 80
MAX_NATIVE_SOURCES = 8
MAX_NATIVE_BEATS = 4
DIVERSITY_SECONDS = 12.0  # one point of score lost per 12 s already used from a clip
SHARP_SHARE = 0.66
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
    source_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

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
            self.source_seconds[str(source)] += float(seconds)

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
    """(season, episode ids, exact rows, same-season generic rows, other-season rows)."""
    season = _current_season(section, usage)
    every = [(int(key), row) for key, row in _rows(assets)
             if _integer(key) is not None and _valid_asset(row, _integer(key), usage)]
    if season is None:
        # Intro and outro: any covered season's footage, general footage first. Showing an
        # episode's clip under orientation narration makes no claim about that episode.
        generic = [row for _, row in every if not _episode_ids(row.get("episode_ids"))]
        specific = [row for _, row in every if _episode_ids(row.get("episode_ids"))]
        return None, [], [], generic, specific
    ids = _requested_episodes(section, paragraph, explicit, season, usage)
    valid = [row for number, row in every if number == season]
    exact = [row for row in valid if ids and _episode_ids(row.get("episode_ids")) and
             set(_episode_ids(row.get("episode_ids"))).issubset(set(ids))]
    generic = [row for row in valid if not _episode_ids(row.get("episode_ids"))]
    others = [row for number, row in every if number != season and not _episode_ids(row.get("episode_ids"))]
    # A different episode's still never becomes generic fallback footage.
    if ids and not exact:
        usage.warn(f"Season {season}: exact current-episode imagery was unavailable; using season-level footage.")
    return season, ids, exact, generic, others


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
            usage.warn(f"Visual analysis stopped at {MAX_ANALYSED_SOURCES} sources; remaining footage uses evenly spaced windows.")
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
            usage.shots[path] = _clean_shots(raw, length) if raw else _even_windows(length)
            if not usage.shots[path]:
                usage.warn(f"Season {season}: no coherent moving shots were found in {Path(path).name}; using evenly spaced windows.")
        except Exception as exc:  # Optional scene analysis must not fail an otherwise valid recap.
            usage.shots[path] = _even_windows(_number(row.get("duration")))
            usage.warn(f"Season {season}: scene analysis was unavailable for {Path(path).name} "
                       f"({type(exc).__name__}); using evenly spaced windows of its footage.")
    return dict(usage.shots)


def _even_windows(length, size=8.0):
    """Plain 8-second windows that skip the opening and closing slates of a clip."""
    start, end = max(1.0, length * 0.05), length * 0.9
    windows, point = [], start
    while end - point >= 2.0:
        windows.append({"start": round(point, 2), "end": round(min(end, point + size), 2), "motion": 0.0})
        point += size
    return windows


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


def _shot_key(source, shot):
    from .gemini_media import scene_key
    return scene_key(source, shot["start"])


def _quality(shot):
    """Gemini picture notes: favour sharp clean shots, avoid logo bugs and darkness."""
    return (_number(shot.get("quality"), 5) - 5) / 2 - (3 if shot.get("watermark") else 0) \
        - (2 if shot.get("dark") else 0)


def _visual_options(rows, paragraph, seconds, usage, exact, preferred=None, *, reuse=False, events=None):
    """Candidate shots for one beat. ``reuse`` allows already used footage, least used first."""
    result = []
    text = paragraph.get("text", "") if isinstance(paragraph, dict) else paragraph
    ranks = {key: index for index, key in enumerate(preferred or [])}
    for row in rows:
        source = str(row["path"])
        match = _relevance(text, str(row.get("title", "")) + " " + str(row.get("description", "")))
        if row["kind"] == "image":
            uses = usage.image_uses[source]
            if uses and not reuse:
                continue
            result.append((100 * exact + 12 - uses * 2 + match,
                           {"row": row, "start": 0.0, "still": True, "image": True}))
            continue
        length = _number(row.get("duration"))
        if source in usage.shots:
            raw = usage.shots[source]
        elif isinstance(row.get("shots"), list):
            raw = _clean_shots(row["shots"], length)  # a clip of only title cards stays unused
        else:
            raw = _even_windows(length)
        for shot in raw:
            if _number(shot.get("motion")) > 30:
                continue  # A fast or unstable source becomes a gentle moving frame instead.
            key = _shot_key(source, shot)
            bonus = (40 - 4 * ranks[key]) if key in ranks else 0
            # A scene linked to the very story event being narrated is the strongest match.
            if events and shot.get("event_id") in events:
                bonus += 30
            description = str(shot.get("description", "")) + " " + str(shot.get("text", ""))
            base = (100 * exact + 20 + match + _relevance(text, description) + bonus + _quality(shot)
                    - min(max(0.0, _number(shot.get("motion"))), 100) / 50
                    - (2 if usage.last_source == source else 0)
                    # Spread the edit across every clip instead of leaning on one.
                    - usage.source_seconds[source] / DIVERSITY_SECONDS)
            if reuse:
                overlap = 1 - usage.fresh(source, shot["start"], shot["end"] - shot["start"])
                score = base - 30 * overlap - (8 if usage.last_source == source else 0)
                if shot["end"] - shot["start"] + .001 >= seconds:
                    result.append((score, {"row": row, "start": shot["start"], "still": False, "image": False,
                                           "event_id": shot.get("event_id")}))
                else:
                    # Too short to run for the whole beat: a gently moving frame from its middle.
                    middle = (shot["start"] + shot["end"]) / 2
                    result.append((score - 5 - 3 * usage.image_uses[source + f"@{middle:.3f}"],
                                   {"row": row, "start": middle, "still": True, "image": False,
                                    "frame_key": source + f"@{middle:.3f}", "event_id": shot.get("event_id")}))
                continue
            for start, end in _free_windows(source, shot["start"], shot["end"], usage):
                if end - start + .001 < seconds:
                    continue
                result.append((base, {"row": row, "start": start, "still": False, "image": False,
                                      "event_id": shot.get("event_id")}))
        # A short coherent shot can provide one gently moving frame. Never freeze
        # an arbitrary trailer end or loop its last shot to fill narration.
        if not reuse and not any(value[1]["row"] is row for value in result):
            for shot in raw:
                middle = (shot["start"] + shot["end"]) / 2
                marker = source + f"@{middle:.3f}"
                if usage.image_uses[marker] == 0:
                    result.append((100 * exact + 5 + match + _quality(shot),
                                   {"row": row, "start": middle, "still": True,
                                    "image": False, "frame_key": marker}))
    return result


def scene_catalogue(section: dict, paragraph, assets: dict, usage: UsageTracker, *, episode_ids=None) -> list[dict]:
    """In-scope scenes with descriptions, for Gemini to rank against one paragraph."""
    _, _, exact, generic, others = _pool(section, paragraph, assets, usage, episode_ids)
    rows = exact or generic or others
    catalogue = []
    for row in rows:
        if row.get("kind") != "video":
            continue
        source = str(row["path"])
        length = _number(row.get("duration"))
        for shot in usage.shots.get(source, _clean_shots(row.get("shots"), length)):
            if not shot.get("description"):
                continue
            catalogue.append({"key": _shot_key(source, shot), "description": str(shot["description"]),
                              "event": str(shot.get("event_summary") or ""),
                              "seconds": shot["end"] - shot["start"], "quality": shot.get("quality", 5),
                              "watermark": bool(shot.get("watermark")),
                              "freshness": usage.fresh(source, shot["start"], shot["end"] - shot["start"])})
    # Fresh footage first so the model sees what is still unused.
    catalogue.sort(key=lambda row: -row["freshness"])
    return catalogue


def plan_visuals(section: dict, paragraph, seconds: float, assets: dict, usage: UsageTracker,
                 *, episode_ids=None, preferred=None, events=None) -> list[dict]:
    """Fill all narration with real footage, normally coherent 6–10 second cuts.

    Order of preference: this paragraph's episode, fresh same-season footage,
    fresh footage from another covered season, then the least-used footage again.
    ``preferred`` is Gemini's ranking of scene keys for this paragraph.
    """
    seconds = _number(seconds)
    if seconds <= 0:
        raise PipelineError("A recap paragraph must have a positive measured duration.")
    season, ids, exact, generic, others = _pool(section, paragraph, assets, usage, episode_ids)
    count = max(1, math.ceil(seconds / 10.0))
    beat_seconds = seconds / count
    plan = []
    for index in range(count):
        wanted = seconds - sum(shot["duration"] for shot in plan) if index == count - 1 else beat_seconds
        borrowed = False
        options = _visual_options(exact, paragraph, wanted, usage, True, preferred, events=events)
        if not options:
            options = _visual_options(generic, paragraph, wanted, usage, False, preferred, events=events)
        if not options and others:
            options = _visual_options(others, paragraph, wanted, usage, False, preferred, events=events)
            borrowed = bool(options)
        if not options:
            options = _visual_options(exact + generic, paragraph, wanted, usage, False, preferred, reuse=True, events=events)
            if options and season is not None:
                usage.warn(f"Season {season}: fresh footage ran out, so some shots are reused.")
        if not options and others:
            options = _visual_options(others, paragraph, wanted, usage, False, preferred, reuse=True, events=events)
            borrowed = bool(options)
        if not options:
            raise PipelineError("No footage or imagery is available for this recap; "
                                "no blank plate is used. Add or refresh YOUTUBE_COOKIES and run again.")
        _, chosen = max(options, key=lambda item: item[0])
        row = chosen["row"]
        borrowed = borrowed and season is not None
        if borrowed:
            usage.warn(f"Season {season}: footage from another covered season fills part of its narration.")
        source = str(row["path"])
        row_ids = _episode_ids(row.get("episode_ids"))
        if season is not None and row_ids and (not ids or not set(row_ids).issubset(set(ids))):
            raise PipelineError("An episode-specific asset escaped the current paragraph's scope.")
        shot = {"source": source, "path": source, "start": chosen["start"], "duration": wanted,
                "still": chosen["still"], "image": chosen["image"], "freeze": 0.0,
                "still_motion": MOTIONS[usage.beat % len(MOTIONS)],
                "season": _integer(row.get("season")) if (borrowed or season is None) else season,
                "episode_id": row_ids[0] if len(row_ids) == 1 else None, "episode_ids": row_ids,
                "scope": row.get("scope", "season"), "illustrative": bool(row.get("illustrative")),
                "source_url": row.get("source_url", ""), "event_id": chosen.get("event_id"),
                "selection": "current-episode" if row_ids else
                             "other-season-fallback" if borrowed else
                             "orientation-footage" if season is None else "same-season-generic"}
        plan.append(shot)
        usage.beat += 1
        usage.last_source = source
        if shot["still"]:
            usage.image_uses[chosen.get("frame_key", source)] += 1
            if not chosen["image"]:
                usage.reserve(source, max(0.0, chosen["start"] - .1), .2)
        else:
            usage.reserve(source, shot["start"], wanted)
    return plan


def choose_resolution(assets: dict, try_4k: bool) -> tuple[int, int, list[str]]:
    """Use actual scoped source detail; report any enlargement and quality limits."""
    ceiling = 2160 if try_4k else 1080
    rows = [row for _, row in _rows(assets)]
    dimensions = [(int(_number(row.get("width"))), int(_number(row.get("height")))) for row in rows]
    dimensions = [(width, height) for width, height in dimensions if width > 0 and height > 0]
    if not dimensions:
        return 1280, 720, ["No verified photographic source detail was available; rendering at 720p."]
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
                      episode_ids=None, analyser=None, events=None) -> dict | None:
    """Optional complete utterances, restricted to verified current-episode clips.

    At most four unique sources receive ASR in one recap. No recognizer, complete
    phrase, fresh coherent scene, or lexical connection means no native insert.
    """
    season, ids, exact, generic, _ = _pool(section, paragraph, assets, usage, episode_ids)
    section_id = str(section.get("id", ""))
    if (season is None or len(ids) != 1 or not callable(analyser)
            or section.get("kind") not in {"story", "ending"}
            or section_id in usage.native_sections or len(usage.native_sections) >= MAX_NATIVE_BEATS):
        return None
    query = paragraph.get("text", "") if isinstance(paragraph, dict) else str(paragraph)
    events = set(events or [])
    candidates = []
    for row in exact:
        if row.get("kind") == "video" and _episode_ids(row.get("episode_ids")) == ids:
            shots = usage.shots.get(str(row["path"]), _clean_shots(row.get("shots"), _number(row.get("duration"))))
            candidates.append((row, shots, False))
    # Season clips whose scenes Gemini linked to this paragraph's episode can speak too,
    # but only inside those linked scenes.
    for row in generic:
        if row.get("kind") != "video":
            continue
        shots = [shot for shot in usage.shots.get(str(row["path"]), _clean_shots(row.get("shots"), _number(row.get("duration"))))
                 if str(shot.get("episode_id") or "") == ids[0]]
        if shots:
            candidates.append((row, shots, True))
    for row, shots, linked in candidates:
        source = str(row["path"])
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
            on_event = any(shot.get("event_id") in events for shot in shots
                           if shot["start"] <= group[0]["start"] and shot["end"] >= group[-1]["end"])
            # Words in common, or a scene linked to the very event being narrated.
            if not on_event and (len(shared) < 2 or len(shared) / max(1, len(_terms(text))) < .25):
                continue
            start, end = group[0]["start"] - .14, group[-1]["end"] + .20
            if start < 0 or end > _number(row.get("duration")) or not 3 <= end - start <= 12:
                continue
            if not any(shot["start"] <= start and shot["end"] >= end for shot in shots):
                continue
            if usage.fresh(source, start, end - start) < .98:
                continue
            row_ids = list(ids) if linked else _episode_ids(row.get("episode_ids"))
            clip = {"source": source, "path": source, "start": start, "end": end, "duration": end - start,
                    "text": text, "cues": [dict(cue) for cue in group], "season": season,
                    "episode_ids": row_ids, "episode_id": row_ids[0] if len(row_ids) == 1 else None,
                    "scope": row.get("scope", "episode"), "illustrative": False,
                    "source_url": row.get("source_url", ""), "native_verified": True, "muted": False}
            usage.reserve(source, start, end - start)
            usage.native_sections.add(section_id)
            return clip
    return None


def prefer_sharp_sources(assets: dict) -> tuple[dict, list[str]]:
    """Leave out clearly low-resolution clips when at least three sharp ones exist.

    One 480p upload otherwise pulls the whole recap down to its resolution.
    """
    seasons = assets.get("seasons", {}) if isinstance(assets, dict) else {}
    videos = [row for rows in seasons.values() if isinstance(rows, list) for row in rows
              if isinstance(row, dict) and row.get("kind") == "video"]
    heights = sorted((int(_number(row.get("height"))) for row in videos), reverse=True)
    if len(heights) < 4:
        return assets, []
    floor = heights[2] * SHARP_SHARE
    kept = {key: [row for row in rows if not (isinstance(row, dict) and row.get("kind") == "video"
                                               and _number(row.get("height")) < floor)]
            for key, rows in seasons.items()}
    # Never empty a season to gain sharpness.
    for key, rows in seasons.items():
        if not kept[key]:
            kept[key] = rows
    dropped = sum(len(rows) for rows in seasons.values()) - sum(len(rows) for rows in kept.values())
    warnings = [f"{dropped} low-resolution clip(s) were left out so the recap keeps sharp footage."] if dropped else []
    return {**assets, "seasons": kept}, warnings
