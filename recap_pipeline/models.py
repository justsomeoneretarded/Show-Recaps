"""Validated immutable request and the final boundary between research and rendering."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from collections.abc import Mapping

from movie_pipeline.core import PipelineError, normalized


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise PipelineError(f"{label} must be a whole number from {low} to {high}.")
    return value


def _minutes(value, label):
    if value == "Auto":
        return value
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(r"(?:[1-9]|[12][0-9]|30)", value):
        raise PipelineError(f"{label} must be Auto or a whole number from 1 to 30.")
    return value


@dataclass(frozen=True)
class RecapSpec:
    show_title: str
    premiere_year: int
    upcoming_season: int
    brief: str = ""
    narrator: str = "Warguy"
    tts_model: str = "original"
    season_minutes: str = "Auto"
    series_minutes: str = "Auto"
    try_4k: bool = True

    def __post_init__(self):
        if (not isinstance(self.show_title, str) or not self.show_title.strip()
                or len(self.show_title) > 300 or "\n" in self.show_title or "\r" in self.show_title):
            raise PipelineError("Provide one show title, up to 300 characters.")
        object.__setattr__(self, "show_title", self.show_title.strip())
        _integer(self.premiere_year, 1900, 2100, "Premiere year")
        _integer(self.upcoming_season, 2, 100, "Upcoming season")
        if not isinstance(self.brief, str) or len(self.brief) > 4000:
            raise PipelineError("The recap brief must be text up to 4000 characters.")
        if self.narrator not in ("Warguy", "Kit Laser", "Female"):
            raise PipelineError("Choose Warguy, Kit Laser, or Female as narrator.")
        if self.tts_model not in ("original", "turbo", "nano"):
            raise PipelineError("Choose original, turbo, or nano as speech model.")
        for field in ("season_minutes", "series_minutes"):
            object.__setattr__(self, field, _minutes(getattr(self, field), field.replace("_", " ").title()))
        if type(self.try_4k) is not bool:
            raise PipelineError("Try 4K must be a boolean.")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env

        def integer(name):
            value = env.get(name, "").strip()
            if not re.fullmatch(r"[0-9]{1,4}", value):
                raise PipelineError(f"{name} must be a whole number.")
            return int(value)

        four_k = env.get("TRY_4K", "true").strip().lower()
        if four_k not in ("true", "false"):
            raise PipelineError("TRY_4K must be true or false.")
        return cls(
            show_title=env.get("SHOW_TITLE", ""),
            premiere_year=integer("SHOW_YEAR"), upcoming_season=integer("UPCOMING_SEASON"),
            brief=env.get("RECAP_BRIEF", ""), narrator=env.get("NARRATOR", "Warguy"),
            tts_model=env.get("TTS_MODEL", "original"),
            season_minutes=env.get("SEASON_MINUTES", "Auto"),
            series_minutes=env.get("SERIES_MINUTES", "Auto"), try_4k=four_k == "true",
        )

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise PipelineError("The saved recap request must be an object.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise PipelineError("The saved recap request has missing or unknown fields.") from exc

    def to_dict(self):
        return asdict(self)


def recap_seasons(kind, spec):
    if kind == "season":
        return [spec.upcoming_season - 1]
    if kind == "series":
        return list(range(1, spec.upcoming_season))
    raise PipelineError("Recap kind must be season or series.")


def _list(value, label, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise PipelineError(f"{label} must be a{' nonempty' if nonempty else ''} list.")
    return value


def _ids(value, label, *, nonempty=False):
    rows = _list(value, label, nonempty=nonempty)
    if any(not isinstance(item, str) or not item.strip() for item in rows) or len(rows) != len(set(rows)):
        raise PipelineError(f"{label} must contain unique nonempty text IDs.")
    return set(rows)


# Spoken season numbers occur naturally in narration as well as digits.
_SEASON_WORDS = {word: number for number, word in enumerate(
    ("one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(), 1)}
_SEASON_NUMBER = re.compile(
    r"\bseason\s+(\d+|" + "|".join(_SEASON_WORDS) + r")\b", re.I)
_CUTOFF_PREFIX = re.compile(
    r"\b(?:before|ahead\s+of)\s+(?:(?:the\s+)?(?:incoming|upcoming)\s+)?$", re.I)


def narration_within_scope(text, upcoming):
    """Allow an incoming-season cutoff, never incoming-season plot references.

    This checks explicit scope, not the truth of prose. The factual edit still
    checks every narrated claim against its cited evidence.
    """
    if any(int(number) >= upcoming
           for number in re.findall(r"\bS(\d+)E\d+\b", text, re.I)):
        return False
    for match in _SEASON_NUMBER.finditer(text):
        label = match.group(1).lower()
        number = int(label) if label.isdigit() else _SEASON_WORDS[label]
        if number >= upcoming:
            # "Before season N" locates the end of the recap's scope. It does
            # not grant permission to include another future-season reference.
            if number != upcoming or not _CUTOFF_PREFIX.search(text[:match.start()]):
                return False
    return True


def validate_script_scope(script, spec, bundle):
    """Reject crossed identities, future seasons and uncited story before TTS.

    This validates provenance and structure. It does not claim to fact-check prose
    or impose a word quota; gaps remain visible in the script's coverage warnings.
    """
    if not isinstance(script, dict) or not isinstance(bundle, dict):
        raise PipelineError("The recap script and research bundle must be objects.")
    show = bundle.get("show")
    if (not isinstance(show, dict) or type(show.get("id")) is not int or show["id"] < 1
            or not isinstance(show.get("title"), str)
            or normalized(show["title"]) != normalized(spec.show_title)
            or type(show.get("year")) is not int or show["year"] != spec.premiere_year):
        raise PipelineError("Research identity does not match the requested show and premiere year.")
    if (not isinstance(script.get("show_title"), str)
            or normalized(script["show_title"]) != normalized(spec.show_title)
            or type(script.get("upcoming_season")) is not int
            or script["upcoming_season"] != spec.upcoming_season):
        raise PipelineError("Script identity or incoming season does not match the saved request.")
    expected = recap_seasons(script.get("kind"), spec)
    declared = _list(script.get("seasons"), "Script seasons", nonempty=True)
    if any(type(number) is not int for number in declared) or declared != expected:
        raise PipelineError("Script seasons must exactly match the requested recap scope.")

    source_map = {}
    for source in _list(bundle.get("sources"), "Research sources", nonempty=True):
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"]:
            raise PipelineError("Research sources require nonempty IDs.")
        if source["id"] in source_map:
            raise PipelineError("Research source IDs must be unique.")
        number = source.get("season")
        if number is not None and (type(number) is not int or not 1 <= number < spec.upcoming_season):
            raise PipelineError("Research includes an incoming or later-season source.")
        source_map[source["id"]] = source

    episodes = {}
    seen_seasons = set()
    for season in _list(bundle.get("seasons"), "Research seasons", nonempty=True):
        if not isinstance(season, dict):
            raise PipelineError("Research seasons must be objects.")
        number = season.get("number")
        if type(number) is not int or not 1 <= number < spec.upcoming_season or number in seen_seasons:
            raise PipelineError("Research includes a duplicate, incoming or later season.")
        seen_seasons.add(number)
        for episode in _list(season.get("episodes"), "Research episodes", nonempty=True):
            if (not isinstance(episode, dict) or type(episode.get("season")) is not int
                    or episode["season"] != number or type(episode.get("number")) is not int
                    or episode["number"] < 1):
                raise PipelineError("Research episode season/number is invalid.")
            identity = episode.get("id")
            if not isinstance(identity, (str, int)) or isinstance(identity, bool) or not str(identity):
                raise PipelineError("Research episode IDs must be nonempty.")
            identity = str(identity)
            if identity in episodes:
                raise PipelineError("Research episode IDs must be unique.")
            episodes[identity] = episode
    if not set(expected).issubset(seen_seasons):
        raise PipelineError("Research is missing a season required by this recap.")

    allowed = {identity for identity, episode in episodes.items() if episode["season"] in expected}
    scoped_sources = {identity for identity, source in source_map.items()
                      if source.get("season") is None or source.get("season") in expected}
    cited_sources = _ids(script.get("source_ids"), "Script source IDs", nonempty=True)
    if not cited_sources.issubset(scoped_sources):
        raise PipelineError("Script sources include unsupported or out-of-scope evidence.")
    for field in ("title", "description"):
        if not isinstance(script.get(field), str) or not script[field].strip():
            raise PipelineError(f"Script {field} must be nonempty text.")

    sections = _list(script.get("sections"), "Script sections", nonempty=True)
    section_ids = set()
    story_seasons = set()
    ending_ids = set()
    for section in sections:
        if not isinstance(section, dict):
            raise PipelineError("Script sections must be objects.")
        identity = section.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in section_ids:
            raise PipelineError("Script section IDs must be unique nonempty text.")
        section_ids.add(identity)
        paragraphs = _list(section.get("paragraphs"), "Section paragraphs", nonempty=True)
        if any(not isinstance(text, str) or not text.strip() for text in paragraphs):
            raise PipelineError("Every narration paragraph must contain text.")
        for text in paragraphs:
            if not narration_within_scope(text, spec.upcoming_season):
                raise PipelineError("Narration explicitly references incoming or later-season plot.")
        delivery = section.get("delivery")
        if delivery is not None and (not isinstance(delivery, list) or len(delivery) != len(paragraphs)
                or any(label not in ("normal", "calm", "hype", "dramatic") for label in delivery)):
            raise PipelineError("Section deliveries must match its paragraphs.")
        kind, number = section.get("kind"), section.get("season")
        episode_ids = _ids(section.get("episode_ids"), "Section episode IDs")
        event_ids = _ids(section.get("event_ids"), "Section event IDs")
        if kind in ("intro", "outro"):
            if number is not None or episode_ids or event_ids:
                raise PipelineError("Intro/outro sections cannot smuggle story evidence outside scope.")
            continue
        if kind not in ("story", "ending") or type(number) is not int or number not in expected:
            raise PipelineError("Story sections must belong to a covered season.")
        if not episode_ids or not event_ids or not episode_ids.issubset(allowed):
            raise PipelineError("Story sections require supported, in-scope episode and event IDs.")
        for identity in episode_ids:
            episode = episodes[identity]
            refs = _ids(episode.get("sources"), "Episode source IDs", nonempty=True)
            if (not isinstance(episode.get("summary"), str) or not episode["summary"].strip()
                    or not refs.issubset(source_map) or not refs.intersection(cited_sources)
                    or episode["season"] > number):
                raise PipelineError("Narration has no matching sourced episode evidence in this season's scope.")
            if any(source_map[ref].get("season") not in (None, episode["season"]) for ref in refs):
                raise PipelineError("Episode evidence crosses seasons.")
        for field, parent_ids in (("paragraph_episode_ids", episode_ids), ("paragraph_event_ids", event_ids)):
            if field in section:
                groups = _list(section[field], field)
                if len(groups) != len(paragraphs):
                    raise PipelineError("Paragraph evidence must align with narration.")
                for group in groups:
                    if not _ids(group, field, nonempty=True).issubset(parent_ids):
                        raise PipelineError("Paragraph evidence is outside the section's supported scope.")
        if kind == "story":
            # Earlier-season context is valid, but cannot substitute for this
            # season's own story. Count actual paragraph citations when present.
            story_episode_ids = (set().union(*map(set, section["paragraph_episode_ids"]))
                                 if "paragraph_episode_ids" in section else episode_ids)
            if any(episodes[identity]["season"] == number for identity in story_episode_ids):
                story_seasons.add(number)
        else:
            if number != spec.upcoming_season - 1:
                raise PipelineError("The ending must describe the latest prior season.")
            ending_ids.update(episode_ids)
    if story_seasons != set(expected):
        raise PipelineError("A recap must include sourced story narration for every covered season.")
    latest = max((episode for episode in episodes.values() if episode["season"] in expected),
                 key=lambda episode: (episode["season"], episode["number"]))
    if str(latest["id"]) not in ending_ids:
        raise PipelineError("The ending must cite the latest prior-season episode.")
    return set(expected)
