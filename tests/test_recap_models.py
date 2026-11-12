"""Input and final scope checks do not require live research or an LLM."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from movie_pipeline.core import PipelineError
from recap_pipeline.models import RecapSpec, recap_seasons, validate_script_scope


def request(**changes):
    return RecapSpec(**{"show_title": "Example", "premiere_year": 2020, "upcoming_season": 3, **changes})


def research():
    return {
        "show": {"id": 1, "title": "Example", "year": 2020},
        "seasons": [{"number": n, "episodes": [
            {"id": f"e{n}", "season": n, "number": 1, "airdate": "2020-01-01",
             "summary": f"A supported and consequential event in season {n}.", "sources": [f"s{n}"]},
        ]} for n in (1, 2)],
        "sources": [{"id": f"s{n}", "season": n} for n in (1, 2)], "warnings": [],
    }


def script(kind="series"):
    numbers = [2] if kind == "season" else [1, 2]
    sections = []
    for n in numbers:
        sections.append({"id": f"story-{n}", "kind": "story", "season": n,
                         "title": f"Season {n}", "paragraphs": ["A supported story event."],
                         "episode_ids": [f"e{n}"], "event_ids": [f"v{n}"],
                         "paragraph_episode_ids": [[f"e{n}"]], "paragraph_event_ids": [[f"v{n}"]],
                         "delivery": ["normal"]})
    sections.append({**deepcopy(sections[-1]), "id": "ending", "kind": "ending"})
    return {
        "kind": kind, "show_title": "Example", "upcoming_season": 3, "seasons": numbers,
        "title": "Example recap", "description": "Sourced prior-season story.",
        "sections": sections, "source_ids": [f"s{n}" for n in numbers], "warnings": [],
    }


def test_defaults_are_immutable_and_serialization_contains_only_request():
    spec = request()
    assert spec.narrator == "Warguy" and spec.tts_model == "original" and spec.try_4k
    assert spec.season_minutes == spec.series_minutes == "Auto"
    with pytest.raises(FrozenInstanceError):
        spec.tts_model = "turbo"
    assert RecapSpec.from_dict(spec.to_dict()) == spec
    assert set(spec.to_dict()) == {"show_title", "premiere_year", "upcoming_season", "brief",
                                  "narrator", "tts_model", "season_minutes", "series_minutes", "try_4k",
                                      "voice_speed"}
    assert recap_seasons("season", spec) == [2]
    assert recap_seasons("series", spec) == [1, 2]


def test_environment_parses_only_known_inputs_and_keeps_brief_as_data():
    spec = RecapSpec.from_env({"SHOW_TITLE": " Example ", "SHOW_YEAR": "2020", "UPCOMING_SEASON": "3",
                              "RECAP_BRIEF": "Use a calm delivery. $() ; \"quotes\"",
                              "NARRATOR": "Kit Laser", "TTS_MODEL": "nano", "SEASON_MINUTES": "8",
                              "SERIES_MINUTES": "12", "TRY_4K": "false",
                              "TELEGRAM_BOT_TOKEN": "private", "UNRELATED": "ignored"})
    assert spec.show_title == "Example" and spec.narrator == "Kit Laser"
    assert spec.tts_model == "nano" and not spec.try_4k
    assert "private" not in repr(spec.to_dict())
    assert spec.brief.endswith('\"quotes\"')


@pytest.mark.parametrize("field,value", [
    ("show_title", ""), ("show_title", "A\nB"), ("show_title", "x" * 301),
    ("premiere_year", 1899), ("premiere_year", 2101), ("premiere_year", True),
    ("upcoming_season", 1), ("upcoming_season", 101), ("upcoming_season", 3.0),
    ("brief", "x" * 4001), ("brief", None), ("narrator", "warguy"),
    ("tts_model", "other"), ("try_4k", "true"),
    ("season_minutes", "0"), ("season_minutes", "31"), ("season_minutes", "1.5"),
    ("series_minutes", True), ("series_minutes", "auto"),
])
def test_invalid_requests_fail(field, value):
    with pytest.raises(PipelineError):
        request(**{field: value})


@pytest.mark.parametrize("value", ["Auto", *map(str, range(1, 21)), *range(1, 21)])
def test_whole_runtime_preferences_are_soft_and_normalized(value):
    assert request(season_minutes=value).season_minutes == str(value)


@pytest.mark.parametrize("value", ["21", 25, "30", "0"])
def test_no_video_may_be_requested_longer_than_twenty_minutes(value):
    with pytest.raises(PipelineError, match="1 to 20"):
        request(series_minutes=value)


@pytest.mark.parametrize("value", ["0.9", "1.0", "1", "1.15", "1.3"])
def test_voice_speed_choices_are_accepted(value):
    assert request(voice_speed=value).voice_speed == ("1.0" if value == "1" else value)


@pytest.mark.parametrize("value", ["2", "0.5", "fast", ""])
def test_unknown_voice_speed_is_rejected(value):
    with pytest.raises(PipelineError, match="Voice speed"):
        request(voice_speed=value)


def test_voice_speed_is_read_from_the_environment():
    env = {"SHOW_TITLE": "Example", "SHOW_YEAR": "2020", "UPCOMING_SEASON": "3", "VOICE_SPEED": "1.2"}
    assert RecapSpec.from_env(env).voice_speed == "1.2"
    assert RecapSpec.from_env({**env, "VOICE_SPEED": ""}).voice_speed == "1.0"


@pytest.mark.parametrize("env", [
    {}, {"SHOW_TITLE": "Example", "SHOW_YEAR": "2020; echo secret", "UPCOMING_SEASON": "3"},
    {"SHOW_TITLE": "Example", "SHOW_YEAR": "2020", "UPCOMING_SEASON": "3", "TRY_4K": "yes"},
])
def test_malformed_environment_is_rejected(env):
    with pytest.raises(PipelineError):
        RecapSpec.from_env(env)


@pytest.mark.parametrize("value", [None, [], {"show_title": "Example"}, {**request().to_dict(), "token": "secret"}])
def test_saved_request_rejects_missing_unknown_and_malformed_fields(value):
    with pytest.raises(PipelineError):
        RecapSpec.from_dict(value)


def test_both_scripts_validate_against_shared_prior_seasons_without_word_quota():
    assert validate_script_scope(script(), request(), research()) == {1, 2}
    assert validate_script_scope(script("season"), request(), research()) == {2}


@pytest.mark.parametrize("mutation", [
    lambda s, b: s.update(show_title="Wrong production"),
    lambda s, b: b["show"].update(year=2021),
    lambda s, b: s.update(upcoming_season=4),
    lambda s, b: s.update(seasons=[1, 2, 3]),
    lambda s, b: b["seasons"].append({"number": 3, "episodes": []}),
    lambda s, b: b["sources"].append({"id": "future", "season": 3}),
    lambda s, b: s["sections"][0].update(season=3),
    lambda s, b: s["sections"][0].update(episode_ids=["unknown"]),
    lambda s, b: s["sections"][0].update(episode_ids=["e2"]),
    lambda s, b: s["sections"][0].update(event_ids=[]),
    lambda s, b: s["sections"][0].update(paragraphs=[""]),
    lambda s, b: s["sections"][0].update(paragraphs=["Season 3 changes the ending."]),
    lambda s, b: s["sections"][0].update(paragraphs=["S03E01 resolves it."]),
    lambda s, b: b["seasons"][0]["episodes"][0].update(summary=""),
    lambda s, b: b["seasons"][0]["episodes"][0].update(sources=[]),
    lambda s, b: b["seasons"][0]["episodes"][0].update(sources=["s2"]),
    lambda s, b: s.update(source_ids=["unknown"]),
    lambda s, b: s["sections"][0].update(paragraph_episode_ids=[["e2"]]),
    lambda s, b: s["sections"][0].update(paragraph_event_ids=[]),
    lambda s, b: s["sections"].pop(),
    lambda s, b: s["sections"].pop(0),
    lambda s, b: s.update(sections=[]),
    lambda s, b: s["sections"].append(deepcopy(s["sections"][0])),
])
def test_wrong_identity_future_cross_season_and_missing_evidence_are_rejected(mutation):
    data, bundle = script(), research()
    mutation(data, bundle)
    with pytest.raises(PipelineError):
        validate_script_scope(data, request(), bundle)


@pytest.mark.parametrize("value", [None, [], {"kind": "season"}])
def test_malformed_scripts_fail_with_pipeline_error(value):
    with pytest.raises(PipelineError):
        validate_script_scope(value, request(), research())


def test_previous_season_recap_cannot_cite_an_earlier_season():
    data = script("season")
    data["sections"][0]["episode_ids"] = ["e1"]
    with pytest.raises(PipelineError):
        validate_script_scope(data, request(), research())


def test_uncited_intro_and_outro_are_allowed_but_cannot_carry_story_evidence():
    data = script()
    intro = {"id": "intro", "kind": "intro", "season": None, "paragraphs": ["The story so far."],
             "episode_ids": [], "event_ids": []}
    data["sections"].insert(0, intro)
    validate_script_scope(data, request(), research())
    intro["episode_ids"] = ["e1"]
    with pytest.raises(PipelineError):
        validate_script_scope(data, request(), research())



@pytest.mark.parametrize("unused_current_metadata", [False, True])
def test_relabelled_earlier_story_does_not_cover_a_later_season(unused_current_metadata):
    data = script()
    earlier = deepcopy(data["sections"][0])
    earlier.update(id="story-2", season=2, title="Season 2")
    if unused_current_metadata:
        earlier["episode_ids"].append("e2")
        earlier["event_ids"].append("v2")
    data["sections"][1] = earlier
    # Keep the genuine latest-season ending; it cannot replace missing body story.
    with pytest.raises(PipelineError, match="every covered season"):
        validate_script_scope(data, request(), research())


def test_prior_context_and_current_story_together_cover_the_later_season():
    data = script()
    data["sections"][1].update(
        paragraphs=["An earlier choice still matters.", "A current-season consequence follows."],
        episode_ids=["e1", "e2"], event_ids=["v1", "v2"],
        paragraph_episode_ids=[["e1"], ["e2"]], paragraph_event_ids=[["v1"], ["v2"]],
        delivery=["normal", "normal"],
    )
    assert validate_script_scope(data, request(), research()) == {1, 2}


@pytest.mark.parametrize("kind", ["intro", "outro", "story", "ending"])
@pytest.mark.parametrize("text", [
    "Here is where the story stands before season 3.",
    "Ahead of season three, the story so far.",
    "The recap stops before the incoming season 3.",
    "Before the upcoming season three, these are the completed events.",
])
def test_incoming_season_cutoff_is_orientation_in_every_section(kind, text):
    data = script()
    if kind in ("intro", "outro"):
        section = {"id": kind, "kind": kind, "season": None,
                   "paragraphs": [text], "episode_ids": [], "event_ids": []}
        data["sections"].append(section)
    else:
        section = next(section for section in data["sections"] if section["kind"] == kind)
        section["paragraphs"] = [text]
    assert validate_script_scope(data, request(), research()) == {1, 2}


@pytest.mark.parametrize("text", [
    "In season 3, Mara escapes.",
    "Season three reveals who betrayed Mara.",
    "Season four changes the ending.",
    "Here is the story before season 4.",
    "Before season 3, remember that in season 3 Mara escapes.",
    "Ahead of season three, S03E01 resolves the conflict.",
    "Before S03E01, Mara escapes.",
    "Before season three, S04E01 changes everything.",
])
def test_cutoff_wording_cannot_hide_future_plot_or_episode_references(text):
    data = script()
    data["sections"].append({"id": "outro", "kind": "outro", "season": None,
                             "paragraphs": [text], "episode_ids": [], "event_ids": []})
    with pytest.raises(PipelineError, match="incoming or later-season plot"):
        validate_script_scope(data, request(), research())


def test_valid_cutoff_does_not_authorize_incoming_season_sources():
    data, bundle = script(), research()
    data["sections"].append({"id": "outro", "kind": "outro", "season": None,
                             "paragraphs": ["The story so far before season 3."],
                             "episode_ids": [], "event_ids": []})
    bundle["sources"].append({"id": "future", "season": 3})
    data["source_ids"].append("future")
    with pytest.raises(PipelineError, match="incoming or later-season source"):
        validate_script_scope(data, request(), bundle)
