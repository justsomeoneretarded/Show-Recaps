"""Gemini footage help, clip harvesting rules, letterbox removal and the 20-minute cap.

Gemini is always faked here: no test reaches the live API.
"""
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from movie_pipeline.core import PipelineError, ffmpeg
from recap_pipeline import assets, gemini_media
from recap_pipeline.gemini import GeminiUnavailable
from recap_pipeline.writing import _fit_budget, _section, apply_packaging

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
media = pytest.mark.skipif(not HAS_FFMPEG, reason="Actual FFmpeg is required.")
SHOW = {"id": 1, "title": "The Terminal List", "year": 2022}


class FakeGemini:
    model = "gemini-test"

    def __init__(self, *answers, error=None):
        self.answers, self.error, self.calls, self.uploads, self.deleted = list(answers), error, [], [], []

    def generate_json(self, system, prompt, schema, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error:
            raise self.error
        return self.answers.pop(0)

    def upload_file(self, path, mime, **kwargs):
        self.uploads.append(Path(path))
        return {"name": "files/abc", "uri": "https://files.example/abc", "mimeType": mime}

    def delete_file(self, resource):
        self.deleted.append(resource["name"])


def test_suggested_queries_keep_only_this_seasons_official_footage():
    client = FakeGemini({"queries": [
        "The Terminal List season 1 Reece boat scene", "The Terminal List season 2 trailer",
        "The Terminal List season 1 reaction", "Jack Reacher clip", "The Terminal List season 1 Reece boat scene",
        "The Terminal List season 1 explained ending"]})
    queries = gemini_media.suggest_queries(client, SHOW, 1, ["The Engram"])
    assert queries == ["The Terminal List season 1 Reece boat scene"]
    assert "The Engram" in client.calls[0][0]


def test_query_suggestions_are_optional():
    assert gemini_media.suggest_queries(None, SHOW, 1, []) == []
    assert gemini_media.suggest_queries(FakeGemini(error=GeminiUnavailable("quota")), SHOW, 1, []) == []


def test_scene_log_drops_unusable_rows_and_splits_long_scenes():
    raw = {"scenes": [
        {"start": 2, "end": 47, "description": "Reece walks along a dock", "characters": ["James Reece"],
         "quality": 8, "usable": True, "watermark": True, "letterbox": False, "on_screen_text": False,
         "interview": False, "dark": False},
        {"start": 50, "end": 55, "description": "Cast interview", "usable": True, "interview": True},
        {"start": 56, "end": 58, "description": "Release date card", "usable": True, "on_screen_text": True},
        {"start": 60, "end": 62, "description": "Black frame", "usable": False},
        {"start": 90, "end": 99, "description": "Beyond the clip"}],
        "thumbnail_times": [12.5, 300, -1]}
    scenes, times = gemini_media.clean_scenes(raw, 70.0)
    assert [(s["start"], s["end"]) for s in scenes] == [(2, 22), (22, 42), (42, 47)]
    assert "James Reece" in scenes[0]["description"] and scenes[0]["watermark"] and scenes[0]["quality"] == 8
    assert times == [12.5]


@media
def test_clip_is_watched_through_a_small_proxy_and_cached(tmp_path):
    clip = tmp_path / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=6",
           "-f", "lavfi", "-i", "sine=frequency=330:duration=6", "-shortest",
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", clip)
    client = FakeGemini({"scenes": [{"start": 0.5, "end": 5.5, "description": "Test pattern", "characters": [],
                                     "quality": 6, "usable": True, "watermark": False, "letterbox": False,
                                     "on_screen_text": False, "interview": False, "dark": False}],
                         "thumbnail_times": [3.0]})
    first = gemini_media.analyse_clip(client, clip, 6.0, SHOW, 1, [], tmp_path / "work")
    assert first["scenes"][0]["description"] == "Test pattern" and first["thumbnail_times"] == [3.0]
    assert client.uploads[0].name.endswith("-gemini-proxy.mp4") and client.deleted == ["files/abc"]
    assert client.calls[0][1]["media"][0]["fileData"]["fileUri"] == "https://files.example/abc"
    # The second request is served from the cache next to the clip, with no new call.
    assert gemini_media.analyse_clip(FakeGemini(), clip, 6.0, SHOW, 1, [], tmp_path / "work") == first


def test_ranking_only_returns_catalogue_keys():
    catalogue = [{"key": "aaa", "description": "dock", "seconds": 8},
                 {"key": "bbb", "description": "house", "seconds": 8}]
    client = FakeGemini({"ranked": ["bbb", "zzz", "aaa", "bbb"]})
    assert gemini_media.rank_scenes(client, "He returns home.", catalogue) == ["bbb", "aaa"]
    assert gemini_media.rank_scenes(None, "x", catalogue) == []


def test_upload_windows_end_well_before_the_next_season():
    seasons = [{"number": 1, "episodes": [{"airdate": "2022-07-01"}, {"airdate": "2022-07-29"}]},
               {"number": 2, "episodes": [{"airdate": "2023-01-01"}]}]
    windows = assets.season_windows(seasons)
    assert windows[1][0] == date(2022, 4, 2)
    assert windows[1][1] == date(2022, 6, 15)  # 200 days before season 2, not 150 after the finale
    assert windows[2] == (date(2022, 10, 3), date(2023, 5, 31))


def _clip(**extra):
    return {"title": "Reece Finds the Truth - The Terminal List Clip", "description": "Prime Video",
            "duration": 120, **extra}


def test_clip_naming_no_season_needs_an_upload_inside_the_window():
    window = (date(2022, 4, 1), date(2022, 12, 1))
    assert assets.candidate_ok(_clip(upload_date="20220705"), SHOW, 1, 2, window=window)
    assert not assets.candidate_ok(_clip(upload_date="20250705"), SHOW, 1, 2, window=window)
    assert not assets.candidate_ok(_clip(upload_date="20220705"), SHOW, 1, 2)
    assert not assets.candidate_ok(_clip(upload_date="20220705", title="The Terminal List Season 2 Clip"),
                                   SHOW, 1, 3, window=window)


def test_more_official_searches_including_episode_titles():
    season = {"number": 1, "episodes": [{"title": "The Engram"}, {"title": "Encoding"}]}
    queries = assets.extra_queries(SHOW, season)
    assert any("official clip" in q for q in queries) and any('"The Engram" clip' in q for q in queries)
    assert all('"season 1"' in q or "season 1" in q for q in queries)


@media
def test_letterboxed_clip_is_reframed_to_fill_the_picture(tmp_path):
    clip = tmp_path / "wide.mp4"
    # A 2.39:1 picture inside a 16:9 frame, with black bars top and bottom.
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x268:rate=24:duration=12",
           "-vf", "pad=640:360:0:46:black", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", clip)
    asset, warnings = {"path": str(clip), "kind": "video"}, []
    assert assets._letterbox_crop(clip)
    assets._remove_letterbox(asset, warnings)
    assert asset["letterbox_removed"] and (asset["width"], asset["height"]) == (640, 360)
    assert assets._letterbox_crop(clip) is None and not warnings


@media
def test_full_frame_clip_is_left_alone(tmp_path):
    clip = tmp_path / "full.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=12",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", clip)
    assert assets._letterbox_crop(clip) is None


def _story(texts):
    events = {f"e{i}": {"episode_id": "1"} for i in range(len(texts))}
    paragraphs = [{"text": t, "event_ids": [f"e{i}"], "delivery": "normal"} for i, t in enumerate(texts)]
    return _section("story-s1-1", "story", "The list", 1, paragraphs, events), events


def test_long_narration_is_trimmed_to_the_twenty_minute_budget():
    sentence = "Reece follows the trail to another name on the list. "
    story, events = _story([sentence * 6, sentence * 6, sentence * 6])
    ending = {**story, "id": "end", "kind": "ending", "paragraphs": ["The final name is crossed off."]}
    total = sum(len(p.split()) for p in story["paragraphs"])
    sections, trimmed = _fit_budget([story, ending], total // 2, events)
    words = sum(len(p.split()) for s in sections for p in s["paragraphs"])
    assert trimmed and words <= total // 2 + len(ending["paragraphs"][0].split())
    assert sections[1] is ending
    assert all(p.startswith("Reece follows") for p in sections[0]["paragraphs"])
    assert len(sections[0]["paragraph_event_ids"]) == len(sections[0]["paragraphs"])


def test_short_narration_is_untouched():
    story, events = _story(["One sentence."])
    sections, trimmed = _fit_budget([story], 1000, events)
    assert not trimmed and sections[0] is story


def _script():
    return {"show_title": "The Terminal List", "upcoming_season": 2, "seasons": [1],
            "title": "The Terminal List: Season 1 Recap — Before Season 2", "description": "Recap.",
            "thumbnail_concepts": [{"headline": "WHERE WE LEFT OFF", "concept": "x"}],
            "sections": [{"paragraphs": ["Reece crosses off the final name."]}]}


def test_packaging_applies_only_safe_suggestions():
    client = FakeGemini({"title": "The Terminal List Season 1 Recap: Every Name on Reece's List",
                         "headline": "the final name", "hook": "One SEAL, ten names, no mercy."})
    script = apply_packaging(_script(), client)
    assert script["title"].endswith("Every Name on Reece's List")
    assert script["thumbnail_concepts"][0]["headline"] == "THE FINAL NAME"
    assert script["description"].startswith("One SEAL, ten names")


def test_packaging_rejects_later_seasons_and_missing_show_name():
    client = FakeGemini({"title": "What happens in season 3?", "headline": "SEASON 2 IS HERE",
                         "hook": "Watch it on Prime Video now."})
    script = apply_packaging(_script(), client)
    assert script == _script()
    assert apply_packaging(_script(), None) == _script()


def test_rejected_upload_is_reported_as_unavailable(tmp_path):
    from recap_pipeline.gemini import GeminiClient

    class Response:
        status_code, headers = 403, {}

        def json(self):
            return {}

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"x")
    with pytest.raises(GeminiUnavailable, match="could not start"):
        GeminiClient("key", session=Session(), sleep=lambda s: None).upload_file(proxy, "video/mp4")


def test_collecting_with_no_reachable_footage_fails_instead_of_a_blank_plate(monkeypatch, tmp_path):
    from recap_pipeline.models import RecapSpec
    monkeypatch.setattr(assets, "_discover_videos", lambda *args, **kwargs: [])
    monkeypatch.setattr(assets, "_fallback_images", lambda *args, **kwargs: [])
    bundle = {"show": {**SHOW}, "seasons": [{"number": 1, "episodes": []}]}
    spec = RecapSpec(show_title="The Terminal List", premiere_year=2022, upcoming_season=2)
    with pytest.raises(PipelineError, match="no blank-plate video"):
        assets.collect_assets(bundle, spec, tmp_path)


def test_logging_attaches_gemini_scenes_to_each_clip(monkeypatch, tmp_path):
    seen = []

    def analyse(client, path, length, show, season, titles, work):
        seen.append(path.name)
        return {"scenes": [{"start": 1, "end": 9, "description": "Dock", "motion": 0.0}], "thumbnail_times": [4]}
    monkeypatch.setattr(gemini_media, "analyse_clip", analyse)
    monkeypatch.setattr(assets, "_remove_letterbox", lambda asset, warnings: None)
    rows = [{"kind": "video", "path": str(tmp_path / "a.mp4"), "duration": 30},
            {"kind": "image", "path": str(tmp_path / "b.jpg")}]
    warnings = []
    assets._log_clips(rows, SHOW, {"number": 1, "episodes": []}, tmp_path, FakeGemini(), warnings)
    assert seen == ["a.mp4"] and rows[0]["shots"][0]["description"] == "Dock"
    assert rows[0]["thumbnail_times"] == [4] and "shots" not in rows[1]
    assert any("Gemini logged scenes in 1 of 1" in w for w in warnings)
    assert json.dumps(rows)  # still a serialisable manifest
