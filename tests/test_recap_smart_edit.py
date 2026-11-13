"""Scene-to-event links, sentence-level cuts on key words, footage-aware writing, real dialogue."""
import pytest

from recap_pipeline import assets, gemini_media, rendering
from recap_pipeline.editing import UsageTracker, plan_source_audio, plan_visuals
from recap_pipeline.gemini import GeminiUnavailable
from recap_pipeline.sync import segments, sentences, word_times


class FakeGemini:
    model = "gemini-test"

    def __init__(self, *answers, error=None):
        self.answers, self.error, self.calls = list(answers), error, []

    def generate_json(self, system, prompt, schema, **kwargs):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.answers.pop(0)


TEXT = "Reece leaves the hospital. The car explodes behind him. He runs."


def _aligned():
    # One recognised cue per sentence, as the recogniser returns them.
    return [{"text": "Reece leaves the hospital.", "start": 0.0, "end": 2.4},
            {"text": "The car explodes behind him.", "start": 2.6, "end": 5.4},
            {"text": "He runs.", "start": 5.6, "end": 8.0}]


def test_sentences_keep_every_word_once():
    assert sentences(TEXT) == ["Reece leaves the hospital.", "The car explodes behind him.", "He runs."]
    assert " ".join(sentences(TEXT)) == TEXT


def test_word_times_follow_the_recording():
    words = word_times(TEXT, _aligned(), 8.0)
    assert [w for w, _, _ in words] == TEXT.split()
    explodes = next(t for t in words if t[0] == "explodes")
    assert 3.0 < explodes[1] < 4.5
    # Without recognition the words are spread over the whole take.
    even = word_times(TEXT, [], 8.0)
    assert even[0][1] == 0 and even[-1][2] == pytest.approx(8.0)


def test_cut_lands_just_before_the_key_word():
    words = word_times(TEXT, _aligned(), 8.0)
    explodes = next(t for t in words if t[0] == "explodes")[1]
    plan = segments(TEXT, words, 8.0, {1: "explodes"})
    second = next(s for s in plan if s["sentence"] == 1)
    assert second["start"] == pytest.approx(explodes - 0.3, abs=0.01)
    assert plan[0]["start"] == 0 and plan[-1]["end"] == 8.0
    assert all(a["end"] == pytest.approx(b["start"]) for a, b in zip(plan, plan[1:]))


def test_short_sentences_merge_instead_of_flickering():
    plan = segments(TEXT, word_times(TEXT, _aligned(), 8.0), 8.0)
    assert all(s["end"] - s["start"] >= 2.0 for s in plan)
    assert " ".join(s["text"] for s in plan) == TEXT


def test_link_scenes_keeps_only_confident_known_events():
    events = [{"id": "e1", "episode": 2, "summary": "A bomb destroys Reece's car."}]
    scenes = [{"description": "A car explodes on a street"}, {"description": "Two people talk"}]
    client = FakeGemini({"links": [{"scene": 0, "event_id": "e1", "confidence": 0.9},
                                   {"scene": 1, "event_id": "e1", "confidence": 0.3},
                                   {"scene": 0, "event_id": "zzz", "confidence": 1.0}]})
    assert gemini_media.link_scenes(client, scenes, events) == {0: events[0]}
    assert gemini_media.link_scenes(FakeGemini(error=GeminiUnavailable("quota")), scenes, events) == {}


def test_sentence_picks_validate_keys_and_key_words():
    client = FakeGemini({"sentences": [{"index": 0, "keys": ["a", "nope", "a"], "key_word": "hospital"},
                                       {"index": 1, "keys": ["b"], "key_word": "invented"},
                                       {"index": 9, "keys": ["a"], "key_word": "x"}]})
    picks = gemini_media.pick_sentence_shots(client, sentences(TEXT)[:2],
                                             [{"key": "a", "description": "x", "seconds": 4},
                                              {"key": "b", "description": "y", "seconds": 4}])
    assert picks == {0: {"keys": ["a"], "key_word": "hospital"}, 1: {"keys": ["b"], "key_word": ""}}


def _clip(name, shots, season=1):
    return {"kind": "video", "path": f"/{name}.mp4", "season": season, "episode_ids": [], "scope": "season",
            "duration": 60.0, "width": 1920, "height": 1080, "shots": shots}


def test_link_footage_tags_scenes_and_counts_seconds_per_event():
    collected = {"seasons": {"1": [_clip("a", [{"start": 0, "end": 6, "description": "Car explodes"},
                                               {"start": 6, "end": 9, "description": "Sky"}])]}, "warnings": []}
    ledger = {"events": [{"id": "e1", "season": 1, "episode": 2, "episode_id": "ep2", "summary": "Car bomb"}]}
    client = FakeGemini({"links": [{"scene": 0, "event_id": "e1", "confidence": 0.8}]})
    footage = assets.link_footage(collected, ledger, client)
    shot = collected["seasons"]["1"][0]["shots"][0]
    assert footage == {"e1": 6.0} and shot["event_id"] == "e1" and shot["episode_id"] == "ep2"
    assert any("linked 1 of 2" in w for w in collected["warnings"])
    # Saved links are reused without asking again.
    assert assets.link_footage(collected, ledger, FakeGemini()) == {"e1": 6.0}


def test_editor_prefers_the_scene_linked_to_the_narrated_event():
    clip = _clip("a", [{"start": 1, "end": 12, "description": "People talk", "quality": 6},
                       {"start": 20, "end": 32, "description": "Street", "quality": 6, "event_id": "e1"}], season=2)
    usage = UsageTracker(3, episode_seasons={"ep2": 2})
    shot = plan_visuals({"id": "s", "kind": "story", "season": 2, "episode_ids": []}, "Something happens.", 8,
                        {"seasons": {"2": [clip]}}, usage, events=["e1"])[0]
    assert 20 <= shot["start"] < 32 and shot["event_id"] == "e1"


def test_real_dialogue_plays_from_a_scene_linked_to_the_paragraph_episode():
    clip = _clip("a", [{"start": 0, "end": 20, "description": "Reece confronts Ben", "episode_id": "ep8",
                        "event_id": "e8"}], season=2)
    usage = UsageTracker(3, episode_seasons={"ep8": 2})
    cues = [{"start": 4.0, "end": 6.2, "text": "You were my friend, and you let them"},
            {"start": 6.3, "end": 8.0, "text": "kill my family."}]
    section = {"id": "end", "kind": "ending", "season": 2, "episode_ids": ["ep8"]}
    beat = plan_source_audio(section, "Reece faces his closest friend.", {"seasons": {"2": [clip]}}, usage,
                             episode_ids=["ep8"], analyser=lambda path: cues, events=["e8"])
    assert beat and beat["episode_ids"] == ["ep8"] and 3 <= beat["duration"] <= 12
    # Unlinked or other-episode footage never speaks.
    usage = UsageTracker(3, episode_seasons={"ep8": 2, "ep1": 2})
    assert plan_source_audio({**section, "episode_ids": ["ep1"]}, "Reece faces his closest friend.",
                             {"seasons": {"2": [clip]}}, usage, episode_ids=["ep1"],
                             analyser=lambda path: cues, events=["e1"]) is None


def test_sentence_shots_cover_the_paragraph_with_one_cut_per_sentence():
    clip = _clip("a", [{"start": 1, "end": 15, "description": "Hospital corridor", "quality": 7},
                       {"start": 20, "end": 35, "description": "A car explodes", "quality": 7, "event_id": "e2"},
                       {"start": 40, "end": 55, "description": "A man runs", "quality": 7}], season=2)
    usage = UsageTracker(3, episode_seasons={})
    text = "Reece slowly leaves the busy hospital. The parked car explodes right behind him. He runs for cover."
    aligned = [{"text": s, "start": i * 3.0, "end": i * 3.0 + 2.8} for i, s in enumerate(sentences(text))]
    shots, cuts = rendering._sentence_shots(None, {"id": "s", "kind": "story", "season": 2, "episode_ids": []},
                                            text, aligned, 9.0, {"seasons": {"2": [clip]}}, usage, [], ["e2"], "ctx")
    assert len(cuts) == 3 and sum(s["duration"] for s in shots) == pytest.approx(9.0)
    assert len({s["start"] for s in shots}) == len(shots)  # a fresh shot for every sentence
    assert any(c["linked_event"] for c in cuts)


def test_writer_is_told_which_events_have_footage(tmp_path):
    from test_recap_writing import Model, bundle_with_seasons
    from recap_pipeline.models import RecapSpec
    from recap_pipeline.writing import build_ledger, write_recap
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    event = ledger["events"][0]["id"]
    write_recap("season", RecapSpec(show_title="The Gate", premiere_year=2021, upcoming_season=2),
                bundle, ledger, tmp_path / "script", writer=model, footage={event: 14.0})
    drafts = [data for task, data, _ in model.calls if "DRAFT_RECAP" in task]
    assert any(data["footage_seconds"] == {event: 14.0} for data in drafts)
    assert any("footage_seconds lists events the viewer will SEE" in task for task, _, _ in model.calls)


def test_a_few_4k_clips_never_push_out_the_1080p_ones():
    from recap_pipeline.editing import prefer_sharp_sources
    rows = [dict(_clip(f"uhd{i}", []), height=2160) for i in range(3)]
    rows += [dict(_clip(f"hd{i}", []), height=1080) for i in range(6)] + [dict(_clip("sd", []), height=360)]
    kept, _ = prefer_sharp_sources({"seasons": {"1": rows}})
    assert len(kept["seasons"]["1"]) == 9 and all(r["height"] >= 1080 for r in kept["seasons"]["1"])


def test_profane_dialogue_is_never_played():
    clip = _clip("a", [{"start": 0, "end": 20, "description": "Reece", "episode_id": "ep8", "event_id": "e8"}], season=2)
    usage = UsageTracker(3, episode_seasons={"ep8": 2})
    cues = [{"start": 4.0, "end": 8.0, "text": "Cleaning up all those motherfuckers that took them."}]
    section = {"id": "end", "kind": "ending", "season": 2, "episode_ids": ["ep8"]}
    assert plan_source_audio(section, "Reece cleans up.", {"seasons": {"2": [clip]}}, usage,
                             episode_ids=["ep8"], analyser=lambda path: cues, events=["e8"]) is None


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="FFmpeg required")
def test_dark_grey_letterbox_bars_are_detected(tmp_path):
    from movie_pipeline.core import ffmpeg
    clip = tmp_path / "grey.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x288:rate=24:duration=12",
           "-vf", "pad=640:360:0:36:0x1c1c14", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", clip)
    assert assets._letterbox_crop(clip)


def test_unlogged_scenes_with_captions_or_subscribe_screens_are_dropped(tmp_path):
    from recap_pipeline.editing import analyse_assets
    clip = {"kind": "video", "path": "/fan.mp4", "season": 2, "episode_ids": [], "scope": "season",
            "duration": 40.0, "width": 1920, "height": 1080,
            "shots": [{"start": 1, "end": 9}, {"start": 10, "end": 18}, {"start": 20, "end": 28},
                      {"start": 30, "end": 38, "source": "gemini", "description": "logged and clean"}]}
    texts = {5.0: "YOU MAY LIKE SUBSCRIBE NOW", 14.0: "Reacher Does Russian Roulette", 24.0: "IP"}
    usage = UsageTracker(3, episode_seasons={})
    shots = analyse_assets({"seasons": {"2": [clip]}}, tmp_path, usage,
                           text_reader=lambda path, second: texts.get(second, ""))["/fan.mp4"]
    assert [s["start"] for s in shots] == [20, 30]
    assert any("on-screen text" in w for w in usage.warnings)
