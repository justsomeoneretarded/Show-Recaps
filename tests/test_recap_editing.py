"""No model downloads: edit policy is exercised with bounded injected analyses."""

import pytest

from movie_pipeline.core import PipelineError
from recap_pipeline.editing import (
    MAX_ANALYSED_SOURCES, MAX_NATIVE_SOURCES, UsageTracker, analyse_assets,
    choose_resolution, plan_source_audio, plan_visuals, scene_catalogue,
)


def video(name="clip", *, season=2, ids=None, start=1.0, end=36.0, **extra):
    return {"path": f"/{name}.mp4", "kind": "video", "season": season,
            "scope": "episode" if ids else "season", "episode_ids": ids or [],
            "duration": 40.0, "width": 1920, "height": 1080,
            "shots": [{"start": start, "end": end, "motion": 2.0}], **extra}


def image(name="still", *, season=2, ids=None):
    return {"path": f"/{name}.jpg", "kind": "image", "season": season,
            "scope": "episode" if ids else "season-poster", "episode_ids": ids or [],
            "width": 1280, "height": 720, "illustrative": not bool(ids)}


def assets(*rows, season=2):
    return {"seasons": {str(season): list(rows)}}


def section(*, season=2, ids=None, name="story", kind="story"):
    return {"id": name, "kind": kind, "season": season, "episode_ids": ids or []}


def tracker():
    return UsageTracker(3, episode_seasons={"s2e1": 2, "s2e2": 2, "s1e1": 1, "s3e1": 3})


def test_rejects_incoming_section_before_selecting_any_visual():
    with pytest.raises(PipelineError, match="incoming|out-of-scope"):
        plan_visuals(section(season=3), "Incoming plot", 8, assets(video(season=3), season=3), tracker())


def test_manifest_key_cannot_relabel_actual_future_source():
    with pytest.raises(PipelineError, match="no blank plate"):
        plan_visuals(section(), "A choice changes things.", 8, assets(video(season=3)), tracker())


def test_episode_map_rejects_wrong_season_paragraph():
    with pytest.raises(PipelineError, match="episode"):
        plan_visuals(section(ids=["s1e1"]), "A choice.", 8, assets(video()), tracker())


def test_episode_map_rejects_spoofed_source_episode_provenance():
    row = image(ids=["s3e1"])
    with pytest.raises(PipelineError, match="no blank plate"):
        plan_visuals(section(ids=["s2e1"]), "A choice.", 8, assets(row), tracker())


def test_exact_known_episode_preferred_to_season_generic():
    row = video("exact", ids=["s2e1"])
    result = plan_visuals(section(ids=["s2e1"]), "Marin finds the signal.", 8,
                          assets(video("generic"), row), tracker())
    assert result[0]["source"] == "/exact.mp4"
    assert result[0]["episode_id"] == "s2e1"


def test_wrong_episode_never_becomes_fallback():
    usage = tracker()
    result = plan_visuals(section(ids=["s2e2"]), "Marin finds the signal.", 8,
                          assets(image(ids=["s2e1"]), video("generic")), usage)
    assert result[0]["source"] == "/generic.mp4"
    assert result[0]["episode_ids"] == []
    assert any("exact current-episode" in message for message in usage.warnings)


def test_unknown_episode_never_borrows_another_episodes_still():
    with pytest.raises(PipelineError, match="no blank plate"):
        plan_visuals(section(ids=["s2e1", "s2e2"]), "An uncertain attribution.", 8,
                     assets(image(ids=["s2e1"])), tracker(), episode_ids=[])


def test_intro_shows_real_covered_season_footage():
    result = plan_visuals(section(season=None, kind="intro"), "The story so far.", 8,
                          assets(video()), tracker())
    assert result[0]["source"] == "/clip.mp4" and result[0]["season"] == 2
    assert result[0]["selection"] == "orientation-footage"


def test_season_without_footage_borrows_another_covered_season():
    usage = tracker()
    data = {"seasons": {"1": [video("season-one", season=1)], "2": []}}
    result = plan_visuals(section(), "A choice.", 8, data, usage)
    assert result[0]["source"] == "/season-one.mp4" and result[0]["season"] == 1
    assert result[0]["selection"] == "other-season-fallback"
    assert any("another covered season" in message for message in usage.warnings)


def test_long_paragraph_uses_even_six_to_ten_second_beats():
    result = plan_visuals(section(), "A long narration.", 32, assets(video()), tracker())
    assert sum(shot["duration"] for shot in result) == pytest.approx(32)
    assert all(6 <= shot["duration"] <= 10 for shot in result)
    assert all(not shot["freeze"] for shot in result)


def test_usage_survives_paragraph_boundary_instead_of_restarting_clip():
    usage, data = tracker(), assets(video())
    first = plan_visuals(section(), "First paragraph.", 8, data, usage)
    second = plan_visuals(section(), "Second paragraph.", 8, data, usage)
    assert first[0]["source"] == second[0]["source"]
    assert second[0]["start"] >= first[0]["start"] + first[0]["duration"]
    assert usage.fresh(first[0]["source"], first[0]["start"], first[0]["duration"]) == 0


def test_exhausted_source_reuses_real_footage_instead_of_a_blank_plate():
    usage, data = tracker(), assets(video(start=1, end=9))
    first = plan_visuals(section(), "First.", 8, data, usage)
    frame = plan_visuals(section(), "A limited moving frame.", 8, data, usage)
    final = plan_visuals(section(), "No more footage.", 8, data, usage)
    assert not first[0]["still"]
    # Moving footage, reused if necessary, always beats a frozen frame.
    assert all(not shot["still"] for shot in frame + final)
    assert final[0]["source"] == "/clip.mp4"
    assert final[0]["selection"] == "same-season-generic"
    assert any("reused" in message for message in usage.warnings)


def test_short_scenes_become_a_quick_moving_cut_not_a_frozen_frame():
    result = plan_visuals(section(), "A full paragraph.", 8, assets(video(start=3, end=5)), tracker())
    assert all(not shot["still"] for shot in result)
    assert sum(shot["duration"] for shot in result) == 8
    assert all(shot["duration"] >= 1.5 for shot in result)


def test_only_a_clip_of_stills_falls_back_to_a_moving_still():
    result = plan_visuals(section(), "A full paragraph.", 8, assets(image("one")), tracker())
    assert result[0]["still"] and result[0]["image"]


def test_available_stills_rotate_across_paragraphs():
    usage, data = tracker(), assets(image("one"), image("two"))
    first = plan_visuals(section(), "First.", 8, data, usage)[0]
    second = plan_visuals(section(), "Second.", 8, data, usage)[0]
    assert first["source"] != second["source"]
    assert first["still_motion"] != second["still_motion"]


def test_weak_semantic_match_is_nonfatal():
    result = plan_visuals(section(), "Unrelated novel vocabulary.", 8,
                          assets(video(title="No lexical overlap")), tracker())
    assert result[0]["source"] == "/clip.mp4"


def test_analysis_has_shared_source_ceiling(tmp_path):
    calls = []

    def analyse(path, cache):
        calls.append((path, cache))
        return [{"start": 1, "end": 30, "motion": 1}]

    rows = [video(f"source-{i}") for i in range(MAX_ANALYSED_SOURCES + 3)]
    for row in rows:
        row.pop("shots")
    usage = tracker()
    result = analyse_assets(assets(*rows), tmp_path, usage, analyser=analyse)
    analyse_assets(assets(*rows), tmp_path, usage, analyser=analyse)
    assert len(calls) == MAX_ANALYSED_SOURCES
    assert len(result) == MAX_ANALYSED_SOURCES
    assert all(path.name.startswith("source-") for path, _ in calls)
    assert any(f"{MAX_ANALYSED_SOURCES} sources" in message for message in usage.warnings)


def test_analysis_does_not_process_wrong_season_or_invalid_episode(tmp_path):
    calls = []
    row = video("future", season=3)
    row.pop("shots")
    spoof = video("spoof", ids=["s3e1"])
    spoof.pop("shots")
    analyse_assets(assets(row, spoof), tmp_path, tracker(), analyser=lambda *args: calls.append(args))
    assert calls == []


def test_failed_optional_analysis_still_uses_the_clip(tmp_path):
    def unavailable(*args):
        raise RuntimeError("Optional analysis is offline")

    usage, row = tracker(), video()
    row.pop("shots")
    analyse_assets(assets(row), tmp_path, usage, analyser=unavailable)
    result = plan_visuals(section(), "Complete narration.", 8, assets(row), usage)
    assert result[0]["source"] == "/clip.mp4" and not result[0]["still"]
    assert any("unavailable" in message for message in usage.warnings)


def test_title_cards_are_never_selected():
    row = video()
    row["shots"] = [{"start": 1, "end": 30, "text_card": "promo"}]
    with pytest.raises(PipelineError, match="no blank plate"):
        plan_visuals(section(), "A choice.", 8, assets(row), tracker())


def test_gemini_ranking_picks_the_matching_scene():
    row = video(shots=[{"start": 1, "end": 12, "description": "Two people talk in a kitchen", "quality": 7},
                       {"start": 20, "end": 32, "description": "A boat sails toward the horizon", "quality": 7}])
    usage, data = tracker(), assets(row)
    catalogue = scene_catalogue(section(), "He sails away.", data, usage)
    assert len(catalogue) == 2 and {"key", "description", "seconds"} <= set(catalogue[0])
    boat = next(c["key"] for c in catalogue if "boat" in c["description"])
    result = plan_visuals(section(), "He leaves.", 8, data, usage, preferred=[boat])
    assert 20 <= result[0]["start"] < 32


def test_watermarked_or_dark_scenes_are_avoided_when_clean_ones_exist():
    row = video(shots=[{"start": 1, "end": 12, "watermark": True, "quality": 6},
                       {"start": 20, "end": 32, "quality": 6, "dark": False}])
    result = plan_visuals(section(), "A scene.", 8, assets(row), tracker())
    assert result[0]["start"] >= 20


def test_resolution_uses_source_detail_and_respects_4k_option():
    data = assets(video(width=3840, height=2160))
    assert choose_resolution(data, True)[:2] == (3840, 2160)
    assert choose_resolution(data, False)[:2] == (1920, 1080)
    low = assets(video(width=640, height=360))
    width, height, warnings = choose_resolution(low, True)
    assert (width, height) == (640, 360)
    assert any("Source quality" in message for message in warnings)





def native_cues():
    return [{"start": 3.0, "end": 6.2, "text": "Marin found the signal."}]


def test_native_audio_uses_whole_relevant_phrase_and_reserves_freshness():
    usage = tracker()
    result = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal in the lab.",
                               assets(video(ids=["s2e1"])), usage,
                               analyser=lambda path: native_cues())
    assert result["start"] < 3
    assert result["start"] + result["duration"] > 6.2
    assert result["text"] == "Marin found the signal."
    assert result["episode_id"] == "s2e1" and result["native_verified"]
    assert result["duration"] <= 12
    assert usage.fresh(result["source"], result["start"], result["duration"]) == 0


def test_native_audio_has_no_beat_quota_and_skips_unrelated_complete_dialogue():
    result = plan_source_audio(section(ids=["s2e1"]), "The council closes the bridge.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: native_cues())
    assert result is None


@pytest.mark.parametrize("ids", [[], ["s2e1", "s2e2"]])
def test_native_audio_requires_one_known_current_episode_before_asr(ids):
    calls = []
    result = plan_source_audio(section(ids=ids), "Marin found the signal.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: calls.append(path))
    assert result is None and calls == []


def test_native_audio_wrong_episode_is_never_analyzed():
    calls = []
    result = plan_source_audio(section(ids=["s2e2"]), "Marin found the signal.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: calls.append(path))
    assert result is None and calls == []


def test_native_audio_skips_incomplete_phrase_instead_of_chopping_words():
    result = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: [{"start": 3, "end": 6.2, "text": "Marin found the signal"}])
    assert result is None


def test_native_audio_stays_within_one_coherent_scene():
    result = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.",
                               assets(video(ids=["s2e1"], start=4, end=10)), tracker(),
                               analyser=lambda path: native_cues())
    assert result is None


def test_native_audio_has_four_source_analysis_ceiling():
    usage, calls = tracker(), []
    rows = [video(f"native-{i}", ids=["s2e1"]) for i in range(8)]

    def no_dialogue(path):
        calls.append(path)
        return []

    assert plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.",
                              assets(*rows), usage, analyser=no_dialogue) is None
    assert len(calls) == MAX_NATIVE_SOURCES
    assert len(usage.native_sources) == MAX_NATIVE_SOURCES


def test_native_asr_failure_is_optional():
    def unavailable(path):
        raise RuntimeError("No ASR")

    usage = tracker()
    assert plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.",
                              assets(video(ids=["s2e1"])), usage, analyser=unavailable) is None
    assert any("source-audio" in message for message in usage.warnings)


def test_only_one_native_beat_per_story_section():
    usage, data = tracker(), assets(video(ids=["s2e1"]))
    first = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.", data, usage,
                              analyser=lambda path: native_cues())
    second = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.", data, usage,
                               analyser=lambda path: native_cues())
    assert first is not None
    assert second is None


def test_fast_unstable_source_becomes_gently_moving_still():
    row = video()
    row["shots"][0]["motion"] = 70
    result = plan_visuals(section(), "A measured account.", 8, assets(row), tracker())
    assert result[0]["still"]
    assert result[0]["still_motion"] in {"zoom_in", "zoom_out"}


def test_native_phrase_uses_actual_asr_words_and_safe_whole_word_padding():
    cues = [{"start": 1, "end": 9, "text": "Marin found the signal.", "words": [
        {"start": 3, "end": 3.5, "word": "Marin"},
        {"start": 3.5, "end": 4.3, "word": "found"},
        {"start": 4.3, "end": 4.5, "word": "the"},
        {"start": 4.5, "end": 6.2, "word": "signal."},
    ]}]
    result = plan_source_audio(section(ids=["s2e1"]), "Marin found the signal.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: cues)
    assert result["start"] == pytest.approx(2.86)
    assert result["start"] + result["duration"] == pytest.approx(6.4)
    assert result["text"] == "Marin found the signal."


def test_native_generic_function_words_do_not_establish_relevance():
    result = plan_source_audio(section(ids=["s2e1"]), "You must now leave the bridge.",
                               assets(video(ids=["s2e1"])), tracker(),
                               analyser=lambda path: [{"start": 3, "end": 6.2,
                                                        "text": "You must now accept your crown."}])
    assert result is None


def test_image_only_analysis_does_not_import_optional_opencv(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"cv2", "movie_pipeline.shots"}:
            raise AssertionError("Image-only recap attempted optional video analysis")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    usage = tracker()
    assert analyse_assets(assets(image()), tmp_path, usage) == {}
    assert usage.analysed_sources == set()


def test_missing_optional_video_dependency_still_uses_footage(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "movie_pipeline.shots":
            raise ImportError("Optional OpenCV is unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    usage, row = tracker(), video()
    row.pop("shots")
    analyse_assets(assets(row), tmp_path, usage)
    shot = plan_visuals(section(), "Complete narration.", 8, assets(row), usage)[0]
    assert shot["source"] == "/clip.mp4" and shot["selection"] == "same-season-generic"


@pytest.mark.parametrize("seconds", [6.0, 8.0, 9.1, 10.0])
def test_complete_six_to_ten_second_paragraph_stays_one_coherent_beat(seconds):
    result = plan_visuals(section(), "A complete short paragraph.", seconds, assets(video()), tracker())
    assert len(result) == 1
    assert result[0]["duration"] == seconds


def test_low_resolution_clips_are_dropped_when_sharp_ones_exist():
    from recap_pipeline.editing import prefer_sharp_sources
    rows = [video(f"hd{i}", height=1080) for i in range(3)] + [video("sd", height=480)]
    kept, warnings = prefer_sharp_sources(assets(*rows))
    assert [r["path"] for r in kept["seasons"]["2"]] == ["/hd0.mp4", "/hd1.mp4", "/hd2.mp4"]
    assert warnings and choose_resolution(kept, False)[:2] == (1920, 1080)
    # With too few sharp clips nothing is removed.
    few, none = prefer_sharp_sources(assets(video("hd", height=1080), video("sd", height=480)))
    assert len(few["seasons"]["2"]) == 2 and none == []


def test_edit_spreads_across_clips_instead_of_leaning_on_one():
    usage, data = tracker(), assets(video("one", start=1, end=39), video("two", start=1, end=39),
                                    video("three", start=1, end=39))
    used = {plan_visuals(section(), f"Paragraph {i}.", 8, data, usage)[0]["source"] for i in range(6)}
    assert used == {"/one.mp4", "/two.mp4", "/three.mp4"}
