import numpy as np
import pytest
from PIL import Image, ImageDraw

from recap_pipeline import footage
from recap_pipeline.editing import UsageTracker, _window, plan_visuals


def test_shots_follow_real_cuts_and_skip_fades():
    shots = footage._intervals(60, [3.0, 5.2, 9.0], [(5.2, 5.6)], 1.0, 12.0)
    assert [(s["start"], s["end"]) for s in shots] == [(1.0, 3.0), (3.0, 5.2), (5.6, 9.0), (9.0, 12.0)]
    # The shot after a fade does not start on a cut, so it cannot continue the earlier run.
    assert [s["cut_in"] for s in shots] == [True, True, False, True]


def test_window_starts_on_a_cut_and_never_ends_just_after_one():
    raw = [{"start": 1.0, "end": 3.0, "run": 1}, {"start": 3.0, "end": 4.1, "run": 1},
           {"start": 4.1, "end": 8.0, "run": 1}, {"start": 8.2, "end": 12.0, "run": 2}]
    start, covered = _window(raw, 0, 3.2)
    end = start + 3.2
    assert start >= 1.0 and covered[0] is raw[0]
    assert all(end - shot["start"] >= footage.FLASH or shot["start"] <= start for shot in covered)
    # Runs never join across a removed shot or fade.
    assert _window(raw, 2, 5.0) is None
    assert _window(raw, 3, 3.5) is not None


def test_gemini_notes_attach_to_precise_shots_and_unlogged_parts_are_dropped():
    shots = [{"start": 1.0, "end": 2.0, "run": 1}, {"start": 2.0, "end": 3.5, "run": 1},
             {"start": 3.5, "end": 5.0, "run": 1}, {"start": 5.0, "end": 7.0, "run": 1}]
    scenes = [{"start": 1.0, "end": 3.0, "description": "Reacher at the door", "event_id": "e1"},
              {"start": 5.0, "end": 7.0, "description": "A car chase"}]
    merged = footage.merge_notes(shots, scenes, "clip.mp4", logged=True)
    assert [s["start"] for s in merged] == [1.0, 2.0, 5.0]  # 3.5-5.0 was left out by Gemini
    assert merged[0]["event_id"] == "e1" and merged[0]["gemini_key"] == merged[1]["gemini_key"]
    assert merged[2]["run"] != merged[1]["run"]  # the dropped shot splits the run
    assert footage.merge_notes(shots, [], "clip.mp4", logged=False) == shots


@pytest.mark.parametrize("words,flat,junk", [
    (["THIS", "FEBRUARY"], False, True),
    (["SUBSCRIBE", "NOW"], False, True),
    (["Credit-", "Prime", "Video"], False, True),
    (["ACTION", "SOCIETY"], False, True),
    (["prime"], False, False),          # the broadcaster's corner logo on official footage
    (["rime"], False, False),
    ([], False, False),
    (["THIS"], True, True),             # one big word on a plain frame is a title card
])
def test_on_screen_text_rules(words, flat, junk):
    assert footage.junk_text(words, flat=flat) is junk


def _frames(tmp_path, overlay=None):
    rng = np.random.default_rng(3)
    paths = []
    for index in range(10):
        picture = (rng.random((270, 480)) * 160 + 40).astype(np.uint8)
        image = Image.fromarray(picture).convert("RGB")
        if overlay == "top":
            ImageDraw.Draw(image).text((60, 6), "REACHER DOES RUSSIAN ROULETTE " * 3, fill="white")
        if overlay == "bottom-left":
            ImageDraw.Draw(image).text((6, 255), "Credit: Prime Video", fill="white")
        path = tmp_path / f"{overlay}-{index}.jpg"
        image.save(path, quality=95)
        paths.append(path)
    return paths


def test_fixed_overlays_are_found_and_cropped_away(tmp_path):
    assert footage.static_overlay(_frames(tmp_path)) is None
    assert footage.static_overlay(_frames(tmp_path, "top")) == (0.5, 1.0)       # keep the lower part
    assert footage.static_overlay(_frames(tmp_path, "bottom-left")) == (1.0, 0.0)  # keep upper right


def test_reused_footage_is_reframed_and_overlay_clips_cropped():
    usage = UsageTracker(4, allowed_seasons={3})
    row = {"path": "/clip.mp4", "kind": "video", "season": 3, "duration": 40, "scope": "season", "episode_ids": []}
    usage.analysed_sources.add("/clip.mp4")
    usage.shots["/clip.mp4"] = [{"start": 1.0 + 4 * i, "end": 5.0 + 4 * i, "run": 1, "precise": True,
                                 "motion": 5.0, "crop_anchor": [0.5, 1.0]} for i in range(3)]
    assets = {"seasons": {"3": [row]}}
    section = {"id": "s", "kind": "story", "season": 3, "episode_ids": []}
    first = plan_visuals(section, "Reacher arrives.", 8.0, assets, usage)
    assert all(not shot["still"] and shot["crop_anchor"] == [0.5, 1.0] for shot in first)
    assert not any(shot["reframe"] for shot in first)
    again = plan_visuals(section, "Reacher arrives again.", 4.0, assets, usage)
    assert again[0]["reframe"] and not again[0]["still"]


def test_title_card_dissolved_into_a_shot_is_cut_out(tmp_path):
    rng = np.random.default_rng(5)
    frames = {}
    for index in range(4 * 8):   # 8 seconds at 4 frames per second, no cut detected anywhere
        t = round(index / footage.FRAME_RATE, 2)
        coarse = (rng.random((9, 16)) * 160 + 40).astype(np.uint8)   # broad shapes, like real footage
        path = tmp_path / f"{index:03}.jpg"
        Image.fromarray(coarse).resize((480, 270)).convert("RGB").save(path)
        frames[t] = path
    card = {path for t, path in frames.items() if 3.0 <= t <= 4.25}

    def reader(image):
        return ["NEW", "SEASON", "FEBRUARY"] if image.filename and __import__("pathlib").Path(image.filename) in card else []

    shots, dropped = footage.analyse(tmp_path / "clip.mp4", 9.0, tmp_path, reader=reader,
                                     scanner=lambda path, folder: ([], [], frames), promo_cutoff=9.0)
    assert shots, dropped
    assert all(s["end"] <= 3.0 or s["start"] >= 4.25 for s in shots)
    assert any(s["start"] < 3.0 for s in shots) and any(s["start"] >= 4.25 for s in shots)
    assert len({s["run"] for s in shots}) == 2  # the card splits the source's continuous run
