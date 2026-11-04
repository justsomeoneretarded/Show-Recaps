"""Synthetic media tests: actual FFmpeg, mocked speech/ASR, no model downloads.

These fixtures validate assembly, timing and guards. They are not evidence of
the quality of a real cloned narrator or an actual show recap.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from movie_pipeline import audio_review, render
from movie_pipeline.core import PipelineError, duration, ffmpeg, probe, run
from recap_pipeline import rendering
from recap_pipeline.models import RecapSpec

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
media = pytest.mark.skipif(not HAS_FFMPEG, reason="Actual FFmpeg and FFprobe are required.")


def _spec():
    return RecapSpec(show_title="Fixture Show", premiere_year=2020, upcoming_season=3,
                     brief="A restrained drama recap.", narrator="Warguy", tts_model="original",
                     season_minutes="Auto", series_minutes="Auto", try_4k=False)


def _bundle():
    return {
        "show": {"id": 12345, "title": "Fixture Show", "year": 2020, "genres": ["Drama"]},
        "seasons": [
            {"number": 1, "episodes": [{"id": "e1", "season": 1, "number": 1,
                "summary": "Mara finds the gate.", "sources": ["s1"]}]},
            {"number": 2, "episodes": [{"id": "e2", "season": 2, "number": 1,
                "summary": "Mara closes the gate and leaves.", "sources": ["s2"]}]},
        ],
        "sources": [
            {"id": "s1", "title": "Fixture season one", "season": 1, "episode": 1, "url": "https://example.test/season-1"},
            {"id": "s2", "title": "Fixture season two", "season": 2, "episode": 1, "url": "https://example.test/season-2"},
        ],
        "warnings": [],
    }


def _section(identifier, kind, season, text, episode=None):
    event = "event-" + str(episode) if episode else None
    return {"id": identifier, "kind": kind, "season": season, "title": identifier.replace("-", " ").title(),
            "paragraphs": [text], "delivery": ["normal"], "episode_ids": [episode] if episode else [],
            "event_ids": [event] if event else [],
            "paragraph_episode_ids": [[episode] if episode else []],
            "paragraph_event_ids": [[event] if event else []]}


def _script(kind):
    seasons = [2] if kind == "season" else [1, 2]
    sections = [_section("intro", "intro", None, "Let us follow the story.")]
    if kind == "series":
        sections.append(_section("story-s1", "story", 1, "Mara finds the hidden gate.", "e1"))
    sections.extend([
        _section("story-s2", "story", 2, "Mara closes the gate and leaves.", "e2"),
        _section("where-we-left-off", "ending", 2, "The gate remains closed behind Mara.", "e2"),
        _section("outro", "outro", None, "That brings this recap to a close."),
    ])
    return {"kind": kind, "title": "Fixture Show: The Story So Far",
            "show_title": "Fixture Show", "upcoming_season": 3, "seasons": seasons,
            "description": "Synthetic test fixture. Spoilers for the covered prior seasons.",
            "sections": sections, "source_ids": ["s2"] if kind == "season" else ["s1", "s2"],
            "coverage": {"complete": True, "summary_only": False, "missing_seasons": []}, "warnings": []}


def _audio(path, seconds=2.8, *, seed=1, gain=.16):
    """A varied speech-band envelope, deliberately not a synthetic spoken voice."""
    rate = 48000
    count = round(rate * seconds)
    time = np.arange(count) / rate
    generator = np.random.default_rng(seed)
    noise = generator.normal(0, .16, count)
    tone = sum(np.sin(2 * np.pi * frequency * time + seed) / (index + 1)
               for index, frequency in enumerate((360, 680, 1150, 1800, 2800)))
    envelope = (.45 + .35 * np.sin(2 * np.pi * 2.9 * time) ** 2)
    envelope *= np.minimum(1, time / .08) * np.minimum(1, np.maximum(0, seconds - time) / .1)
    samples = gain * (tone + noise) * envelope
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.clip(samples * 32767, -32767, 32767).astype("<i2").tobytes())
    return path


class SyntheticRecovery:
    recognizer = None

    def __init__(self, work):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def recover(self, text, initial=None, *, label="speech", **unused):
        self.calls.append((text, label))
        token = hashlib.sha256((label + text).encode()).hexdigest()[:12]
        path = _audio(self.work / (token + ".wav"), seed=len(self.calls))
        return path, [{"start": .12, "end": duration(path) - .12, "text": text}]

    def check(self, text, audio, *, label="speech"):
        # This is the explicitly mocked ASR boundary, never a real quality claim.
        return [{"start": .08, "end": max(.09, duration(audio) - .08), "text": text}]

    def release_voice(self):
        pass


def _quiet_music(path, seconds, action=False, *, theme=None):
    """Exercise real final mixing without depending on a bundled track decoder."""
    count = round(48000 * seconds)
    time = np.arange(count) / 48000
    signal = .00035 * np.sin(2 * np.pi * 140 * time)
    stereo = np.column_stack((signal, signal))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((stereo * 32767).astype("<i2").tobytes())


@pytest.fixture
def assets(tmp_path):
    source = tmp_path / "coherent-source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=14",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=14",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", source)
    image = tmp_path / "verified-episode.jpg"
    canvas = Image.new("RGB", (320, 180), "#142c4c")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((24, 25, 122, 152), fill="#b68c45")
    draw.ellipse((170, 30, 278, 140), fill="#5a9d9b")
    draw.line((10, 165, 302, 15), fill="white", width=3)
    canvas.save(image)
    result = {"seasons": {}, "sources": [], "warnings": []}
    for season in (1, 2, 3):
        common = {"season": season, "episode_ids": [f"e{season}"], "scope": "episode",
                  "illustrative": False, "title": f"Fixture Show season {season} verified episode",
                  "width": 320, "height": 180, "source_url": f"https://example.test/visual-{season}"}
        # Distinct paths preserve per-source provenance even with identical test content.
        video = tmp_path / f"season-{season}.mp4"
        shutil.copyfile(source, video)
        result["seasons"][str(season)] = [
            {**common, "path": str(video.resolve()), "kind": "video", "duration": 14,
             "shots": [{"start": .5, "end": 13.5, "motion": 2.0, "text_card": False}]},
            {**common, "path": str(image.resolve()), "kind": "image"},
        ]
        result["sources"].append({"season": season, "title": common["title"], "url": common["source_url"]})
    return result


@pytest.mark.parametrize("kind", ["season", "series"])
@media
def test_real_ffmpeg_complete_recap_packages_both_scopes(tmp_path, monkeypatch, assets, kind):
    output, work = tmp_path / "package", tmp_path / "work"
    recovery = SyntheticRecovery(tmp_path / "synthetic-voice")
    monkeypatch.setattr(render, "music_bed", _quiet_music)
    monkeypatch.setattr(rendering, "_make_recovery", lambda *a, **k: pytest.fail("Tests must never load speech models."))
    monkeypatch.setenv("VIDEO_RESOLUTION", "720")
    monkeypatch.setenv("TTS_MODEL", "nano")
    monkeypatch.setenv("GENERATED_SCORE", str(tmp_path / "no-generated-score.wav"))
    review_calls, decode_calls = [], []
    actual_review, actual_decode = audio_review.verify_export_audio, rendering._strict_decode

    def checked_review(video, timeline, speed, checker, directory):
        review_calls.append((video, timeline, speed, checker))
        assert all(Path(event["source_audio"]).is_file()
                   for event in timeline if event["type"] == "narration")
        return actual_review(video, timeline, speed, checker, directory)

    def checked_decode(path):
        decode_calls.append(path)
        return actual_decode(path)

    monkeypatch.setattr(audio_review, "verify_export_audio", checked_review)
    monkeypatch.setattr(rendering, "_strict_decode", checked_decode)
    script = _script(kind)
    report = rendering.render_recap(script, _spec(), _bundle(), assets, output, work, recovery=recovery)

    assert report["status"] == "passed"
    assert report["full_media_decode"] == "passed"
    assert report["audio_review"]["accepted"]
    assert len(review_calls) == 1 and len(decode_calls) == 2
    assert len(recovery.calls) == sum(len(section["paragraphs"]) for section in script["sections"])
    assert report["resolution"]["height"] <= 180
    assert report["final_speed"] == 1
    # Runtime never trims or pads the speech to satisfy Auto.
    expected = sum(event["duration"] for event in review_calls[0][1]) + 3
    assert abs(report["duration"] - expected) < .12
    assert report["end_screen_seconds"] == 10 and report["ending_hold_seconds"] == 3

    final = output / "final-video.mp4"
    render.validate_audio_video_duration(final, report["duration"])
    metadata = probe(final)
    audio = next(stream for stream in metadata["streams"] if stream["codec_type"] == "audio")
    assert audio["codec_name"] == "aac" and audio["sample_rate"] == "48000" and audio["channels"] == 2
    part_audio = next(stream for stream in probe(work / "parts/000-000/narration.mp4")["streams"]
                      if stream["codec_type"] == "audio")
    assert part_audio["codec_name"] == "alac" and part_audio["sample_rate"] == "48000"
    assert part_audio["bits_per_raw_sample"] == "24" and part_audio["channels"] == 2

    timeline = json.loads((output / "timeline.json").read_text())
    events = timeline["events"]
    expected_seasons = {2} if kind == "season" else {1, 2}
    assert {event["season"] for event in events if event.get("season")} == expected_seasons
    assert all(shot.get("season") in expected_seasons | {None}
               for event in events for shot in event["shots"])
    assert all(shot.get("season") != 3 for event in events for shot in event["shots"])
    assert events[-1]["type"] == "end-screen" and abs(events[-1]["duration"] - 10) < .04
    assert all(event["duration"] >= duration(Path(event["source_audio"])) - .04
               for event in events if event["type"] == "narration")

    chapters = json.loads(run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", final]))["chapters"]
    assert len(chapters) == len(report["chapters"])
    for actual, planned in zip(chapters, report["chapters"]):
        assert abs(float(actual["start_time"]) - planned["start"]) < .002
        assert actual["tags"]["title"] == planned["title"]
    assert chapters[-1]["tags"]["title"] == "Thanks for watching"

    srt = (output / "subtitles.srt").read_text()
    normalized_srt = " ".join(srt.split())
    assert all(" ".join(text.split()) in normalized_srt
               for section in script["sections"] for text in section["paragraphs"])
    assert "Mara" in srt and "-->" in srt
    description = (output / "description.txt").read_text()
    assert all(p in description for p in [p for s in script["sections"] for p in s["paragraphs"]][:2])
    assert "#FixtureShow" in description and "#PreviouslyOn" in description
    for chapter in report["youtube_chapters"]:
        assert f"{rendering._youtube_time(chapter['start'])} {chapter['title']}" in description
    assert "visual-3" not in (output / "source-credits.txt").read_text()
    assert (output / "script.json").is_file() and (output / "recommended-title.txt").is_file()
    assert json.loads((output / "production-report.json").read_text())["status"] == "passed"
    for index in (1, 2, 3):
        with Image.open(output / f"thumbnail-{index:02}.jpg") as thumbnail:
            assert thumbnail.size == (1280, 720) and thumbnail.format == "JPEG"
    assert "nano" == __import__("os").environ["TTS_MODEL"]
    assert "720" == __import__("os").environ["VIDEO_RESOLUTION"]


@media
def test_actual_ffmpeg_moving_still_is_smooth_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_RESOLUTION", "320x180")
    still = tmp_path / "point.png"
    image = Image.new("RGB", (320, 180), "black")
    ImageDraw.Draw(image).rectangle((38, 75, 65, 104), fill="white")
    image.save(still)
    audio = _audio(tmp_path / "sound.wav", seconds=3)
    output = tmp_path / "moving-still.mp4"
    render.speech_video(still, audio, [{"source": str(still), "start": 0, "duration": 3,
        "image": True, "still": True, "still_motion": "zoom_in"}], output, None, tmp_path / "render")
    render.validate_audio_video_duration(output, 3)
    pattern = tmp_path / "motion-%02d.png"
    ffmpeg("-i", output, "-vf", "select=not(mod(n\\,15))", "-vsync", "0", pattern)
    centroids = []
    for path in sorted(tmp_path.glob("motion-*.png")):
        with Image.open(path) as frame:
            light = np.asarray(frame.convert("L"), dtype=float)
        light = np.maximum(0, light - 30)
        centroids.append(float((light.sum(axis=0) * np.arange(320)).sum() / light.sum()))
    assert len(centroids) == 6
    # The off-centre point drifts continuously; whole-pixel crop jumps would fail.
    assert centroids[0] - centroids[-1] > .5
    changes = np.diff(centroids)
    assert np.all(changes <= .12) and max(abs(changes)) < 1.8


@pytest.mark.parametrize("shot", [
    {"season": 3, "episode_ids": ["e3"], "scope": "episode"},
    {"season": 1, "episode_ids": ["e1"], "scope": "episode"},
    {"season": 2, "episode_ids": ["unknown"], "scope": "episode"},
    {"season": 2, "episode_ids": ["e2", "other"], "scope": "episode"},
    {"season": None, "episode_ids": [], "scope": "season"},
])
def test_timeline_guard_rejects_future_wrong_and_unknown_episodes(shot):
    script = _script("season")
    event = {"type": "narration", "section_id": "story-s2", "episode_ids": ["e2"], "shots": [shot]}
    with pytest.raises(PipelineError):
        rendering._validate_timeline([event], script, _spec(), _bundle())


def test_native_guard_requires_every_source_episode_to_be_the_current_episode():
    bundle = _bundle()
    bundle["seasons"][1]["episodes"][0]["number"] = 2
    bundle["sources"][1]["episode"] = 2
    bundle["sources"].append({"id": "s2-old", "season": 2, "episode": 1,
                              "title": "Earlier episode", "url": "https://example.test/earlier"})
    bundle["seasons"][1]["episodes"].append(
        {"id": "e2-other", "season": 2, "number": 1, "summary": "Earlier supported event.", "sources": ["s2-old"]})
    event = {"type": "trailer-audio", "section_id": "story-s2", "episode_ids": ["e2"],
             "shots": [{"season": 2, "episode_ids": ["e2", "e2-other"], "scope": "episode"}]}
    with pytest.raises(PipelineError):
        rendering._validate_timeline([event], _script("season"), _spec(), bundle)


def test_paragraph_provenance_uses_current_season_only_and_ambiguous_legacy_uses_generic():
    section = {"season": 2, "episode_ids": ["e1", "e2", "e2-other"],
               "paragraph_episode_ids": [["e1", "e2"], ["e2-other"]]}
    episode_seasons = {"e1": 1, "e2": 2, "e2-other": 2}
    assert rendering._paragraph_episodes(section, 0, episode_seasons) == ["e2"]
    assert rendering._paragraph_episodes(section, 1, episode_seasons) == ["e2-other"]
    section.pop("paragraph_episode_ids")
    assert rendering._paragraph_episodes(section, 0, episode_seasons) == []


@media
def test_final_audio_guard_failure_never_publishes_a_production_video(tmp_path, monkeypatch, assets):
    monkeypatch.setattr(render, "music_bed", _quiet_music)
    calls = []
    def refuse(video, events, speed, recovery, work):
        calls.append(events)
        return {"accepted": False, "status": "failed", "failed_events": [0]}
    monkeypatch.setattr(audio_review, "verify_export_audio", refuse)
    output = tmp_path / "package"
    output.mkdir()
    (output / "final-video.mp4").write_bytes(b"stale prior failed-run fixture")
    with pytest.raises(PipelineError, match="did not preserve"):
        rendering.render_recap(_script("season"), _spec(), _bundle(), assets, output, tmp_path / "work",
                              recovery=SyntheticRecovery(tmp_path / "voice"))
    assert len(calls) == 1 and calls[0][0]["source_audio"]
    assert not (output / "final-video.mp4").exists()
    assert json.loads((output / "production-report.json").read_text())["status"] == "failed"


@media
def test_optional_native_media_failure_keeps_complete_narration(tmp_path, monkeypatch, assets):
    monkeypatch.setattr(render, "music_bed", _quiet_music)
    monkeypatch.setenv("DIALOGUE_SEPARATION", "false")
    source = assets["seasons"]["2"][0]["path"]
    calls = []
    def optional_beat(section, paragraph, pool, usage, **kwargs):
        if section["id"] != "story-s2":
            return None
        return {"source": source, "start": 8, "duration": 3, "season": 2,
                "episode_ids": ["e2"], "episode_id": "e2", "scope": "episode",
                "text": "Mara closes the gate.", "native_verified": True}
    def damaged_native(*args, **kwargs):
        calls.append(args)
        raise PipelineError("Synthetic optional native-audio decode failure.")
    monkeypatch.setattr(rendering, "plan_source_audio", optional_beat)
    monkeypatch.setattr(render, "trailer_video", damaged_native)
    output = tmp_path / "package"
    report = rendering.render_recap(_script("season"), _spec(), _bundle(), assets, output, tmp_path / "work",
                                    recovery=SyntheticRecovery(tmp_path / "voice"))
    assert calls and report["status"] == "passed"
    assert any("source-audio beat was skipped" in warning for warning in report["warnings"])
    events = json.loads((output / "timeline.json").read_text())["events"]
    assert not any(event["type"] == "trailer-audio" for event in events)
    assert sum(event["type"] == "narration" for event in events) == 4



def test_short_description_chapters_merge_without_padding_the_actual_timeline():
    chapters = [{"start": 0, "title": "Intro"}, {"start": 6, "title": "Season 1"},
                {"start": 30, "title": "Where we left off"}, {"start": 35, "title": "Outro"}]
    result = rendering._description_chapters(chapters, 49)
    assert result == [{"start": 0, "title": "Intro / Season 1"},
                      {"start": 30, "title": "Where we left off / Outro"}]
    assert chapters[1]["start"] == 6
