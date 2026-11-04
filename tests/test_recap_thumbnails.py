import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageStat

from movie_pipeline.core import PipelineError
from recap_pipeline import thumbnails


def bundle():
    return {
        "show": {"id": 44, "title": "Northern Lights", "year": 2020},
        "seasons": [
            {"number": season, "episodes": [{"id": f"ep-{season}", "season": season}]}
            for season in range(1, 5)
        ],
    }


def script(kind="season"):
    return {"kind": kind, "seasons": [3] if kind == "season" else [1, 2, 3],
            "upcoming_season": 4, "show_title": "Northern Lights"}


def source(path: Path, color="#3a6494"):
    image = Image.new("RGB", (1600, 900), color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((780, 100, 1350, 790), fill="#dda579")
    draw.rectangle((880, 340, 1260, 850), fill="#152944")
    draw.line((0, 250, 1600, 600), fill="#b8d9ef", width=20)
    image.save(path)
    return path


def asset(path, season=3, **kwargs):
    return {"kind": "image", "path": str(path), "season": season,
            "source_url": f"https://example.org/episode-{season}.jpg",
            "title": f"Episode still season {season}", "episode_ids": [f"ep-{season}"],
            **kwargs}


def assets(*rows):
    result = {"seasons": {}, "sources": [], "warnings": []}
    for row in rows:
        result["seasons"].setdefault(str(row["season"]), []).append(row)
    return result


def report(output):
    return json.loads((output / "thumbnail-report.json").read_text())


def test_three_finished_jpegs_with_genuine_source_and_recap_titles(tmp_path):
    still = source(tmp_path / "episode.jpg")
    output = tmp_path / "thumbs"
    paths = thumbnails.build_thumbnails(script(), bundle(), assets(asset(still)), output)
    assert len(paths) == 3
    assert len({path.read_bytes() for path in paths}) == 3
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.size == (1280, 720)
            assert image.mode == "RGB"
            assert max(ImageStat.Stat(image).stddev) > 25
        assert 8_000 < path.stat().st_size < 2_000_000
    rows = report(output)["thumbnails"]
    assert {row["headline"] for row in rows} == {"Season 3 Recap", "Before Season 4"}
    assert all(row["source"]["path"] == str(still) for row in rows)
    assert all(row["scope"] == [3] for row in rows)
    assert all("TOP " not in row["headline"] for row in rows)


def test_only_requested_season_can_supply_images(tmp_path):
    old = source(tmp_path / "old.jpg")
    prior = source(tmp_path / "prior.jpg", "#872d44")
    future = source(tmp_path / "future.jpg", "#228422")
    output = tmp_path / "thumbs"
    thumbnails.build_thumbnails(script(), bundle(), assets(asset(old, 1), asset(prior, 3), asset(future, 4)), output)
    assert {row["source"]["path"] for row in report(output)["thumbnails"]} == {str(prior)}


def test_wrong_episode_owner_and_mismatched_asset_season_are_rejected(tmp_path):
    image = source(tmp_path / "wrong.jpg")
    wrong_episode = asset(image, 3, episode_ids=["ep-4"])
    mismatched = asset(image, 4)
    listing = {"seasons": {"3": [wrong_episode, mismatched]}}
    with pytest.raises(PipelineError, match="No readable, verified"):
        thumbnails.build_thumbnails(script(), bundle(), listing, tmp_path / "thumbs")


@pytest.mark.parametrize("replacement", [
    {"seasons": [4]},
    {"seasons": [2]},
    {"seasons": []},
    {"show_title": "A Different Show"},
])
def test_invalid_or_ambiguous_scope_is_rejected_before_source_use(tmp_path, replacement):
    request = {**script(), **replacement}
    with pytest.raises(PipelineError):
        thumbnails.build_thumbnails(request, bundle(), {"seasons": {}}, tmp_path / "thumbs")


def test_series_recap_never_uses_incoming_material(tmp_path):
    prior = source(tmp_path / "prior.jpg")
    incoming = source(tmp_path / "incoming.jpg")
    output = tmp_path / "thumbs"
    thumbnails.build_thumbnails(script("series"), bundle(),
                                assets(asset(prior, 2), asset(incoming, 4)), output)
    rows = report(output)["thumbnails"]
    assert all(row["source"]["season"] < 4 for row in rows)
    assert all(row["scope"] == [1, 2, 3] for row in rows)
    assert "Seasons 1-3 Recap" in {row["headline"] for row in rows}


def test_video_frames_are_local_and_ffmpeg_is_mocked(tmp_path, monkeypatch):
    video = tmp_path / "scoped.mp4"
    video.write_bytes(b"fixture")
    calls = []

    def fake_ffmpeg(*args):
        calls.append(args)
        source(Path(args[-1]))

    monkeypatch.setattr(thumbnails, "ffmpeg", fake_ffmpeg)
    row = asset(video, kind="video", duration=90.0)
    output = tmp_path / "thumbs"
    paths = thumbnails.build_thumbnails(script(), bundle(), assets(row), output)
    assert len(paths) == 3
    assert len(calls) == 2
    assert all(args[args.index("-i") + 1] == video for args in calls)
    assert all("-frames:v" in args for args in calls)
    assert {row["source"]["frame_second"] for row in report(output)["thumbnails"]} == {25.2, 55.8}
    assert not list(output.glob("recap-thumbnails-*"))


def test_real_episode_image_is_preferred_to_illustrative_title_card(tmp_path):
    title_card = source(tmp_path / "title-card.jpg")
    episode = source(tmp_path / "episode.jpg", "#923239")
    output = tmp_path / "thumbs"
    thumbnails.build_thumbnails(script(), bundle(),
                                assets(asset(title_card, illustrative=True, episode_ids=[], scope="neutral-title-card"),
                                       asset(episode)), output)
    assert all(row["source"]["path"] == str(episode) for row in report(output)["thumbnails"])


def test_illustrative_fallback_is_labelled_and_provenance_retained(tmp_path, monkeypatch):
    plate = source(tmp_path / "neutral-title-card.jpg")
    drawn = []
    original = ImageDraw.ImageDraw.text

    def record_text(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    output = tmp_path / "thumbs"
    thumbnails.build_thumbnails(script(), bundle(),
                                assets(asset(plate, illustrative=True, episode_ids=[], scope="neutral-title-card")), output)
    assert sum("NOT AN EXACT SEASON SCENE" in value for value in drawn) == 3
    assert all(row["source"]["illustrative"] for row in report(output)["thumbnails"])
    assert all(row["source"]["scope"] == "neutral-title-card" for row in report(output)["thumbnails"])


def test_zero_or_unreadable_assets_fail_without_invented_show_images(tmp_path):
    output = tmp_path / "thumbs"
    with pytest.raises(PipelineError, match="cannot invent"):
        thumbnails.build_thumbnails(script(), bundle(), {"seasons": {}}, output)
    assert not list(output.glob("*.jpg"))
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    with pytest.raises(PipelineError, match="No readable, verified"):
        thumbnails.build_thumbnails(script(), bundle(), assets(asset(broken)), output)


def test_poisoned_research_episode_season_cannot_supply_thumbnail(tmp_path):
    still = source(tmp_path / "wrong-owner.jpg")
    research = bundle()
    research["seasons"][2]["episodes"][0]["season"] = 4
    with pytest.raises(PipelineError, match="No readable, verified"):
        thumbnails.build_thumbnails(script(), research, assets(asset(still)), tmp_path / "thumbs")


def test_collector_fallback_warnings_survive_thumbnail_report(tmp_path):
    plate = source(tmp_path / "plate.jpg")
    listing = assets(asset(plate, illustrative=True, episode_ids=[], scope="neutral-title-card"))
    listing["warnings"] = ["Season 3: verified season imagery was unavailable."]
    output = tmp_path / "thumbs"
    thumbnails.build_thumbnails(script(), bundle(), listing, output)
    assert report(output)["warnings"] == listing["warnings"]
