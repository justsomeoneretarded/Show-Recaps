from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageStat

from movie_pipeline.core import PipelineError
from recap_pipeline import assets


@pytest.fixture
def spec():
    return SimpleNamespace(show_title="North Star", premiere_year=2020,
                           upcoming_season=2, try_4k=False)


@pytest.fixture
def bundle():
    return {
        "show": {"id": 90, "title": "North Star", "year": 2020,
                 "genres": ["Drama"], "imdb_id": "tt1234567",
                 "image": {"original": "https://images.example/current-poster.jpg"}},
        "seasons": [
            {"number": 1, "year": 2020, "episodes": [
                {"id": "ep1", "season": 1, "number": 1, "title": "Arrival",
                 "airdate": "2020-01-01", "summary": "Mara reaches town.",
                 "image": {"original": "https://images.example/ep1.jpg"}, "sources": ["src1"]}]},
            {"number": 2, "year": 2021, "episodes": [
                {"id": "incoming", "season": 2, "number": 1, "title": "Spoiler",
                 "airdate": "2021-01-01", "summary": "Incoming plot must stay out.",
                 "image": {"original": "https://images.example/incoming.jpg"}, "sources": ["future"]}]},
        ]}


def info(title="North Star Season 1 Official Trailer", **changes):
    return {"title": title, "duration": 95, "description": "North Star (2020).", **changes}


@pytest.mark.parametrize("candidate,okay", [
    (info(), True),
    (info("North Star S01E02 Best Scene"), True),
    (info("North Star first season official trailer"), True),
    (info("North Star Official Trailer", description="North Star (2020), season 1"), True),
    (info("North Star Season 1 Official Trailer", description="North Star season 1"), True),
    (info("North Star Season 1 Official Trailer", description="", imdb_id="tt1234567"), True),
    (info("North Star Official Trailer"), False),
    (info("North Star Season 2 Official Trailer"), False),
    (info("North Star Seasons 1 and 2 Best Scenes"), False),
    (info("North Star S01-S02 Best Scenes"), False),
    (info("North Star Seasons 1, 2 Best Scenes"), False),
    (info("North Star Season 1 Official Trailer", description="Footage from season 2."), False),
    (info("Another Show Season 1 Official Trailer"), False),
    (info("North Star Adventures Season 1 Trailer"), False),
    (info("North Star Season 1 Trailer (1999)"), False),
    (info("North Star Season 1 Concept Trailer"), False),
    (info("North Star Season 1 Official Trailer", description="A fan-made concept trailer."), False),
    (info("North Star Season 1 Official Trailer", description="A remake of the original series."), False),
    (info("North Star Season 1 Official Trailer", description="North Star is a 1999 television series."), False),
    (info("North Star Season 1 Review and Trailer"), False),
    (info("North Star Season 1 Full Episode Clip"), False),
    (info("North Star Season 1 Trailer", series="Different Show"), False),
    (info("North Star Season 1 Trailer", imdb_id="tt0000000"), False),
    (info("North Star Season 1 Trailer", season_number=2), False),
    (info("North Star Season 1 Trailer", duration=1500), False),
    (info("North Star Season 1 Trailer", duration=float("nan")), False),
])
def test_metadata_is_identity_and_season_evidence(bundle, candidate, okay):
    assert assets.candidate_ok(candidate, bundle["show"], 1, 2, 2020) is okay


def test_short_name_does_not_match_different_show():
    assert not assets.candidate_ok(
        info("Dark Matter Season 1 Trailer"), {"title": "Dark", "year": 2017}, 1, 2, 2017)


def test_incoming_season_is_never_a_candidate(bundle):
    assert not assets.candidate_ok(info("North Star Season 2 Trailer"), bundle["show"], 2, 2, 2021)


def test_queries_always_name_one_exact_season(bundle):
    bundle["show"]["genres"] = ["Comedy"]
    queries = assets.search_queries(bundle["show"], 1)
    assert len(queries) == 3
    assert all('"North Star"' in query and '"season 1"' in query for query in queries)
    assert any("official trailer" in query for query in queries)
    assert any("best scenes" in query for query in queries)
    assert any("funniest scenes" in query for query in queries)


def _jpeg():
    output = io.BytesIO()
    image = Image.new("RGB", (640, 360), "#20405e")
    for x in range(320, 640):
        for y in range(360):
            image.putpixel((x, y), (150, 80 + y % 90, 40))
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()


class Response:
    def __init__(self, body, declared=None):
        self.body = body
        self.headers = {"Content-Length": str(len(body) if declared is None else declared)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]


def no_video(monkeypatch):
    monkeypatch.setattr(assets, "require_disk", lambda *args: None)
    calls = []

    def discover(*args):
        calls.append(args[1]["number"])
        return []
    monkeypatch.setattr(assets, "_discover_videos", discover)
    return calls


def test_prior_episode_image_fallback_and_cross_recap_cache(monkeypatch, tmp_path, bundle, spec):
    searches = no_video(monkeypatch)
    requests = []

    def get(url, **kwargs):
        requests.append(url)
        return Response(_jpeg())
    monkeypatch.setattr(assets.requests, "get", get)
    season_recap = assets.collect_assets(bundle, spec, tmp_path)
    series_recap = assets.collect_assets(bundle, spec, tmp_path)
    assert searches == [1]
    assert requests == ["https://images.example/ep1.jpg"]
    assert set(season_recap["seasons"]) == {"1"}
    item = season_recap["seasons"]["1"][0]
    assert item["episode_ids"] == ["ep1"]
    assert item["scope"] == "episode" and not item["illustrative"]
    assert item["path"] == series_recap["seasons"]["1"][0]["path"]
    assert season_recap["sources"][0]["season"] == 1
    assert any("cached" in message for message in series_recap["warnings"])
    with Image.open(item["path"]) as image:
        assert image.format == "JPEG"
        assert image.size == (640, 360)


def test_future_or_wrong_season_images_and_unscoped_poster_are_excluded(monkeypatch, tmp_path, bundle, spec):
    no_video(monkeypatch)
    original = bundle["seasons"][0]["episodes"][0]
    original["season"] = 2
    future = copy.deepcopy(original)
    future.update(id="not-aired", season=1, airdate="2099-01-01")
    bundle["seasons"][0]["episodes"].append(future)
    undated = copy.deepcopy(original)
    undated.update(id="undated", season=1, airdate=None)
    bundle["seasons"][0]["episodes"].append(undated)
    monkeypatch.setattr(assets.requests, "get",
                        lambda *args, **kwargs: pytest.fail("Unsafe image URL was requested."))
    result = assets.collect_assets(bundle, spec, tmp_path)
    item = result["seasons"]["1"][0]
    assert item["scope"] == "neutral-title-card"
    assert item["episode_ids"] == [] and item["illustrative"]
    assert item["source_url"] == "" and result["sources"] == []
    assert any("poster was excluded" in warning for warning in result["warnings"])
    with Image.open(item["path"]) as image:
        assert image.size == (1280, 720)
        assert ImageStat.Stat(image).stddev[0] > 10


def test_network_failure_still_creates_honest_neutral_plate(monkeypatch, tmp_path, bundle, spec):
    no_video(monkeypatch)

    def failed(*args, **kwargs):
        raise assets.requests.ConnectionError("provider unavailable")
    monkeypatch.setattr(assets.requests, "get", failed)
    result = assets.collect_assets(bundle, spec, tmp_path)
    assert result["seasons"]["1"][0]["scope"] == "neutral-title-card"
    assert any("no verified season imagery" in warning for warning in result["warnings"])


def test_only_explicit_prior_season_poster_can_be_illustrative_fallback(monkeypatch, tmp_path, bundle, spec):
    no_video(monkeypatch)
    bundle["seasons"][0]["episodes"] = []
    bundle["show"]["image"]["season"] = 1
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: Response(_jpeg()))
    result = assets.collect_assets(bundle, spec, tmp_path)
    item = result["seasons"]["1"][0]
    assert item["scope"] == "season-poster" and item["illustrative"]
    assert any("does not establish any exact event" in warning for warning in result["warnings"])


def test_poisoned_cache_cannot_relabel_future_footage(monkeypatch, tmp_path, bundle, spec):
    searches = no_video(monkeypatch)
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: Response(_jpeg()))
    assets.collect_assets(bundle, spec, tmp_path)
    manifest = next(tmp_path.rglob("assets.json"))
    record = json.loads(manifest.read_text())
    record["assets"][0]["season"] = 2
    manifest.write_text(json.dumps(record))
    result = assets.collect_assets(bundle, spec, tmp_path)
    assert searches == [1, 1]
    assert result["seasons"]["1"][0]["season"] == 1


def test_wrong_episode_image_url_invalidates_cache(monkeypatch, tmp_path, bundle, spec):
    searches = no_video(monkeypatch)
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: Response(_jpeg()))
    assets.collect_assets(bundle, spec, tmp_path)
    manifest = next(tmp_path.rglob("assets.json"))
    record = json.loads(manifest.read_text())
    record["assets"][0]["source_url"] = "https://images.example/incoming.jpg"
    manifest.write_text(json.dumps(record))
    result = assets.collect_assets(bundle, spec, tmp_path)
    assert searches == [1, 1]
    assert result["seasons"]["1"][0]["source_url"] == "https://images.example/ep1.jpg"


def test_show_and_premiere_year_are_checked_before_requests(monkeypatch, tmp_path, bundle, spec):
    monkeypatch.setattr(assets, "_discover_videos", lambda *args: pytest.fail("Wrong identity searched."))
    spec.show_title = "Another Show"
    with pytest.raises(PipelineError, match="identity"):
        assets.collect_assets(bundle, spec, tmp_path)
    spec.show_title = "North Star"
    spec.premiere_year = 1999
    with pytest.raises(PipelineError, match="identity"):
        assets.collect_assets(bundle, spec, tmp_path)


def test_downloads_only_verified_candidates_and_reuses_probed_cache(monkeypatch, tmp_path, bundle, spec):
    import yt_dlp
    monkeypatch.setattr(assets, "require_disk", lambda *args: None)
    monkeypatch.setattr(assets.download_access, "routes", lambda: ["direct"])
    monkeypatch.setattr(assets.download_access, "options", lambda base, route: dict(base))
    monkeypatch.setattr(assets.download_access, "record_route_success", lambda route: None)
    metadata = {
        "abcde000001": info("North Star Season 2 Official Trailer"),
        "abcde000002": info("Another Show Season 1 Official Trailer"),
        "abcde000003": info("North Star Season 1 Official Trailer"),
        "abcde000004": info("North Star S01E02 Best Scene"),
    }
    downloaded, ceilings = [], []

    class YDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, query, download=False):
            if query.startswith("ytsearch"):
                return {"entries": [{"id": key} for key in metadata]}
            return metadata[query.split("=")[-1]]

        def download(self, urls):
            downloaded.extend(urls)
            ceilings.append(self.options["format"])
            path = Path(self.options["outtmpl"].replace("%(ext)s", "mp4"))
            path.write_bytes(b"mock footage")
            for hook in self.options["progress_hooks"]:
                hook({"filename": str(path), "downloaded_bytes": path.stat().st_size})

    monkeypatch.setattr(yt_dlp, "YoutubeDL", YDL)
    monkeypatch.setattr(assets, "probe", lambda path: {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
        "format": {"duration": "95.0"}})
    first = assets.collect_assets(bundle, spec, tmp_path)
    second = assets.collect_assets(bundle, spec, tmp_path)
    assert len(downloaded) == 2
    assert [url.split("=")[-1] for url in downloaded] == ["abcde000003", "abcde000004"]
    assert all("1080" in selection and "2160" not in selection for selection in ceilings)
    assert [item["kind"] for item in first["seasons"]["1"]] == ["video", "video"]
    assert all(item["duration"] == 95 and item["width"] == 1920 for item in second["seasons"]["1"])
    assert not list(tmp_path.rglob("footage-*"))


def test_optional_login_block_uses_images_without_requiring_cookies(monkeypatch, tmp_path, bundle, spec):
    import yt_dlp
    monkeypatch.setattr(assets, "require_disk", lambda *args: None)
    monkeypatch.setattr(assets.download_access, "routes", lambda: ["direct"])
    options_seen = []

    class BlockedYDL:
        def __init__(self, options):
            options_seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", BlockedYDL)
    monkeypatch.delenv("YOUTUBE_COOKIE_FILE", raising=False)
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: Response(_jpeg()))
    result = assets.collect_assets(bundle, spec, tmp_path)
    assert result["seasons"]["1"][0]["scope"] == "episode"
    assert all("cookiefile" not in options for options in options_seen)
    assert len(options_seen) == 1


def test_image_declared_size_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "require_disk", lambda *args: None)
    monkeypatch.setattr(assets.requests, "get",
                        lambda *args, **kwargs: Response(b"not read", assets.MAX_IMAGE_BYTES + 1))
    with pytest.raises(PipelineError, match="byte budget"):
        assets._fetch_image("https://images.example/a.jpg", tmp_path / "image.jpg", assets._Budget())
    assert not (tmp_path / "image.jpg").exists()


def test_video_probe_rejects_full_episode_and_oversized_file(monkeypatch, tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr(assets, "probe", lambda path: {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
        "format": {"duration": "1800"}})
    with pytest.raises(PipelineError, match="duration"):
        assets._video_info(path)
    monkeypatch.setattr(assets, "MAX_VIDEO_BYTES", 2)
    monkeypatch.setattr(assets, "probe", lambda path: {
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
        "format": {"duration": "95"}})
    with pytest.raises(PipelineError, match="per-source"):
        assets._video_info(path)


def test_source_download_attempts_are_bounded_even_if_every_probe_fails(monkeypatch, tmp_path, bundle, spec):
    import yt_dlp
    monkeypatch.setattr(assets, "require_disk", lambda *args: None)
    monkeypatch.setattr(assets.download_access, "routes", lambda: ["direct"])
    calls = []

    class YDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, query, download=False):
            if query.startswith("ytsearch"):
                return {"entries": [{"id": f"abcde{number:06}"} for number in range(4)]}
            return info()

        def download(self, urls):
            calls.extend(urls)
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"bad video")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", YDL)
    monkeypatch.setattr(assets, "probe", lambda path: {"streams": [], "format": {"duration": "95"}})
    monkeypatch.setattr(assets.requests, "get", lambda *args, **kwargs: Response(_jpeg()))
    result = assets.collect_assets(bundle, spec, tmp_path)
    assert len(calls) == assets.MAX_DOWNLOAD_ATTEMPTS
    assert result["seasons"]["1"][0]["kind"] == "image"
    assert not list(tmp_path.rglob("*.mp4"))


def test_same_title_remake_ambiguity_requires_positive_metadata_identity(bundle):
    show = {**bundle["show"], "ambiguous_title": True}
    ambiguous = info("North Star Season 1 Trailer", description="Season 1 official trailer.")
    assert not assets.candidate_ok(ambiguous, show, 1, 2, 2020)
    assert assets.candidate_ok(info(), show, 1, 2, 2020)
    assert assets.candidate_ok({**ambiguous, "imdb_id": show["imdb_id"]}, show, 1, 2, 2020)
    assert assets._identity_evidence(info(), show) == "premiere-year"
    assert assets._identity_evidence({**ambiguous, "imdb_id": show["imdb_id"]}, show) == "provider-id"
    assert assets._identity_evidence(ambiguous, bundle["show"]) == "exact-title-and-season"


def test_season_release_year_is_not_mistaken_for_a_conflicting_premiere(bundle):
    show = bundle["show"]
    candidate = info("North Star Season 2 Official Trailer (2021)", description="")
    assert assets.candidate_ok(candidate, show, 2, 3, 2021)
    assert not assets.candidate_ok(candidate, {**show, "ambiguous_title": True}, 2, 3, 2021)
    identified = {**candidate, "description": "North Star (2020), season 2."}
    assert assets.candidate_ok(identified, {**show, "ambiguous_title": True}, 2, 3, 2021)
    assert assets._identity_evidence(identified, show) == "premiere-year"
