"""Stage wiring and artifact portability, with all costly services mocked."""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from movie_pipeline.core import PipelineError, save_json
from recap_pipeline import cli
from recap_pipeline.models import RecapSpec


def _bundle():
    return {"show": {"id": 1, "title": "Example", "year": 2020},
            "seasons": [{"number": n, "episodes": [
                {"id": f"e{n}", "season": n, "number": 1, "summary": "A supported event.",
                 "sources": [f"s{n}"]}]} for n in (1, 2)],
            "sources": [{"id": f"s{n}", "season": n} for n in (1, 2)], "warnings": []}


def _script(kind):
    numbers = [2] if kind == "season" else [1, 2]
    rows = [{"id": f"story-{n}", "kind": "story", "season": n, "title": "The story",
             "paragraphs": ["A supported event."], "episode_ids": [f"e{n}"],
             "event_ids": [f"v{n}"]} for n in numbers]
    rows.append({**deepcopy(rows[-1]), "id": "ending", "kind": "ending"})
    return {"kind": kind, "title": "Example recap", "description": "Sourced recap.",
            "show_title": "Example", "upcoming_season": 3, "seasons": numbers,
            "sections": rows, "source_ids": [f"s{n}" for n in numbers], "warnings": []}


@pytest.fixture
def services(monkeypatch, tmp_path):
    events, renders, footages = [], [], []
    for name, value in {"SHOW_TITLE": "Example", "SHOW_YEAR": "2020", "UPCOMING_SEASON": "3",
                        "RECAP_BRIEF": "Calm prose.", "NARRATOR": "Warguy", "TTS_MODEL": "original",
                        "TRY_4K": "true", "SEASON_MINUTES": "Auto", "SERIES_MINUTES": "Auto"}.items():
        monkeypatch.setenv(name, value)

    def research(spec, cache):
        events.append(("research", spec, cache))
        return _bundle()

    class Writer:
        def __init__(self, cache):
            events.append(("writer", cache))

        def unload(self):
            events.append(("unload",))

    def ledger(bundle, cache, writer):
        events.append(("ledger",))
        return {"events": [{"id": "v1"}, {"id": "v2"}]}

    def write(kind, spec, bundle, ledger, cache, writer, footage=None):
        events.append(("write", kind))
        footages.append(footage)
        return _script(kind)

    def collect(bundle, spec, cache):
        events.append(("assets",))
        cache.mkdir(parents=True, exist_ok=True)
        image = cache / "still.jpg"
        image.write_bytes(b"image")
        return {"seasons": {"1": [{"path": str(image.resolve()), "season": 1, "kind": "image"}],
                            "2": [{"path": str(image.resolve()), "season": 2, "kind": "image"}]},
                "sources": [], "warnings": []}

    def render(script, spec, bundle, assets, output, work, recovery=None):
        events.append(("render", script["kind"]))
        renders.append((script, spec, assets, output, work, recovery))
        return output

    monkeypatch.setitem(sys.modules, "recap_pipeline.research", SimpleNamespace(research_show=research))
    monkeypatch.setitem(sys.modules, "recap_pipeline.writing",
                        SimpleNamespace(RecapWriter=Writer, build_ledger=ledger, write_recap=write,
                                        apply_packaging=lambda script, client: script))
    def link(collected, ledger, client):
        events.append(("link",))
        return {"v2": 12.5}

    monkeypatch.setitem(sys.modules, "recap_pipeline.assets",
                        SimpleNamespace(collect_assets=collect, link_footage=link))
    monkeypatch.setitem(sys.modules, "recap_pipeline.rendering", SimpleNamespace(render_recap=render))
    return SimpleNamespace(events=events, renders=renders, footages=footages, shared=tmp_path / "shared",
                           work=tmp_path / "render", output=tmp_path / "output")


def test_prepare_researches_once_writes_exactly_two_scripts_and_collects_once(services):
    result = cli.run_stage("prepare", services.shared, services.work, services.output)
    assert set(result) == {"season", "series"}
    # Story events first, then footage linked to them, then scripts that know what can be shown.
    assert [event[0] for event in services.events] == [
        "research", "writer", "ledger", "unload", "assets", "link",
        "writer", "ledger", "write", "write", "unload"]
    assert services.footages == [{"v2": 12.5}, {"v2": 12.5}]
    assert [event[1] for event in services.events if event[0] == "write"] == ["season", "series"]
    assert {path.name for path in services.shared.glob("*.json")} == {
        "spec.json", "research.json", "story-ledger.json", "season-script.json", "series-script.json", "assets.json",
        "footage.json"}
    assets = json.loads((services.shared / "assets.json").read_text())
    assert assets["seasons"]["1"][0]["path"] == "asset-cache/still.jpg"
    assert "TELEGRAM" not in (services.shared / "spec.json").read_text()
    assert services.renders == []


@pytest.mark.parametrize("kind", ["season", "series"])
def test_each_render_creates_only_selected_kind_and_uses_frozen_spec(services, monkeypatch, kind):
    cli.run_stage("prepare", services.shared, services.work, services.output)
    monkeypatch.setenv("SHOW_TITLE", "A different show")
    monkeypatch.setenv("NARRATOR", "Female")
    monkeypatch.setenv("TTS_MODEL", "nano")
    recovery = services.work / "recovered"
    cli.run_stage("render", services.shared, services.work, services.output, kind=kind, recovery=recovery)
    assert len(services.renders) == 1
    script, spec, assets, output, work, recovered = services.renders[0]
    assert script["kind"] == kind
    assert spec.show_title == "Example" and spec.tts_model == "original" and spec.narrator == "Warguy"
    assert output == services.output / kind and work == services.work / kind and recovered == recovery
    assert __import__("os").environ["TTS_MODEL"] == "original"
    assert __import__("os").environ["NARRATOR"] == "Warguy"
    assert __import__("pathlib").Path(assets["seasons"]["2"][0]["path"]).is_absolute()
    assert [e[0] for e in services.events][-3:] == ["writer", "unload", "render"]


def test_shared_artifact_can_move_to_another_runner_directory(services, tmp_path):
    import shutil

    cli.run_stage("prepare", services.shared, services.work, services.output)
    moved = tmp_path / "new-runner" / "shared"
    shutil.copytree(services.shared, moved)
    shutil.rmtree(services.shared)
    cli.run_stage("render", moved, services.work, services.output, kind="series")
    assets = services.renders[-1][2]
    assert assets["seasons"]["2"][0]["path"] == str((moved / "asset-cache/still.jpg").resolve())


def test_individual_stages_have_no_surprise_render_or_research(services):
    cli.run_stage("research", services.shared, services.work, services.output)
    assert [event[0] for event in services.events] == ["research"]
    cli.run_stage("write", services.shared, services.work, services.output)
    cli.run_stage("assets", services.shared, services.work, services.output)
    assert sum(event[0] == "research" for event in services.events) == 1
    assert not services.renders


@pytest.mark.parametrize("path", ["../outside.jpg", "/tmp/absolute.jpg"])
def test_render_rejects_nonportable_or_escaping_asset_paths(services, path):
    cli.run_stage("prepare", services.shared, services.work, services.output)
    save_json(services.shared / "assets.json", {"seasons": {"2": [{"path": path}]}})
    with pytest.raises(PipelineError):
        cli.run_stage("render", services.shared, services.work, services.output, kind="season")
    assert not services.renders


def test_collector_cannot_export_an_outside_path_or_symlink(services, tmp_path, monkeypatch):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"private")
    module = sys.modules["recap_pipeline.assets"]
    monkeypatch.setattr(module, "collect_assets", lambda *args: {"seasons": {"2": [{"path": str(outside)}]}})
    cli.run_stage("research", services.shared, services.work, services.output)
    cli.run_stage("write", services.shared, services.work, services.output)
    with pytest.raises(PipelineError):
        cli.assets_stage(services.shared)
    link = services.shared / "external.jpg"
    link.symlink_to(outside)
    monkeypatch.setattr(module, "collect_assets", lambda *args: {"seasons": {"2": [{"path": str(link)}]}})
    with pytest.raises(PipelineError):
        cli.assets_stage(services.shared)


def test_writer_always_unloads_when_generation_or_scope_fails(services, monkeypatch):
    cli.research_stage(services.shared)
    module = sys.modules["recap_pipeline.writing"]
    monkeypatch.setattr(module, "write_recap", lambda *args, **kwargs: {})
    with pytest.raises(PipelineError):
        cli.write_stage(services.shared)
    assert services.events[-1] == ("unload",)
    assert not (services.shared / "assets.json").exists()


@pytest.mark.parametrize("field,content", [("spec.json", "[]"), ("research.json", "{"),
                                         ("season-script.json", "{}"), ("assets.json", "[]")])
def test_malformed_shared_artifacts_cannot_reach_renderer(services, field, content):
    cli.run_stage("prepare", services.shared, services.work, services.output)
    (services.shared / field).write_text(content)
    with pytest.raises(PipelineError):
        cli.render_stage("season", services.shared, services.work, services.output)
    assert not services.renders


def test_render_requires_frozen_spec_and_kind_before_any_work(services):
    with pytest.raises(PipelineError):
        cli.render_stage("season", services.shared, services.work, services.output)
    with pytest.raises(SystemExit):
        cli.main(["--stage", "render"])
    with pytest.raises(SystemExit):
        cli.main(["--stage", "prepare", "--kind", "season"])
    assert not services.events


def test_main_returns_nonzero_without_printing_secret_inputs_or_exception_urls(services, monkeypatch, capsys):
    secret = "bot-secret-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret)
    monkeypatch.setenv("RECAP_BRIEF", secret)

    def fail(*args, **kwargs):
        raise RuntimeError("https://service.invalid/" + secret)

    monkeypatch.setattr(cli, "run_stage", fail)
    assert cli.main(["--stage", "prepare", "--shared", str(services.shared)]) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "https://service.invalid" not in output.out + output.err


def test_main_completes_prepare_with_safe_stage_status(services, capsys):
    assert cli.main(["--stage", "prepare", "--shared", str(services.shared)]) == 0
    assert "Recap prepare complete." in capsys.readouterr().out



def test_main_explains_a_controlled_coverage_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise PipelineError("The latest season has no verified finale ending.")
    monkeypatch.setattr(cli, "run_stage", fail)
    assert cli.main(["--stage", "write"]) == 1
    assert "no verified finale ending" in capsys.readouterr().err



def test_rejected_script_keeps_diagnostic_draft_without_publishing_it(services, monkeypatch):
    cli.research_stage(services.shared)
    def reject(*args):
        raise PipelineError("A scope violation.")
    monkeypatch.setattr(cli, "validate_script_scope", reject)
    with pytest.raises(PipelineError, match="scope violation"):
        cli.write_stage(services.shared)
    assert (services.shared/"story-ledger.json").is_file()
    assert (services.shared/"season-draft.json").is_file()
    assert not (services.shared/"season-script.json").exists()
    assert not (services.shared/"series-script.json").exists()
    assert services.events[-1] == ("unload",)


def test_before_season_two_only_one_recap_is_written_and_rendered(services, monkeypatch):
    monkeypatch.setenv("UPCOMING_SEASON", "2")
    def season_one(kind):
        script = _script(kind)
        script["upcoming_season"], script["seasons"] = 2, [1]
        story = {**script["sections"][0], "id": "story-1", "season": 1, "episode_ids": ["e1"], "event_ids": ["v1"]}
        script["sections"] = [story, {**story, "id": "ending", "kind": "ending"}]
        script["source_ids"] = ["s1"]
        return script
    def bundle():
        value = _bundle()
        value["seasons"], value["sources"] = value["seasons"][:1], value["sources"][:1]
        return value
    research = sys.modules["recap_pipeline.research"].research_show
    sys.modules["recap_pipeline.research"].research_show = lambda spec, cache: (research(spec, cache), bundle())[1]
    write = sys.modules["recap_pipeline.writing"].write_recap
    sys.modules["recap_pipeline.writing"].write_recap = (
        lambda kind, *args, **kwargs: (write(kind, *args, **kwargs), season_one(kind))[1])
    assert set(cli.run_stage("prepare", services.shared, services.work, services.output)) == {"season"}
    assert [event[1] for event in services.events if event[0] == "write"] == ["season"]
    assert not (services.shared / "series-script.json").exists()
    cli.run_stage("render", services.shared, services.work, services.output, kind="season")
    with pytest.raises(PipelineError, match="only the season package"):
        cli.run_stage("render", services.shared, services.work, services.output, kind="series")
    assert [event[1] for event in services.events if event[0] == "render"] == ["season"]
