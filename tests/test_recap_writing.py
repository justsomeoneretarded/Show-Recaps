from copy import deepcopy
import json

import pytest

from movie_pipeline.core import PipelineError
from recap_pipeline.writing import (
    MAX_EPISODES_PER_BATCH, MAX_EVENT_CHARS, RecapWriter, build_ledger, write_recap,
)


def bundle_with_seasons(count=3, episodes=2):
    seasons, sources = [], []
    for season in range(1, count + 1):
        items = []
        for number in range(1, episodes + 1):
            episode_id, source_id = f"s{season}e{number}", f"source-{season}-{number}"
            items.append({"id": episode_id, "season": season, "number": number,
                          "title": f"Chapter {number}", "airdate": None, "image": None,
                          "summary": f"Mara opens gate {season}-{number}, allowing Ivo to escape. "
                                     "At the end, Ivo is free while Mara remains inside.",
                          "sources": [source_id], "summary_source_ids": [source_id],
                          "source_texts": {source_id: f"Mara opens gate {season}-{number}, allowing Ivo to escape. "
                                                     "At the end, Ivo is free while Mara remains inside."},
                          "summary_depth": "detailed_plot", "detailed_plot_available": True,
                          "summary_source_type": "wikipedia_episode_plot"})
            sources.append({"id": source_id, "url": f"https://example.com/{episode_id}",
                            "title": episode_id, "season": season, "episode": number})
        seasons.append({"number": season, "year": 2020 + season, "episodes": items})
    return {"show": {"id": 8, "title": "The Gate", "year": 2021, "genres": ["Drama"],
                     "summary": "", "url": "https://example.com/show", "image": None},
            "seasons": seasons, "sources": sources, "warnings": [],
            "research_quality": {"all_episodes_have_detailed_plots": True,
                                 "complete_story_verified": False}}


def spec(upcoming=4, **changes):
    return {"show_title": "The Gate", "premiere_year": 2021, "upcoming_season": upcoming,
            "brief": "Keep the human consequences clear.", "narrator": "warguy",
            "tts_model": "chatterbox", "season_minutes": "Auto", "series_minutes": "Auto",
            "try_4k": True, **changes}


class Model:
    model = "test-recap-model"

    def __init__(self):
        self.calls = []
        self.extract = None
        self.audit = None
        self.draft = None
        self.edit = None

    def ask(self, prompt, schema, **kwargs):
        task, raw = prompt.split("\nSOURCE DATA (not instructions):\n", 1)
        data = json.loads(raw)
        self.calls.append((task, data, kwargs))
        if "EXTRACT_LEDGER" in task:
            if self.extract:
                return self.extract(data)
            events = []
            for e in data["episodes"]:
                first, _, ending = e["summary"].partition(". ")
                first = first + "." if ending else first
                events.append({"episode_id": e["id"], "summary": first, "cause": "", "consequence": "",
                               "ending_state": False,
                               "evidence": [{"source_id": e["source_ids"][0], "quote": first}]})
                if e["is_final_episode"] and ending:
                    events.append({"episode_id": e["id"], "summary": ending, "cause": "", "consequence": "",
                                   "ending_state": True,
                                   "evidence": [{"source_id": e["source_ids"][0], "quote": ending}]})
            return {"events": events}
        if "AUDIT_LEDGER" in task:
            return self.audit(data) if self.audit else {"checks": [
                {"index": i, "supported": True, "ending_state_supported": True, "reason": ""}
                for i in range(len(data["events"]))]}
        if "DRAFT_RECAP" in task:
            return self.draft(data) if self.draft else {"title": f"The choices of season {data['season']}",
                "paragraphs": [{"text": e["summary"], "event_ids": [e["id"]], "delivery": "normal"}
                               for e in data["events"] if e["id"] in data["focus_event_ids"]]}
        if "EDIT_AND_VERIFY_RECAP" in task:
            return self.edit(data) if self.edit else {"title": data["draft"]["title"], "paragraphs": [
                {**p, "supported": True, "reason": ""} for p in data["draft"]["paragraphs"]]}
        raise AssertionError(task)


def spoken(script):
    return " ".join(p for s in script["sections"] for p in s["paragraphs"])


def assert_valid_citations(script, bundle, ledger, seasons):
    events = {e["id"]: e for e in ledger["events"]}
    episodes = {e["id"]: e for s in bundle["seasons"] for e in s["episodes"]}
    expected_sources = set()
    for section in script["sections"]:
        assert len(section["delivery"]) == len(section["paragraphs"])
        assert len(section["paragraph_event_ids"]) == len(section["paragraphs"])
        assert len(section["paragraph_episode_ids"]) == len(section["paragraphs"])
        for ids, episode_ids in zip(section["paragraph_event_ids"], section["paragraph_episode_ids"]):
            assert episode_ids == list(dict.fromkeys(events[i]["episode_id"] for i in ids))
        assert set(section["delivery"]) <= {"normal", "calm", "hype", "dramatic"}
        assert set(section["episode_ids"]) == {events[i]["episode_id"] for i in section["event_ids"]}
        for event_id in section["event_ids"]:
            event = events[event_id]
            assert event["season"] in seasons
            episode = episodes[event["episode_id"]]
            for evidence in event["evidence"]:
                assert evidence["source_id"] in episode["sources"]
                assert evidence["quote"] in episode["summary"]
                expected_sources.add(evidence["source_id"])
    assert set(script["source_ids"]) == expected_sources


def test_ledger_preserves_exact_evidence_and_owned_source_ids(tmp_path):
    bundle = bundle_with_seasons(1, 1)
    ledger = build_ledger(bundle, tmp_path, writer=Model())
    event = ledger["events"][0]
    episode = bundle["seasons"][0]["episodes"][0]
    assert event["id"].startswith("event_")
    assert event["episode_id"] == episode["id"]
    assert event["evidence"] == [{"source_id": episode["sources"][0],
                                  "quote": episode["summary"].split(". ", 1)[0] + "."}]
    assert ledger["coverage"][0]["status"] == "detailed_plot"


@pytest.mark.parametrize("bad_source,quote", [
    ("nonexistent", "Mara opens gate 1-1, allowing Ivo to escape."),
    ("source-1-2", "Mara opens gate 1-1, allowing Ivo to escape."),
    ("source-1-1", "Mara dies at the gate."),
    ("source-1-1", ""),
])
def test_ledger_rejects_fabricated_or_wrong_owner_evidence(tmp_path, bad_source, quote):
    model = Model()
    model.extract = lambda data: {"events": [{
        "episode_id": "s1e1", "summary": "Mara dies.", "cause": "", "consequence": "",
        "evidence": [{"source_id": bad_source, "quote": quote}]}]}
    ledger = build_ledger(bundle_with_seasons(1), tmp_path, writer=model)
    assert ledger["events"] == []
    assert any("no verified ledger events" in w for w in ledger["warnings"])
    assert not any("AUDIT_LEDGER" in task for task, _, _ in model.calls)


def test_matching_quote_does_not_authorize_unsupported_plot(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    model.extract = lambda data: {"events": [{
        "episode_id": "s1e1", "summary": "Mara dies.", "cause": "", "consequence": "",
        "evidence": [{"source_id": "source-1-1", "quote": data["episodes"][0]["summary"]}]}]}
    model.audit = lambda data: {"checks": [{"index": 0, "supported": False,
                                          "reason": "Opening a gate does not establish her death."}]}
    ledger = build_ledger(bundle, tmp_path, writer=model)
    assert ledger["events"] == []
    assert any("Unsupported or oversized ledger claims" in warning for warning in ledger["warnings"])


def test_season_and_series_have_distinct_enforced_scopes(tmp_path):
    bundle, model = bundle_with_seasons(4), Model()
    # Deliberately include incoming-season poison to test defense beyond the research contract.
    bundle["seasons"][3]["episodes"][0]["summary"] = "INCOMING SPOILER: Mara becomes queen."
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    season = write_recap("season", spec(), bundle, ledger, tmp_path / "scripts", writer=model)
    series = write_recap("series", spec(), bundle, ledger, tmp_path / "scripts", writer=model)
    assert season["seasons"] == [3]
    assert series["seasons"] == [1, 2, 3]
    assert {s["season"] for s in series["sections"] if s["kind"] == "story"} == {1, 2, 3}
    assert {s["season"] for s in season["sections"] if s["kind"] == "story"} == {3}
    for script, scope in [(season, {3}), (series, {1, 2, 3})]:
        assert "INCOMING SPOILER" not in spoken(script)
        assert {s["kind"] for s in script["sections"]} == {"intro", "story", "ending", "outro"}
        assert_valid_citations(script, bundle, ledger, scope)
    series_tasks = [task for task, data, _ in model.calls
                    if "DRAFT_RECAP" in task and data["kind"] == "series"]
    assert all("major linked arcs" in task for task in series_tasks)


def test_wrong_season_draft_and_edited_ids_fail_closed(tmp_path):
    bundle, model = bundle_with_seasons(3), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    wrong_id = next(e["id"] for e in ledger["events"] if e["season"] == 1)
    model.draft = lambda data: {"title": "A false transition", "paragraphs": [
        {"text": "Mara invents an unsupported future.", "event_ids": [wrong_id], "delivery": "normal"}]}
    with pytest.raises(PipelineError, match="No verified story narration"):
        write_recap("season", spec(), bundle, ledger, tmp_path / "draft", writer=model)
    model.draft = None
    model.edit = lambda data: {"title": "A false edit", "paragraphs": [
        {"text": "Mara invents another future.", "event_ids": [wrong_id],
         "delivery": "normal", "supported": True, "reason": ""}]}
    with pytest.raises(PipelineError, match="No verified story narration"):
        write_recap("season", spec(), bundle, ledger, tmp_path / "edit", writer=model)


def test_general_edit_repairs_pitch_and_rejects_unsupported_statement(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    def draft(data):
        event_id = data["events"][0]["id"]
        return {"title": "The turning point", "paragraphs": [
            {"text": "Number one! Stream on Netflix, a must-watch.", "event_ids": [event_id], "delivery": "hype"},
            {"text": "Mara dies.", "event_ids": [event_id], "delivery": "dramatic"}]}
    model.draft = draft
    def edit(data):
        if data["purpose"] == "ending":
            return {"title": "Ivo's position", "paragraphs": [
                {**p, "supported": True, "reason": ""} for p in data["draft"]["paragraphs"]]}
        event = data["events"][0]
        return {"title": "A way through the gate", "paragraphs": [
            {"text": "Mara opens the gate and gives Ivo a way out.", "event_ids": [event["id"]],
             "delivery": "normal", "supported": True, "reason": ""},
            {"text": "Mara dies.", "event_ids": [event["id"]],
             "delivery": "dramatic", "supported": False, "reason": "No death is established."}]}
    model.edit = edit
    script = write_recap("season", spec(2), bundle, ledger, tmp_path / "script", writer=model)
    assert "gives Ivo a way out" in spoken(script)
    assert all(term not in spoken(script).lower() for term in ["netflix", "number one", "must-watch", "mara dies"])
    assert any("EDIT_AND_VERIFY_RECAP" in task for task, _, _ in model.calls)
    assert_valid_citations(script, bundle, ledger, {1})


@pytest.mark.parametrize("unsafe", ["Stream on Netflix for season 2.", "Disney+ has the story.",
                                   "Paramount+ has the story.", "Apple TV+ has the story."])
def test_unsafe_editor_output_cannot_be_certified_by_supported_flag(tmp_path, unsafe):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    model.edit = lambda data: {"title": "Top 10 Netflix moments", "paragraphs": [
        {**data["draft"]["paragraphs"][0], "text": unsafe,
         "supported": True, "reason": ""}]}
    with pytest.raises(PipelineError, match="No verified story narration"):
        write_recap("season", spec(2), bundle, ledger, tmp_path / "script", writer=model)


def test_missing_finale_summary_fails_without_inventing_ending(tmp_path):
    bundle, model = bundle_with_seasons(1), Model()
    bundle["seasons"][0]["episodes"][-1]["summary"] = ""
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    with pytest.raises(PipelineError, match="No verified final-episode ending state"):
        write_recap("series", spec(2), bundle, ledger, tmp_path / "script", writer=model)


def test_second_season_request_keeps_both_outputs_useful(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    scripts = [write_recap(kind, spec(2), bundle, ledger, tmp_path / "script", writer=model)
               for kind in ("season", "series")]
    assert scripts[0]["title"] != scripts[1]["title"]
    assert "1–1" not in scripts[1]["title"]
    for script in scripts:
        assert script["seasons"] == [1]
        assert script["source_ids"]
        assert_valid_citations(script, bundle, ledger, {1})


def test_runtime_preference_is_soft_and_never_requires_padding(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    script = write_recap("season", spec(2, season_minutes="30"), bundle, ledger,
                        tmp_path / "script", writer=model)
    assert len(spoken(script).split()) < 100
    assert script["source_ids"]
    calls = [(task, data) for task, data, _ in model.calls if "DRAFT_RECAP" in task]
    assert calls and all(data["runtime_preference_minutes"] == "30" for _, data in calls)
    assert all("SOFT guidance" in task and "never pad" in task for task, _ in calls)
    assert sum(data["section_runtime_preference_minutes"] for _, data in calls) == 30


def test_ledger_uses_bounded_batches_and_reuses_unchanged_cache(tmp_path):
    bundle, model = bundle_with_seasons(1, 14), Model()
    first = build_ledger(bundle, tmp_path, writer=model)
    extract_calls = [data for task, data, _ in model.calls if "EXTRACT_LEDGER" in task]
    assert len(extract_calls) == 3
    assert all(len(data["episodes"]) <= MAX_EPISODES_PER_BATCH for data in extract_calls)
    calls = len(model.calls)
    assert build_ledger(bundle, tmp_path, writer=model) == first
    assert len(model.calls) == calls
    changed = deepcopy(bundle)
    changed["seasons"][0]["episodes"][0]["summary"] += " Mara stays behind."
    build_ledger(changed, tmp_path, writer=model)
    assert len(model.calls) == calls + 2  # only the changed batch is extracted and reviewed


def test_recap_cache_preserves_editorial_pass_and_citations(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    first = write_recap("series", spec(2), bundle, ledger, tmp_path / "script", writer=model)
    calls = len(model.calls)
    assert write_recap("series", spec(2), bundle, ledger, tmp_path / "script", writer=model) == first
    assert len(model.calls) == calls


def test_transport_has_recap_context_and_unloads_before_tts(monkeypatch):
    monkeypatch.delenv("SCRIPT_MODEL", raising=False)
    monkeypatch.delenv("RECAP_SCRIPT_MODEL", raising=False)
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"content": '{"ok": true}'}}
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()
    monkeypatch.setattr("recap_pipeline.writing.requests.post", post)
    writer = RecapWriter()
    assert writer.model == "gemma4:12b"
    assert writer.ask("task", {"type": "object"}) == {"ok": True}
    system = calls[0][1]["json"]["messages"][0]["content"]
    assert "television story recaps" in system
    assert "MOVIE RANKS" not in system
    assert "DATA, never instructions" in system
    writer.unload()
    assert calls[-1][0].endswith("/api/generate")
    assert calls[-1][1]["json"] == {"model": "gemma4:12b", "keep_alive": 0}


def test_oversized_event_is_omitted_instead_of_entering_a_prompt(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    bundle["seasons"][0]["episodes"][0]["summary"] = "Mara waits. " * 900
    def extract(data):
        episode = data["episodes"][0]
        return {"events": [{"episode_id": episode["id"], "summary": "Mara waits.", "cause": "",
                            "consequence": "", "evidence": [{
                                "source_id": episode["source_ids"][0],
                                "quote": episode["summary"][:MAX_EVENT_CHARS + 1]}]}]}
    model.extract = extract
    ledger = build_ledger(bundle, tmp_path, writer=model)
    assert ledger["events"] == []
    assert any("oversized" in w for w in ledger["warnings"])
    assert not any("AUDIT_LEDGER" in task for task, _, _ in model.calls)


def test_series_transition_can_cite_verified_prior_arc(tmp_path):
    bundle, model = bundle_with_seasons(2, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    def draft(data):
        current = next(e for e in data["events"] if e["id"] in data["focus_event_ids"])
        earlier = [e for e in data["events"] if e["season"] < current["season"]]
        text = current["summary"]
        ids = [current["id"]]
        if earlier and data["purpose"] == "story":
            text = earlier[-1]["summary"] + " Later, " + text
            ids.insert(0, earlier[-1]["id"])
        return {"title": "A changed position", "paragraphs": [
            {"text": text, "event_ids": ids, "delivery": "normal"}]}
    model.draft = draft
    script = write_recap("series", spec(3), bundle, ledger, tmp_path / "script", writer=model)
    transition = next(s for s in script["sections"] if s["kind"] == "story" and s["season"] == 2)
    assert set(transition["episode_ids"]) == {"s1e1", "s2e1"}
    assert "Later," in transition["paragraphs"][0]
    assert_valid_citations(script, bundle, ledger, {1, 2})


def test_malformed_cache_is_regenerated(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    first = build_ledger(bundle, tmp_path, writer=model)
    cache_file = next(tmp_path.glob("*.json"))
    cache_file.write_text('{"events": null}', encoding="utf-8")
    calls = len(model.calls)
    assert build_ledger(bundle, tmp_path, writer=model) == first
    assert len(model.calls) == calls + 2


def test_quote_is_bound_to_its_own_source_prose(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    episode = bundle["seasons"][0]["episodes"][0]
    episode["sources"].append("legacy-tvmaze")
    episode["summary_source_ids"].append("legacy-tvmaze")
    episode["source_texts"]["legacy-tvmaze"] = "A visitor arrives with a mysterious warning."
    bundle["sources"].append({"id": "legacy-tvmaze", "url": "https://example.com/legacy",
                              "season": 1, "episode": 1})
    model.extract = lambda data: {"events": [{
        "episode_id": episode["id"], "summary": "Mara opens the gate.", "cause": "", "consequence": "",
        "ending_state": False, "evidence": [{"source_id": "legacy-tvmaze",
                                           "quote": episode["summary"]}]}]}
    assert build_ledger(bundle, tmp_path, writer=model)["events"] == []


def test_overview_finale_cannot_produce_a_detailed_recap(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    episode = bundle["seasons"][0]["episodes"][0]
    episode.update(summary_depth="episode_overview", detailed_plot_available=False,
                   summary_source_type="tvmaze_overview")
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    with pytest.raises(PipelineError, match="needs a detailed sourced plot"):
        write_recap("season", spec(2), bundle, ledger, tmp_path / "script", writer=model)


def test_finale_temporary_event_does_not_establish_ending_state(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    model.audit = lambda data: {"checks": [
        {"index": i, "supported": True, "ending_state_supported": False,
         "reason": "A temporary situation is not established at the end."}
        for i in range(len(data["events"]))]}
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    assert ledger["events"] and not any(e["ending_state"] for e in ledger["events"])
    with pytest.raises(PipelineError, match="No verified final-episode ending state"):
        write_recap("series", spec(2), bundle, ledger, tmp_path / "script", writer=model)


def test_rejected_story_cannot_become_an_intro_outro_video(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    model.edit = lambda data: {"title": "Unsupported", "paragraphs": []}
    with pytest.raises(PipelineError, match="No verified story narration"):
        write_recap("season", spec(2), bundle, ledger, tmp_path / "script", writer=model)


def test_every_required_series_season_needs_narrated_story(tmp_path):
    bundle, model = bundle_with_seasons(2, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    ledger["events"] = [e for e in ledger["events"] if e["season"] == 2]
    with pytest.raises(PipelineError, match=r"required season\(s\): 1"):
        write_recap("series", spec(3), bundle, ledger, tmp_path / "script", writer=model)


def test_ending_facts_are_reserved_and_depth_reaches_prompts_and_report(tmp_path):
    bundle, model = bundle_with_seasons(2, 1), Model()
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    script = write_recap("series", spec(3), bundle, ledger, tmp_path / "script", writer=model)
    ending_ids = {i for section in script["sections"] if section["kind"] == "ending"
                  for i in section["event_ids"]}
    story_ids = {i for section in script["sections"] if section["kind"] == "story"
                 for i in section["event_ids"]}
    assert ending_ids and not ending_ids.intersection(story_ids)
    assert script["coverage"]["finale_state_verified"] is True
    assert script["research_quality"] == bundle["research_quality"] == ledger["research_quality"]
    assert all(value == "detailed_plot" for value in script["coverage"]["episode_depth"].values())
    drafts = [data for task, data, _ in model.calls if "DRAFT_RECAP" in task]
    assert all(data["research_quality"] == bundle["research_quality"] for data in drafts)
    assert all(event["summary_depth"] == "detailed_plot" for data in drafts for event in data["events"])


def test_runtime_total_is_shared_across_seasons_and_titles_fit_youtube(tmp_path):
    bundle, model = bundle_with_seasons(10, 1), Model()
    bundle["show"]["title"] = "A very long television title " * 10
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    script = write_recap("series", spec(11, series_minutes="15"), bundle, ledger,
                        tmp_path / "script", writer=model)
    drafts = [data for task, data, _ in model.calls if "DRAFT_RECAP" in task]
    shares = [data["section_runtime_preference_minutes"] for data in drafts]
    assert all(data["runtime_preference_minutes"] == "15" for data in drafts)
    assert sum(shares) == pytest.approx(15, abs=0.08)
    assert max(shares) < 3
    assert len(script["title"]) <= 100 and "Before Season 11" in script["title"]



def test_truncated_finale_cannot_certify_an_ending_from_an_incomplete_prompt(tmp_path):
    bundle, model = bundle_with_seasons(1, 1), Model()
    episode = bundle["seasons"][0]["episodes"][0]
    episode["summary"] = "Mara waits in the warehouse. " * 500 + "At the end, Ivo is free."
    episode["source_texts"][episode["sources"][0]] = episode["summary"]
    model.extract = lambda data: {"events": [
        {"episode_id": episode["id"], "summary": "Ivo is free at the end.",
         "cause": "", "consequence": "", "ending_state": True,
         "evidence": [{"source_id": episode["sources"][0], "quote": "At the end, Ivo is free."}]}]}
    ledger = build_ledger(bundle, tmp_path / "ledger", writer=model)
    assert ledger["events"] and not any(event["ending_state"] for event in ledger["events"])
    with pytest.raises(PipelineError, match="No verified final-episode ending state"):
        write_recap("season", spec(2), bundle, ledger, tmp_path / "script", writer=model)
