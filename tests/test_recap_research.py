from copy import deepcopy
from datetime import date
import json
from types import SimpleNamespace

import pytest
import requests

from movie_pipeline.core import PipelineError
from recap_pipeline import research


TODAY = date(2026, 10, 1)
PLOT = "Mara discovers the missing ledger and confronts her brother, who admits betraying the family."
FINAL_PLOT = "Mara exposes the conspiracy, frees the prisoners, and leaves her former ally behind after the final confrontation."


def spec(**changes):
    values = dict(show_title="The Bridge", premiere_year=2011, upcoming_season=3)
    values.update(changes)
    return SimpleNamespace(**values)


def show(identifier=10, name="The Bridge", year=2011, imdb="tt1733785"):
    return {"id": identifier, "name": name, "premiered": f"{year}-09-21",
            "genres": ["Drama", "Crime"], "summary": "INCOMING SEASON SPOILER",
            "url": f"https://www.tvmaze.com/shows/{identifier}/the-bridge",
            "image": {"original": "https://static.tvmaze.com/current-season-poster.jpg"},
            "externals": {"imdb": imdb, "thetvdb": identifier + 100}}


def episode(identifier, season, number, *, summary=PLOT, airdate=None, **changes):
    result = {"id": identifier, "season": season, "number": number,
              "name": f"Chapter {season}.{number}",
              "airdate": airdate or f"{2010 + season}-10-{number:02}",
              "summary": summary, "type": "regular",
              "url": f"https://www.tvmaze.com/episodes/{identifier}",
              "image": {"original": f"https://static.tvmaze.com/episode-{identifier}.jpg"}}
    result.update(changes)
    return result


class Response:
    def __init__(self, data=None, status=200):
        self.data, self.status_code = data, status
        self.content = json.dumps(data).encode()
        self.headers = {}

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return deepcopy(self.data)


class Session:
    """Explicit URL routes make unexpected network work fail the test."""
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url not in self.routes:
            raise AssertionError(f"Unexpected HTTP request: {url}")
        value = self.routes[url]
        if callable(value):
            return value(url, kwargs)
        if isinstance(value, list):
            if url == research._WIKIPEDIA:
                params = kwargs.get("params", {})
                matches = [index for index, response in enumerate(value)
                           if (params.get("action") == "query" and isinstance(response.data, dict)
                               and "query" in response.data)
                           or (params.get("action") == "parse" and isinstance(response.data, dict)
                               and isinstance(response.data.get("parse"), dict)
                               and response.data["parse"].get("title") == params.get("page")
                               and params.get("prop") in response.data["parse"])]
                if not matches:
                    return Response({"query": {"search": []}} if params.get("action") == "query" else
                                    {"parse": {"title": params.get("page"), "sections": []}})
                value = value.pop(matches[0])
            else:
                value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


def session_for(episodes=None, candidates=None, extra=None):
    episodes = episodes if episodes is not None else [episode(101, 1, 1), episode(201, 2, 1)]
    routes = {
        research._TVMAZE + "/search/shows": Response([{"show": item} for item in (candidates or [show()])]),
        research._TVMAZE + "/shows/10/episodes": Response(episodes),
        research._WIKIDATA: Response({"query": {"search": []}}),
    }
    routes.update(extra or {})
    return Session(routes)


@pytest.fixture(autouse=True)
def offline_defaults(monkeypatch):
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    monkeypatch.setattr(research.time, "sleep", lambda _: None)
    monkeypatch.setattr(requests.sessions.Session, "request",
                        lambda *args, **kwargs: pytest.fail("Real network access is forbidden in research tests"))


def test_title_and_premiere_year_override_search_rank_and_remakes(tmp_path):
    wrong = show(90, year=2018)
    fuzzy = show(91, name="The Bridge: After Dark")
    right = show()
    client = session_for(candidates=[wrong, fuzzy, right])
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["show"]["id"] == 10
    assert result["show"]["year"] == 2011
    assert result["show"]["ambiguous_title"] is True
    assert result["show"]["summary"] == ""
    assert result["show"]["image"].endswith("episode-201.jpg")
    assert client.calls[0][1]["params"] == {"q": "The Bridge"}
    assert all(call[1]["timeout"] == (5, 25) and call[1]["allow_redirects"] is False for call in client.calls)
    assert not client.closed  # An injected session remains the caller's property.


def test_normalized_accents_and_punctuation_are_exact_title_matches(tmp_path):
    client = session_for(candidates=[show(name="Thé — Bridge")])
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["show"]["title"] == "Thé — Bridge"
    assert result["show"]["ambiguous_title"] is False


@pytest.mark.parametrize("candidate", [show(year=2012), show(name="Another Bridge")])
def test_incorrect_identity_never_substitutes_a_similar_show(tmp_path, candidate):
    client = session_for(candidates=[candidate])
    with pytest.raises(PipelineError, match="No exact TVMaze match"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert len(client.calls) == 1


def test_same_title_same_year_requires_explicit_disambiguation(tmp_path):
    client = session_for(candidates=[show(), show(11)])
    with pytest.raises(PipelineError, match="Multiple TVMaze"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)
    client = session_for(candidates=[show(11, imdb="tt9999999"), show()])
    result = research.research_show(spec(imdb_id="tt1733785"), tmp_path, session=client, today=TODAY)
    assert result["show"]["id"] == 10


def test_explicit_tvmaze_id_still_verifies_title_year_and_external_id(tmp_path):
    client = session_for(extra={research._TVMAZE + "/shows/99": Response(show(99, year=2018))})
    with pytest.raises(PipelineError, match="No exact TVMaze"):
        research.research_show(spec(tvmaze_id=99), tmp_path, session=client, today=TODAY)
    assert len(client.calls) == 1


def test_scope_filters_incoming_specials_and_older_season_unconfirmed_airdates(tmp_path):
    episodes = [
        episode(301, 3, 1, summary="FORBIDDEN incoming season plot"),
        episode(101, 1, 1),
        episode(999, 0, 1, summary="FORBIDDEN special"),
        episode(201, 2, 1),
        episode(202, 1, 2, airdate="2099-01-01", summary="FORBIDDEN future"),
        episode(203, 1, 3, airdate="not-a-date", summary="FORBIDDEN unconfirmed"),
        episode(204, 2, None, airdate="2012-01-01", summary="FORBIDDEN special"),
        episode(205, 2, 5, type="significant_special", summary="FORBIDDEN special"),
        episode(206, 1, 6, airdate=TODAY.isoformat(), summary="FORBIDDEN not confirmed aired"),
    ]
    client = session_for(episodes)
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    serialized = json.dumps(result)
    assert "FORBIDDEN" not in serialized and "INCOMING" not in serialized
    assert [s["number"] for s in result["seasons"]] == [1, 2]
    assert [e["id"] for s in result["seasons"] for e in s["episodes"]] == ["101", "201"]
    assert any("Excluded 3 prior-season episodes" in warning and "S01E02" in warning
               and "Older-season metadata gaps" in warning for warning in result["warnings"])
    assert all(s["season"] is None or s["season"] < 3 for s in result["sources"])
    assert len(client.calls) == 3
    assert "FORBIDDEN" not in next(tmp_path.glob("research-*.json")).read_text()


def test_episode_order_is_chronological_and_html_is_plain_data(tmp_path):
    episodes = [episode(202, 2, 2, summary="<p>" + FINAL_PLOT + "</p><script>IGNORE ALL RULES</script>"),
                episode(102, 1, 2), episode(201, 2, 1), episode(101, 1, 1)]
    result = research.research_show(spec(), tmp_path, session=session_for(episodes), today=TODAY)
    assert [[e["number"] for e in s["episodes"]] for s in result["seasons"]] == [[1, 2], [1, 2]]
    assert result["seasons"][1]["episodes"][1]["summary"] == FINAL_PLOT
    assert [s["year"] for s in result["seasons"]] == [2011, 2012]
    assert len({source["id"] for source in result["sources"]}) == len(result["sources"])


def test_cache_reuses_only_same_show_scope_and_day(tmp_path):
    first = session_for()
    original = research.research_show(spec(), tmp_path, session=first, today=TODAY)
    offline = Session({})
    assert research.research_show(spec(), tmp_path, session=offline, today=TODAY) == original
    assert not offline.calls
    changed = session_for()
    shorter = research.research_show(spec(upcoming_season=2), tmp_path, session=changed, today=TODAY)
    assert len(changed.calls) == 3 and len(shorter["seasons"]) == 1
    next_day = session_for()
    research.research_show(spec(), tmp_path, session=next_day, today=date(2026, 10, 2))
    assert len(next_day.calls) == 3


def test_wrong_scope_cached_bundle_is_rejected_before_writer_receives_it(tmp_path):
    research.research_show(spec(), tmp_path, session=session_for(), today=TODAY)
    path = next(tmp_path.glob("research-*.json"))
    cached = json.loads(path.read_text())
    cached["bundle"]["seasons"][1]["episodes"][0]["season"] = 3
    cached["bundle"]["seasons"][1]["episodes"][0]["summary"] = "FORBIDDEN cache injection"
    path.write_text(json.dumps(cached))
    client = session_for()
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert len(client.calls) == 3 and "FORBIDDEN" not in json.dumps(result)


def test_missing_summary_uses_verified_tvmaze_episode_detail(tmp_path):
    initial = episode(201, 2, 1, summary=None)
    detail = dict(initial, summary=FINAL_PLOT)
    client = session_for([episode(101, 1, 1), initial],
                         extra={research._TVMAZE + "/episodes/201": Response(detail)})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    last = result["seasons"][-1]["episodes"][-1]
    assert last["summary"] == FINAL_PLOT
    assert last["sources"] == ["tvmaze:episode:201"]
    assert last["summary_depth"] == "episode_overview"
    assert last["detailed_plot_available"] is False
    assert result["research_quality"]["can_support_complete_recap"] is False
    assert any("may omit twists and endings" in warning for warning in result["warnings"])


def tmdb_routes(missing, overview=FINAL_PLOT, *, title=None, airdate=None, source_show=None):
    return {
        research._TVMAZE + f"/episodes/{missing['id']}": Response(missing),
        research._TMDB + "/find/tt1733785": Response({"tv_results": [source_show or
            {"id": 77, "name": "The Bridge", "first_air_date": "2011-09-21"}]}),
        research._TMDB + "/tv/77/season/2": Response({"season_number": 2, "episodes": [
            {"episode_number": 1, "season_number": 2, "name": title or missing["name"],
             "air_date": airdate or missing["airdate"], "overview": overview, "still_path": "/prior.jpg"},
            {"episode_number": 2, "season_number": 3, "name": "Incoming", "overview": "FORBIDDEN TMDB"}
        ]}),
    }


def test_verified_tmdb_season_fallback_has_provenance_and_no_incoming_content(tmp_path, monkeypatch):
    monkeypatch.setenv("TMDB_READ_TOKEN", "private-token-never-in-cache")
    missing = episode(201, 2, 1, summary=None, image=None)
    client = session_for([episode(101, 1, 1), missing], extra=tmdb_routes(missing))
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    last = result["seasons"][-1]["episodes"][-1]
    assert last["summary"] == FINAL_PLOT
    assert last["sources"] == ["tvmaze:episode:201", "tmdb:tv:77:season:2"]
    assert last["image"] == "https://image.tmdb.org/t/p/original/prior.jpg"
    assert any(source["url"] == "https://www.themoviedb.org/tv/77/season/2" for source in result["sources"])
    assert "FORBIDDEN" not in json.dumps(result)
    assert "private-token" not in next(tmp_path.glob("research-*.json")).read_text()
    tmdb_calls = [kwargs for url, kwargs in client.calls if url.startswith(research._TMDB)]
    assert all(kwargs["headers"]["Authorization"] == "Bearer private-token-never-in-cache" for kwargs in tmdb_calls)


@pytest.mark.parametrize("changes", [
    {"title": "A Different Episode"},
    {"airdate": "2012-10-02"},
    {"source_show": {"id": 77, "name": "The Bridge", "first_air_date": "2018-01-01"}},
])
def test_tmdb_never_attaches_a_different_show_or_episode_plot(tmp_path, monkeypatch, changes):
    monkeypatch.setenv("TMDB_READ_TOKEN", "test-token")
    missing = episode(201, 2, 1, summary=None)
    client = session_for([episode(101, 1, 1), missing], extra=tmdb_routes(missing, **changes))
    with pytest.raises(PipelineError, match="no useful sourced episode summaries"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)


def test_partial_summary_gaps_are_explicit_and_never_invented(tmp_path):
    missing = episode(101, 1, 1, summary=None)
    client = session_for([missing, episode(102, 1, 2), episode(201, 2, 1)],
                         extra={research._TVMAZE + "/episodes/101": Response(missing)})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["seasons"][0]["episodes"][0]["summary"] == ""
    assert any("Season 1 summary coverage: 1/2" in warning and "E01" in warning for warning in result["warnings"])
    assert any("TMDB_READ_TOKEN" in warning for warning in result["warnings"])


def test_latest_prior_season_finale_without_facts_stops_recap(tmp_path):
    missing = episode(202, 2, 2, summary=None)
    client = session_for([episode(101, 1, 1), episode(201, 2, 1), missing],
                         extra={research._TVMAZE + "/episodes/202": Response(missing)})
    with pytest.raises(PipelineError, match="finale S02E02"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)


def test_long_show_retains_every_episode_and_finale_without_extra_requests(tmp_path):
    episodes = [episode(season * 1000 + number, season, number,
                        airdate=f"{2009 + season}-01-01")
                for season in range(1, 11) for number in range(1, 26)]
    client = session_for(list(reversed(episodes)))
    result = research.research_show(spec(upcoming_season=11), tmp_path, session=client, today=TODAY)
    assert sum(len(season["episodes"]) for season in result["seasons"]) == 250
    assert result["seasons"][-1]["episodes"][-1]["id"] == "10025"
    assert len(client.calls) == 3


def test_detail_budget_prioritises_finales_and_marks_all_remaining_gaps(tmp_path):
    episodes = [episode(number, 1, number, summary=None if number <= 44 or number == 46 else PLOT,
                        airdate="2011-01-01") for number in range(1, 47)]
    routes = {research._TVMAZE + f"/episodes/{item['id']}": Response(dict(item, summary=FINAL_PLOT)
                                                                if item["number"] == 46 else item)
              for item in episodes if not item["summary"]}
    client = session_for(episodes, extra=routes)
    result = research.research_show(spec(upcoming_season=2), tmp_path, session=client, today=TODAY)
    assert len(result["seasons"][0]["episodes"]) == 46
    assert len(client.calls) == 3 + research._DETAIL_LIMIT
    assert result["seasons"][0]["episodes"][-1]["id"] == "46"
    assert any("capped at 36" in warning for warning in result["warnings"])
    assert client.calls[3][0].endswith("/episodes/46")  # Finale gets the first detail request.
    assert any("2/46" in warning and "E44" in warning for warning in result["warnings"])


def test_missing_required_prior_season_is_actionable(tmp_path):
    client = session_for([episode(101, 1, 1), episode(301, 3, 1)])
    with pytest.raises(PipelineError, match=r"prior season\(s\): 2"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)


def test_source_retries_are_bounded_and_do_not_echo_secrets(tmp_path):
    url = research._TVMAZE + "/search/shows"
    client = Session({url: [requests.Timeout("private-token"), Response(status=503), Response(status=503)]})
    with pytest.raises(PipelineError, match="3 attempts") as caught:
        research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert len(client.calls) == 3
    assert "private-token" not in str(caught.value)
    assert not list(tmp_path.glob("*.json"))


def test_success_after_transient_rate_limit_and_owned_session_is_closed(tmp_path, monkeypatch):
    client = session_for()
    url = research._TVMAZE + "/search/shows"
    client.routes[url] = [Response(status=429), Response([{"show": show()}])]
    monkeypatch.setattr(research.requests, "Session", lambda: client)
    result = research.research_show(spec(), tmp_path, today=TODAY)
    assert result["show"]["id"] == 10 and len(client.calls) == 4
    assert client.closed


@pytest.mark.parametrize("changes", [{"upcoming_season": 1}, {"premiere_year": True},
                                      {"tvmaze_id": "10"}, {"tvmaze_id": 0}, {"tvdb_id": False},
                                      {"tmdb_id": 0}, {"imdb_id": ""}, {"imdb_id": "../../bad"}])
def test_invalid_scope_or_identity_is_rejected_before_network(tmp_path, changes):
    client = Session({})
    with pytest.raises(PipelineError):
        research.research_show(spec(**changes), tmp_path, session=client, today=TODAY)
    assert not client.calls


DETAILED_PLOT = (
    "Mara follows the coded ledger to the harbor and discovers that her former partner has been "
    "moving prisoners through the customs warehouse. She persuades her brother to return the stolen "
    "evidence, but he refuses until she reveals that their father was one of the prisoners. Together "
    "they break into the warehouse and free the captives. During the confrontation their partner "
    "destroys the escape boat, leaving Mara to lead the survivors through an abandoned tunnel. "
    "The police arrive with the recovered ledger, arrest the conspirators, and reunite the family."
)


def wikidata_routes(*, correct_imdb=True):
    return [Response({"query": {"search": [{"title": "Q123"}]}}),
            Response({"entities": {"Q123": {
                "claims": {"P345": [{"mainsnak": {"datavalue": {
                    "value": "tt1733785" if correct_imdb else "tt9999999"}}}]},
                "sitelinks": {"enwiki": {"title": "The Bridge (2011 TV series)"}}
            }}})]


def season_html(season, *, text=DETAILED_PLOT, parent="The_Bridge_(2011_TV_series)",
                row_number=1, episode_title=None, link=True):
    episode_title = episode_title or f"Chapter {season}.1"
    page = episode_title.replace(" ", "_") + "_(The_Bridge)"
    title = f'<a href="/wiki/{page}">"{episode_title}"</a>' if link else f'"{episode_title}"'
    return (f'<p>A season of <a href="/wiki/{parent}">The Bridge</a>.</p><h2>Episodes</h2>'
            f'<table><tr class="vevent"><th>{season * 10}</th><td>{row_number}</td>'
            f'<td class="summary">{title}</td></tr>'
            f'<tr><td class="description">{text}</td></tr></table>')


def wiki_search(season):
    return Response({"query": {"search": [{"title": f"The Bridge (season {season})"}]}})


def wiki_parsed(title, text):
    return Response({"parse": {"title": title, "text": text}})


def test_wikipedia_verified_prior_season_tables_and_plot_only_finale_sections(tmp_path):
    responses = []
    for season in (2, 1):
        responses += [wiki_search(season), wiki_parsed(f"The Bridge (season {season})", season_html(season))]
    for season in (2, 1):
        title = f"Chapter {season}.1 (The Bridge)"
        responses += [Response({"parse": {"title": title, "sections": [
            {"line": "Production", "index": "1"}, {"line": "Plot", "index": "2"}]}}),
            wiki_parsed(title, "<h2>Plot</h2><p>" + DETAILED_PLOT + "</p>")]
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    episodes = [item for season in result["seasons"] for item in season["episodes"]]
    assert all(item["summary"] == DETAILED_PLOT for item in episodes)
    assert all(item["summary_depth"] == "detailed_plot" and item["detailed_plot_available"] for item in episodes)
    assert all(any(source.startswith("wikipedia:episode:") for source in item["sources"]) for item in episodes)
    quality = result["research_quality"]
    assert quality["episodes_with_detailed_plot"] == 2 and quality["detailed_finale_gaps"] == []
    assert quality["all_episodes_have_detailed_plots"] is True
    assert quality["complete_story_verified"] is False
    wiki_calls = [kwargs["params"] for url, kwargs in client.calls if url == research._WIKIPEDIA]
    bodies = [call for call in wiki_calls if call.get("prop") == "text"]
    assert all(call["page"] != "The Bridge (2011 TV series)" for call in bodies)
    assert not any("season 3" in call.get("page", "") for call in wiki_calls)
    episode_bodies = [call for call in bodies if call["page"].startswith("Chapter")]
    assert len(episode_bodies) == 2 and all(call["section"] == "2" for call in episode_bodies)
    cached = research.research_show(spec(), tmp_path, session=Session({}), today=TODAY)
    assert cached["research_quality"] == quality


@pytest.mark.parametrize("malformation", ["wrong_parent", "mixed_parent", "wrong_episode_number", "wrong_episode_title",
                                          "wrong_imdb_claim", "incoming_search_result"])
def test_wikipedia_identity_and_episode_scope_are_verified_before_using_plot(tmp_path, malformation):
    responses = []
    for season in (2, 1):
        if malformation == "incoming_search_result":
            responses.append(Response({"query": {"search": [{"title": "The Bridge (season 3)"}]}}))
            continue
        changes = {}
        if malformation in {"wrong_parent", "mixed_parent"}:
            changes["parent"] = "The_Bridge_(2018_TV_series)"
        elif malformation == "wrong_episode_number":
            changes["row_number"] = 2
        elif malformation == "wrong_episode_title":
            changes["episode_title"] = "Another Episode"
        body = season_html(season, **changes)
        if malformation == "mixed_parent":
            body = body.replace("</p>", '<span>Adapted from <a href="/wiki/The_Bridge_(2011_TV_series)">The Bridge</a>.</span></p>', 1)
        responses += [wiki_search(season), wiki_parsed(f"The Bridge (season {season})", body)]
    client = session_for(extra={research._WIKIDATA: wikidata_routes(
        correct_imdb=malformation != "wrong_imdb_claim"), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert all(item["summary"] == PLOT for season in result["seasons"] for item in season["episodes"])
    assert result["research_quality"]["episodes_with_detailed_plot"] == 0
    assert result["research_quality"]["detailed_finale_gaps"] == ["S01E01", "S02E01"]
    assert not any(source["id"].startswith("wikipedia:") for source in result["sources"])
    if malformation == "incoming_search_result":
        assert not any(kwargs["params"].get("prop") == "text"
                       for url, kwargs in client.calls if url == research._WIKIPEDIA)


def test_short_wikipedia_blurb_is_not_misrepresented_as_detailed_plot(tmp_path):
    responses = []
    for season in (2, 1):
        responses += [wiki_search(season), wiki_parsed(f"The Bridge (season {season})",
                                                       season_html(season, text=PLOT, link=False))]
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["research_quality"]["episodes_with_detailed_plot"] == 0
    assert result["research_quality"]["can_support_complete_recap"] is False
    assert any("may omit twists and endings" in warning for warning in result["warnings"])


def test_duplicate_episode_ids_are_rejected_before_overwriting_facts(tmp_path):
    client = session_for([episode(101, 1, 1), episode(101, 2, 1)])
    with pytest.raises(PipelineError, match="repeats an episode ID"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)


def test_wikipedia_accepts_verified_british_series_numbering(tmp_path):
    responses = []
    for season in (2, 1):
        title = f"The Bridge (series {season})"
        responses += [Response({"query": {"search": [{"title": title}]}}),
                      wiki_parsed(title, season_html(season, link=False))]
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["research_quality"]["episodes_with_detailed_plot"] == 2
    assert all("series" in source["title"] for source in result["sources"]
               if source["id"].startswith("wikipedia:season:"))


def test_detailed_wikipedia_plots_are_attempted_before_optional_overview_lookups(tmp_path):
    missing = episode(201, 2, 1, summary=None)
    responses = []
    for season in (2, 1):
        responses += [wiki_search(season), wiki_parsed(f"The Bridge (season {season})",
                                                       season_html(season, link=False))]
    # No /episodes/201 route: detailed Wikipedia should fill it first.
    client = session_for([episode(101, 1, 1), missing],
                         extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["seasons"][1]["episodes"][0]["summary"] == DETAILED_PLOT
    assert result["seasons"][1]["episodes"][0]["detailed_plot_available"] is True
    assert not any("/episodes/201" in url for url, _ in client.calls)


def test_independent_wikipedia_title_searches_support_series_fallback(tmp_path):
    def route(url, kwargs):
        params = kwargs["params"]
        assert " OR " not in params.get("srsearch", "")
        if params.get("action") == "query":
            query = params["srsearch"]
            season = 2 if '"series 2"' in query else 1
            if 'intitle:"series ' in query:
                return wiki_search(season)
            return Response({"query": {"search": []}})
        return wiki_parsed(params["page"], season_html(2 if "season 2" in params["page"] else 1,
                                                       link=False))
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: route})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["research_quality"]["episodes_with_detailed_plot"] == 2
    searches = [kw["params"]["srsearch"] for url, kw in client.calls
                if url == research._WIKIPEDIA and kw["params"].get("action") == "query"]
    assert len(searches) == 4


def test_main_series_fallback_reads_only_exact_prior_season_sections(tmp_path):
    series = "The Bridge (2011 TV series)"
    def route(url, kwargs):
        params = kwargs["params"]
        if params.get("action") == "query":
            return Response({"query": {"search": []}})
        assert params["page"] == series
        if params["prop"] == "sections":
            return Response({"parse": {"title": series, "sections": [
                {"line": "Plot", "index": "1", "toclevel": 1},
                {"line": "Episodes", "index": "2", "toclevel": 1},
                {"line": "Season 1 (2011)", "index": "3", "toclevel": 2},
                {"line": "Season 2 (2012)", "index": "4", "toclevel": 2},
                {"line": "Season 3 (2026)", "index": "5", "toclevel": 2},
                {"line": "Reception", "index": "6", "toclevel": 1}]}})
        assert params["section"] in {"3", "4"}
        season = 1 if params["section"] == "3" else 2
        return wiki_parsed(series, season_html(season, link=False))
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: route})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert result["research_quality"]["episodes_with_detailed_plot"] == 2
    bodies = [kw["params"] for url, kw in client.calls
              if url == research._WIKIPEDIA and kw["params"].get("prop") == "text"]
    assert {body["section"] for body in bodies} == {"3", "4"}
    assert all(body["page"] == series for body in bodies)


def test_plot_source_provenance_does_not_assign_wikipedia_prose_to_tvmaze(tmp_path):
    responses = []
    for season in (2, 1):
        responses += [wiki_search(season), wiki_parsed(f"The Bridge (season {season})",
                                                       season_html(season, link=False))]
    client = session_for(extra={research._WIKIDATA: wikidata_routes(), research._WIKIPEDIA: responses})
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    last = result["seasons"][-1]["episodes"][-1]
    wiki_source = "wikipedia:season:10:2"
    assert last["summary_source_ids"] == [wiki_source]
    assert last["source_texts"][wiki_source] == DETAILED_PLOT
    assert last["source_texts"]["tvmaze:episode:201"] == PLOT
    assert DETAILED_PLOT not in last["source_texts"]["tvmaze:episode:201"]


def test_long_promotional_questions_are_not_detailed_plots_and_dialogue_is_allowed():
    teaser = ("Mara and Ivo face their most dangerous challenge yet as an unexpected visitor arrives "
              "with news that could change the fate of their entire family and everyone they have "
              "ever trusted across the city and the distant islands beyond the harbor. "
              "Will they discover the truth before it is too late?")
    assert len(teaser.split()) > 45
    assert not research._plot_depth(teaser, "wikipedia_episode_table")
    assert not research._plot_depth(DETAILED_PLOT, "tvmaze_overview")
    assert research._plot_depth(DETAILED_PLOT + ' Mara asks, "Are they safe?"', "wikipedia_episode_table")



@pytest.mark.parametrize("changes", [
    {"airdate": "2099-01-01"},
    {"airdate": "not-a-date"},
    {"airdate": TODAY.isoformat()},
    {"airdate": "2012-10-02", "airstamp": "unconfirmed"},
])
def test_unfinished_latest_required_season_cannot_use_last_aired_episode_as_finale(tmp_path, changes):
    client = session_for([episode(101, 1, 1), episode(201, 2, 1),
                          episode(202, 2, 2, summary="FORBIDDEN unaired finale", **changes)])
    with pytest.raises(PipelineError, match="Latest required prior season 2 is not confirmed complete"):
        research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert not list(tmp_path.glob("research-*.json"))
    assert len(client.calls) == 2


def test_current_fully_aired_prior_season_works_and_unaired_specials_do_not_block(tmp_path):
    client = session_for([
        episode(101, 1, 1),
        episode(201, 2, 1, airdate="2026-09-30", airstamp="2026-09-30T20:00:00+00:00"),
        episode(202, 2, 2, airdate=TODAY.isoformat(), airstamp="2026-10-01T20:00:00+00:00"),
        episode(299, 2, 3, airdate="2099-01-01", type="significant_special",
                summary="FORBIDDEN unaired special"),
    ])
    result = research.research_show(spec(), tmp_path, session=client, today=TODAY)
    assert [episode["id"] for episode in result["seasons"][-1]["episodes"]] == ["201", "202"]
    assert "FORBIDDEN" not in json.dumps(result)
