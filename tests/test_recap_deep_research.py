import json

from recap_pipeline import deep_research
from recap_pipeline.gemini import NotGrounded
from recap_pipeline.writing import _valid_event

SUMMARY = ("Reacher saves Richard Beck from a kidnapping. Zachary Beck hires him. "
           "Duffy recruited Reacher to find Teresa Daniels.")
PLOT = ("Reacher stages the rescue of Richard Beck with Duffy's DEA team, who pose as the kidnappers. "
        "Zachary Beck welcomes him into the family home and tests him with Russian roulette. "
        + "Reacher searches the house at night and finds that Teresa Daniels was held in the cellar. " * 12)


def bundle():
    return {"show": {"id": 7, "title": "Reacher", "year": 2022},
            "seasons": [{"number": 3, "episodes": [
                {"id": "31", "season": 3, "number": 1, "title": "Persuader", "airdate": "2025-02-20",
                 "summary": SUMMARY, "sources": ["tvmaze:episode:31"], "summary_source_ids": ["tvmaze:episode:31"],
                 "source_texts": {"tvmaze:episode:31": SUMMARY}, "summary_depth": "episode_overview",
                 "detailed_plot_available": False},
                {"id": "32", "season": 3, "number": 2, "title": "Truth Or Dare", "airdate": "2025-02-20",
                 "summary": "Reacher meets Paulie. Quinn visits Zachary Beck.", "sources": [],
                 "summary_source_ids": [], "source_texts": {}, "summary_depth": "episode_overview",
                 "detailed_plot_available": False}]}],
            "sources": [{"id": "tvmaze:episode:31", "url": "https://www.tvmaze.com/episodes/31",
                         "title": "S03E01", "season": 3, "episode": 1}],
            "warnings": []}


class Client:
    model = "gemini-test"

    def __init__(self, answer, sources=None, error=None):
        self.answer, self.sources, self.error, self.calls = answer, sources, error, 0

    def generate_grounded(self, system, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer, self.sources or [{"uri": "https://vertexaisearch.cloud.google.com/x",
                                              "title": "reacher.fandom.com"}]


def test_grounded_plot_becomes_a_cited_detailed_source(tmp_path):
    answer = "```json\n" + json.dumps({"episodes": [{"number": 1, "plot": PLOT},
                                                     {"number": 2, "plot": "Too short."}]}) + "\n```"
    result = deep_research.deepen(bundle(), tmp_path, Client(answer))
    first, second = result["seasons"][0]["episodes"]
    assert first["detailed_plot_available"] is True and first["summary_depth"] == "detailed_plot"
    assert first["summary"].startswith(SUMMARY) and " ".join(PLOT.split()) in first["summary"]
    source = next(s for s in result["sources"] if s["id"] == "web:7:3:1")
    assert source["url"] == "https://reacher.fandom.com" and source["episode"] == 1
    assert second["detailed_plot_available"] is False  # a short answer adds nothing
    # A story event may now quote the researched plot, and only verbatim.
    event = {"episode_id": "31", "summary": "Reacher finds Teresa was held in the cellar.",
             "evidence": [{"source_id": "web:7:3:1", "quote": "finds that Teresa Daniels was held in the cellar"}]}
    sources = {s["id"]: s for s in result["sources"]}
    assert _valid_event(event, {"31": first}, sources)
    event["evidence"][0]["quote"] = "Teresa escapes on her own"
    assert _valid_event(event, {"31": first}, sources) is None


def test_plot_about_other_characters_is_rejected(tmp_path):
    wrong = ("Jack Bauer races across Los Angeles to stop a bomb while Chloe tracks the signal. " * 12)
    result = deep_research.deepen(bundle(), tmp_path, Client(json.dumps({"episodes": [{"number": 1, "plot": wrong}]})))
    assert result["seasons"][0]["episodes"][0]["summary"] == SUMMARY
    assert not any(s["id"].startswith("web:") for s in result["sources"])


def test_ungrounded_or_failed_research_leaves_the_bundle_unchanged(tmp_path):
    for error in (NotGrounded("no sources"), deep_research.GeminiUnavailable("quota")):
        result = deep_research.deepen(bundle(), tmp_path, Client("", error=error))
        assert result["seasons"][0]["episodes"][0]["summary"] == SUMMARY


def test_research_is_cached(tmp_path):
    answer = json.dumps({"episodes": [{"number": 1, "plot": PLOT}]})
    client = Client(answer)
    deep_research.deepen(bundle(), tmp_path, client)
    deep_research.deepen(bundle(), tmp_path, client)
    assert client.calls == 1


def test_no_key_means_no_change(tmp_path):
    assert deep_research.deepen(bundle(), tmp_path) == bundle()


class FakeWiki:
    """A tiny MediaWiki API: one wiki, one episode page with an infobox and a Plot section."""

    def __init__(self, plot_html, season_text="Season 3"):
        self.plot_html, self.season_text, self.urls = plot_html, season_text, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.urls.append(url)
        class Response:
            def __init__(self, data, status=200):
                self.data, self.status_code = data, status
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError("missing")
            def json(self):
                return self.data
        if not url.startswith("https://reacher.fandom.com"):
            return Response({}, 404)
        if params.get("meta") == "siteinfo":
            return Response({"query": {"general": {"sitename": "Reacher Wiki"}}})
        if params.get("list") == "search":
            return Response({"query": {"search": [{"title": "Persuader (episode)"}, {"title": "Persuader (novel)"}]}})
        if params.get("action") == "parse":
            html = (f'<aside class="portable-infobox">{self.season_text}, Episode 1</aside>'
                    '<p>Intro.</p><h2><span id="Plot">Plot</span></h2>' + self.plot_html
                    + '<h2><span id="Cast">Cast</span></h2><p>Alan Ritchson</p>')
            return Response({"parse": {"text": html}})
        return Response({}, 404)


def test_wiki_plot_is_found_verified_and_cited(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAP_FANDOM", "on")
    plot = "<p>" + " ".join(["Reacher stages the rescue of Richard Beck for Duffy.",
                             "Zachary Beck hires him after Russian roulette.",
                             "Teresa Daniels was held in the cellar."] * 10) + "<sup>[1]</sup></p>"
    wiki = FakeWiki(plot)
    result = deep_research.deepen(bundle(), tmp_path, session=wiki)
    first = result["seasons"][0]["episodes"][0]
    assert first["summary_depth"] == "detailed_plot" and "Teresa Daniels was held in the cellar." in first["summary"]
    assert "[1]" not in first["summary"] and "Alan Ritchson" not in first["summary"]
    source = next(s for s in result["sources"] if s["id"] == "fandom:7:3:1")
    assert source["url"] == "https://reacher.fandom.com/wiki/Persuader_(episode)"
    assert "reacher.fandom.com" in source["title"]


def test_wiki_page_from_another_season_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAP_FANDOM", "on")
    plot = "<p>" + "Reacher meets Richard Beck and Zachary Beck and Duffy and Teresa Daniels. " * 20 + "</p>"
    result = deep_research.deepen(bundle(), tmp_path, session=FakeWiki(plot, season_text="Season 1"))
    assert result["seasons"][0]["episodes"][0]["summary"] == SUMMARY
