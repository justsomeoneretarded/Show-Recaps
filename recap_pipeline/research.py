"""Season-scoped factual research. Retrieved prose is data, never instructions.

Only aired, numbered regular episodes before the requested incoming season can
reach the returned bundle or its cache. TVMaze supplies identity and episode
facts; verified TMDB season overviews can fill gaps when a token is configured.
Series-wide synopses and current posters are deliberately not passed to writers.
"""
from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from datetime import date, datetime, timezone
from urllib.parse import unquote, urlparse, quote

import requests

from movie_pipeline.core import PipelineError

_SCHEMA = 1
_CACHE_REVISION = 5
_TIMEOUT = (5, 25)
_MAX_REQUESTS = 120
_MAX_SECONDS = 360
_DETAIL_LIMIT = 36
_TMDB_SEASON_LIMIT = 50
_WIKI_SEASON_LIMIT = 24
_WIKI_EPISODE_LIMIT = 20
_WIKIDATA = "https://www.wikidata.org/w/api.php"
_WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_RETRY_STATUS = {429, 500, 502, 503, 504}
_TVMAZE = "https://api.tvmaze.com"
_TMDB = "https://api.themoviedb.org/3"


def _get(spec, name, default=None):
    return spec.get(name, default) if isinstance(spec, dict) else getattr(spec, name, default)


def _norm(value):
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", value.casefold()))


def _integer(value):
    return type(value) is int


class _Plain(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _plain(value):
    if not isinstance(value, str):
        return ""
    parser = _Plain()
    parser.feed(value[:100000])
    return " ".join(html.unescape("".join(parser.parts)).split())[:16000]


def _useful(value):
    text = _plain(value)
    return (len(text) >= 40 and len(text.split()) >= 8
            and not re.fullmatch(r"(?:no (?:plot |episode )?(?:summary|overview).*|"
                                 r"(?:summary|overview|plot) (?:is )?(?:unavailable|"
                                 r"not available|to be announced).*|tba\W*)",
                                 text, re.I))


def _date(value):
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _year(value):
    parsed = _date(value)
    return parsed.year if parsed else None


def _safe_url(value, hosts):
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in hosts:
        return None
    if parsed.username or parsed.password:
        return None
    return parsed._replace(scheme="https", fragment="").geturl()


def _image(value):
    if isinstance(value, dict):
        value = value.get("original") or value.get("medium")
    return _safe_url(value, {"static.tvmaze.com", "image.tmdb.org"})


def _aired(episode, today):
    aired = _date(episode.get("airdate"))
    if not aired or aired > today:
        return False
    stamp = episode.get("airstamp")
    if stamp:
        try:
            moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                return False
            # Historical test dates have an unambiguous end-of-day cutoff.
            cutoff = datetime.now(timezone.utc) if today == datetime.now(timezone.utc).date() else datetime.combine(
                today, datetime.max.time(), tzinfo=timezone.utc)
            if moment > cutoff:
                return False
        except (ValueError, TypeError, AttributeError):
            return False
    elif aired == today:
        # An airdate alone cannot establish that today's episode has aired.
        return False
    return True


class _HTTP:
    def __init__(self, session):
        self.session = session
        self.count = 0
        self.deadline = time.monotonic() + _MAX_SECONDS

    def json(self, url, *, params=None, token=None):
        headers = {"User-Agent": "ShowRecapsResearch/1.0 (https://github.com/justsomeoneretarded/Top10-Movies)"}
        if token:
            headers["Authorization"] = "Bearer " + token
        for attempt in range(3):
            if time.monotonic() >= self.deadline:
                raise PipelineError("Research time budget exhausted; missing facts remain explicit gaps.")
            if self.count >= _MAX_REQUESTS:
                raise PipelineError("Research request budget exhausted; no episodes were truncated.")
            self.count += 1
            try:
                response = self.session.get(url, params=params, headers=headers,
                                            timeout=_TIMEOUT, allow_redirects=False)
                status = response.status_code
                if status in _RETRY_STATUS:
                    if attempt < 2:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise PipelineError(f"Research source unavailable after 3 attempts (HTTP {status}).")
                if status == 404:
                    return None
                if not 200 <= status < 300:
                    raise PipelineError(f"Research source rejected the request (HTTP {status}).")
                if len(response.content) > _MAX_JSON_BYTES:
                    raise PipelineError("Research source response exceeded the safe size limit.")
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    # Do not echo exception text: upstream URLs/headers can contain secrets.
                    raise PipelineError("Research source failed after 3 attempts; check connectivity.") from exc
                time.sleep(0.2 * (attempt + 1))
        raise PipelineError("Research request failed.")


def _matches(show, title, year, *, tmdb=False):
    if not isinstance(show, dict):
        return False
    names = [show.get("name"), show.get("original_name")] if tmdb else [show.get("name")]
    premiered = show.get("first_air_date") if tmdb else show.get("premiered")
    return any(name and _norm(name) == _norm(title) for name in names) and _year(premiered) == year


def _resolve_show(http, title, year, ids):
    if ids.get("tvmaze_id"):
        candidate = http.json(f"{_TVMAZE}/shows/{ids['tvmaze_id']}")
        candidates = [candidate] if candidate else []
    else:
        results = http.json(f"{_TVMAZE}/search/shows", params={"q": title})
        if not isinstance(results, list):
            raise PipelineError("TVMaze returned an invalid show search response.")
        candidates = [item.get("show") for item in results if isinstance(item, dict)]
    matches = {item.get("id"): item for item in candidates
               if _matches(item, title, year) and _integer(item.get("id"))
               and (not ids.get("tvmaze_id") or item["id"] == ids["tvmaze_id"])
               and all(not ids.get(key) or str((item.get("externals") or {}).get(external_key)) == str(ids[key])
                       for key, external_key in (("imdb_id", "imdb"), ("tvdb_id", "thetvdb")))}
    if len(matches) != 1:
        if len(matches) > 1:
            raise PipelineError(f"Multiple TVMaze shows match {title} ({year}); specify tvmaze_id.")
        raise PipelineError(f"No exact TVMaze match for {title} ({year}); check the premiere year and title.")
    show = dict(next(iter(matches.values())))
    ambiguity_candidates = candidates
    if ids.get("tvmaze_id"):
        try:
            results = http.json(f"{_TVMAZE}/search/shows", params={"q": title})
            if not isinstance(results, list):
                raise PipelineError("Invalid title ambiguity search response.")
            ambiguity_candidates = [item.get("show") for item in results if isinstance(item, dict)]
        except PipelineError:
            show["_recap_ambiguity_unknown"] = True
    exact_productions = {item["id"] for item in ambiguity_candidates
                         if isinstance(item, dict) and _integer(item.get("id"))
                         and _norm(item.get("name")) == _norm(title)}
    show["_recap_ambiguous_title"] = len(exact_productions) > 1 or show.get("_recap_ambiguity_unknown", False)
    external = show.get("externals") or {}
    for key, external_key in (("imdb_id", "imdb"), ("tvdb_id", "thetvdb")):
        if ids.get(key) and str(external.get(external_key)) != str(ids[key]):
            raise PipelineError(f"The selected TVMaze show does not match the supplied {key}.")
    return show


def _resolve_tmdb(http, show, title, year, ids, token):
    if ids.get("tmdb_id"):
        candidates = [http.json(f"{_TMDB}/tv/{ids['tmdb_id']}", token=token)]
    else:
        external = show.get("externals") or {}
        imdb, tvdb = external.get("imdb"), external.get("thetvdb")
        if imdb or tvdb:
            identifier, source = (imdb, "imdb_id") if imdb else (tvdb, "tvdb_id")
            result = http.json(f"{_TMDB}/find/{identifier}",
                               params={"external_source": source}, token=token)
            candidates = result.get("tv_results", []) if isinstance(result, dict) else []
        else:
            result = http.json(f"{_TMDB}/search/tv",
                               params={"query": title, "first_air_date_year": year}, token=token)
            candidates = result.get("results", []) if isinstance(result, dict) else []
    matches = {item.get("id"): item for item in candidates
               if _matches(item, title, year, tmdb=True) and _integer(item.get("id"))}
    if len(matches) != 1:
        raise PipelineError("TMDB fallback could not verify a unique title and premiere-year match.")
    return next(iter(matches))


def _wiki_title(href):
    if not isinstance(href, str):
        return None
    parsed = urlparse(href)
    if parsed.hostname and parsed.hostname != "en.wikipedia.org":
        return None
    if parsed.path.startswith("/wiki/"):
        return unquote(parsed.path[6:]).replace("_", " ")
    return None


class _EpisodeTable(HTMLParser):
    """Read only episode rows and lead links; ignore prose outside the table."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.lead_links = [], []
        self.row = self.cell = None
        self.in_lead = True
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "h2":
            self.in_lead = False
        if tag == "tr":
            self.row, self.cell = [], None
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = {"class": attrs.get("class", ""), "text": [], "links": []}
            self.row.append(self.cell)
        elif tag == "a":
            linked = _wiki_title(attrs.get("href"))
            if linked:
                if self.in_lead:
                    self.lead_links.append(linked)
                if self.cell is not None:
                    self.cell["links"].append(linked)
        elif tag in {"br", "p", "div"} and self.cell is not None:
            self.cell["text"].append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag == "tr":
            if self.row is not None:
                self.rows.append(self.row)
            self.row = self.cell = None
        elif tag in {"td", "th"}:
            self.cell = None
        elif tag in {"p", "div"} and self.cell is not None:
            self.cell["text"].append(" ")

    def handle_data(self, data):
        if self.cell is not None and not self.hidden:
            self.cell["text"].append(data)


def _wiki_clean(text):
    return " ".join(re.sub(r"\[(?:\d+|[a-z])\]", "", _plain(text)).split())


def _table_entries(parser):
    entries, pending = [], None
    for row in parser.rows:
        summary = next((index for index, cell in enumerate(row)
                        if "summary" in cell["class"].split()), None)
        if summary is not None:
            cell = row[summary]
            title = _wiki_clean("".join(cell["text"])).strip('"“” ')
            numeric = [_wiki_clean("".join(value["text"])) for value in row[:summary]]
            numeric = [int(value) for value in numeric if re.fullmatch(r"\d+", value)]
            if not numeric:
                pending = None
                continue
            links = [link for link in cell["links"]
                     if _norm(link.split("(")[0]) == _norm(title)]
            pending = {"number": numeric[-1], "title": title,
                       "page": links[0] if len(set(links)) == 1 else None, "plot": ""}
            entries.append(pending)
        elif pending:
            descriptions = [cell for cell in row if "description" in cell["class"].split()]
            if descriptions:
                pending["plot"] = _wiki_clean(" ".join("".join(cell["text"]) for cell in descriptions))
                pending = None
    return entries


def _wiki_series(http, imdb):
    if not isinstance(imdb, str) or not re.fullmatch(r"tt\d{5,12}", imdb):
        return None
    found = http.json(_WIKIDATA, params={"action": "query", "list": "search",
                      "srsearch": f"haswbstatement:P345={imdb}", "srlimit": 3, "format": "json"})
    hits = found.get("query", {}).get("search", []) if isinstance(found, dict) else []
    entity_ids = [hit.get("title") for hit in hits if isinstance(hit, dict)
                  and re.fullmatch(r"Q\d+", str(hit.get("title")))]
    if not entity_ids:
        return None
    entities = http.json(_WIKIDATA, params={"action": "wbgetentities", "ids": "|".join(entity_ids),
                         "props": "claims|sitelinks", "sitefilter": "enwiki", "format": "json"})
    matches = set()
    for entity in (entities.get("entities", {}) if isinstance(entities, dict) else {}).values():
        claims = entity.get("claims", {}).get("P345", [])
        if not any(claim.get("mainsnak", {}).get("datavalue", {}).get("value") == imdb
                   for claim in claims if isinstance(claim, dict)):
            continue
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
        if isinstance(title, str) and title:
            matches.add(title)
    return next(iter(matches)) if len(matches) == 1 else None


def _season_candidate(page, series_page, season):
    # Title shape is only a prefilter. The parsed season page must also link
    # from its lead to the independently IMDb-verified series article.
    base = _norm(series_page.split("(")[0])
    value = _norm(page)
    tail = next((f"{label} {season}" for label in ("season", "series")
                 if value.endswith(f"{label} {season}")), None)
    if not tail or not value.startswith(base + " "):
        return False
    middle = value[len(base): -len(tail)].strip()
    allowed = set(_norm(series_page)[len(base):].split())
    return all(word in allowed for word in middle.split())


def _wiki_request(http, **params):
    return http.json(_WIKIPEDIA, params={"format": "json", "formatversion": 2, **params})


def _wiki_html(data):
    if not isinstance(data, dict) or not isinstance(data.get("parse"), dict):
        return None, ""
    result = data["parse"]
    text = result.get("text", "")
    if isinstance(text, dict):
        text = text.get("*", "")
    return result.get("title"), text if isinstance(text, str) else ""


def _plot_depth(text, source_type):
    """Known plot sources plus substantive prose; a long teaser is still an overview."""
    if source_type not in {"wikipedia_episode_plot", "wikipedia_episode_table"}:
        return False
    minimum = 60 if source_type == "wikipedia_episode_plot" else 45
    if len(text.split()) < minimum:
        return False
    # Ignore questions inside quoted dialogue. Unanswered promotional questions
    # with little declarative story material do not establish a detailed plot.
    unquoted = re.sub(r'"[^"\n]*"|“[^”]*”', "", text)
    sentences = re.split(r"(?<=[.!?])\s+", unquoted)
    statements = [part for part in sentences if part.strip() and "?" not in part
                  and not re.match(r"^(?:will|can|could|what|who|how|when|where)\b", part.strip(), re.I)]
    return not ("?" in unquoted and len(statements) < 2)


def _quality(seasons):
    episodes = [episode for season in seasons for episode in season["episodes"]]
    detailed = sum(episode["detailed_plot_available"] for episode in episodes)
    gaps = [f"S{season['number']:02}E{season['episodes'][-1]['number']:02}"
            for season in seasons if not season["episodes"][-1]["detailed_plot_available"]]
    return {"aired_episodes": len(episodes), "episodes_with_detailed_plot": detailed,
            "episodes_with_overview_only": sum(bool(episode["summary"]) and not episode["detailed_plot_available"]
                                               for episode in episodes),
            "all_episodes_have_detailed_plots": detailed == len(episodes),
            "detailed_finale_gaps": gaps,
            # Source coverage alone cannot certify every twist or ending.
            "can_support_complete_recap": False, "complete_story_verified": False}


def _enrich_wikipedia(http, show, seasons, sources, warnings):
    """Read verified season tables or exact prior-season sections, then episode plots."""
    links = {}
    source_ids = {source["id"] for source in sources}
    series_metadata = None

    def use(item, text, source, source_type):
        if not _useful(text):
            return
        detailed = _plot_depth(text, source_type)
        if detailed or not item["summary"]:
            item["summary"] = text
            item["summary_depth"] = "detailed_plot" if detailed else "episode_overview"
            item["summary_source_type"] = source_type
            item["detailed_plot_available"] = detailed
            item["summary_source_ids"] = [source["id"]]
            item["source_texts"][source["id"]] = text
            if source["id"] not in item["sources"]:
                item["sources"].append(source["id"])
            if source["id"] not in source_ids:
                sources.append(source)
                source_ids.add(source["id"])

    def apply_table(parser, season, page, section=None):
        entries = _table_entries(parser)
        matched = False
        for item in season["episodes"]:
            matches = [entry for entry in entries if entry["number"] == item["number"]
                       and _norm(entry["title"]) == _norm(item["title"])]
            if len(matches) != 1:
                continue
            matched = True
            entry = matches[0]
            if entry["page"]:
                links[(season["number"], item["number"])] = entry["page"]
            source = {"id": f"wikipedia:season:{show['id']}:{season['number']}",
                      "url": "https://en.wikipedia.org/wiki/" + quote(page.replace(" ", "_"), safe="()_")
                             + ("#" + quote(section.replace(" ", "_")) if section else ""),
                      "title": page + " — episode plot table", "season": season["number"], "episode": None}
            use(item, entry["plot"], source, "wikipedia_episode_table")
        return matched

    def scoped_series_table(series_page, season):
        nonlocal series_metadata
        if series_metadata is None:
            data = _wiki_request(http, action="parse", page=series_page, prop="sections")
            parsed = data.get("parse", {}) if isinstance(data, dict) else {}
            series_metadata = parsed.get("sections", []) if parsed.get("title") == series_page else []
        sections = series_metadata if isinstance(series_metadata, list) else []
        in_episodes, level, matches = False, 0, []
        for section in sections:
            if not isinstance(section, dict):
                continue
            line = _plain(section.get("line"))
            try:
                current_level = int(section.get("toclevel", 0))
            except (ValueError, TypeError):
                continue
            if line.casefold() == "episodes":
                in_episodes, level = True, current_level
                continue
            if in_episodes and current_level <= level:
                in_episodes = False
            if not in_episodes:
                continue
            match = re.fullmatch(r"(?:season|series)\s+(\d+)(?:\s*[(:–-].*)?", line, re.I)
            if match and int(match.group(1)) == season["number"] and re.fullmatch(r"\d+", str(section.get("index"))):
                matches.append(section)
        if len(matches) != 1:
            return
        section = matches[0]
        data = _wiki_request(http, action="parse", page=series_page, prop="text", section=str(section["index"]))
        page, body = _wiki_html(data)
        if page != series_page:
            return
        parser = _EpisodeTable()
        parser.feed(body)
        apply_table(parser, season, page, _plain(section.get("line")))

    try:
        series_page = _wiki_series(http, (show.get("externals") or {}).get("imdb"))
        if not series_page:
            warnings.append("Wikipedia detailed plots unavailable: no unique IMDb-verified series article.")
            return
        for season in reversed(seasons[-_WIKI_SEASON_LIMIT:]):
            number = season["number"]
            matched = False
            # CirrusSearch boolean operators do not combine intitle parameters.
            # Search independently, stopping once a verified table is found.
            for label in ("season", "series"):
                result = _wiki_request(http, action="query", list="search",
                                       srsearch=f'intitle:"{series_page.split("(")[0].strip()}" intitle:"{label} {number}"',
                                       srlimit=5, srprop="")
                hits = result.get("query", {}).get("search", []) if isinstance(result, dict) else []
                candidates = list(dict.fromkeys(hit["title"] for hit in hits
                    if isinstance(hit, dict) and isinstance(hit.get("title"), str)
                    and _season_candidate(hit["title"], series_page, number)))
                for candidate in candidates[:2]:
                    parsed = _wiki_request(http, action="parse", page=candidate, prop="text")
                    page, body = _wiki_html(parsed)
                    if not page or not _season_candidate(page, series_page, number):
                        continue
                    parser = _EpisodeTable()
                    parser.feed(body)
                    parent_links = [link for link in parser.lead_links
                                    if _norm(link.split("(")[0]) == _norm(series_page.split("(")[0])]
                    if not parent_links or _norm(parent_links[0]) != _norm(series_page):
                        continue
                    matched = apply_table(parser, season, page)
                    if matched:
                        break
                if matched:
                    break
            if not matched:
                # Metadata and the exact season subsection are safe; never fetch
                # the general series synopsis or an incoming-season section.
                scoped_series_table(series_page, season)
        if len(seasons) > _WIKI_SEASON_LIMIT:
            warnings.append(f"Wikipedia season enrichment capped at {_WIKI_SEASON_LIMIT}; all prior seasons remain in research.")

        finales = [season["episodes"][-1] for season in reversed(seasons)]
        sparse = [item for season in reversed(seasons) for item in season["episodes"]
                  if not item["detailed_plot_available"]]
        priority, selected = [], set()
        for item in finales + sparse:
            pair = (item["season"], item["number"])
            if pair in links and pair not in selected:
                priority.append(item)
                selected.add(pair)
        for item in priority[:_WIKI_EPISODE_LIMIT]:
            page = links[(item["season"], item["number"])]
            try:
                metadata = _wiki_request(http, action="parse", page=page, prop="sections")
                meta = metadata.get("parse", {}) if isinstance(metadata, dict) else {}
                canonical = meta.get("title")
                if not canonical or _norm(canonical.split("(")[0]) != _norm(item["title"]):
                    continue
                sections = meta.get("sections", [])
                plot = next((section for section in sections if isinstance(section, dict)
                             and _plain(section.get("line")).casefold() in {"plot", "synopsis", "plot summary"}), None)
                if not plot or not re.fullmatch(r"\d+", str(plot.get("index"))):
                    continue
                parsed = _wiki_request(http, action="parse", page=canonical, prop="text", section=str(plot["index"]))
                resolved, body = _wiki_html(parsed)
                if resolved != canonical:
                    continue
                text = _wiki_clean(body)
                text = re.sub(r"^(?:Plot summary|Synopsis|Plot)\s*(?:\[edit\])?\s*", "", text, flags=re.I)
                source = {"id": f"wikipedia:episode:{show['id']}:{item['season']}:{item['number']}",
                          "url": "https://en.wikipedia.org/wiki/" + quote(canonical.replace(" ", "_"), safe="()_") + "#" + quote(_plain(plot["line"]).replace(" ", "_")),
                          "title": canonical + " — plot section", "season": item["season"], "episode": item["number"]}
                use(item, text, source, "wikipedia_episode_plot")
            except (PipelineError, ValueError, TypeError, KeyError):
                continue
        if len(priority) > _WIKI_EPISODE_LIMIT:
            warnings.append(f"Wikipedia episode plot requests capped at {_WIKI_EPISODE_LIMIT}; detailed coverage is reported explicitly.")
    except (PipelineError, ValueError, TypeError, KeyError, AttributeError):
        warnings.append("Wikipedia detailed plot enrichment was unavailable or could not be verified.")
    finally:
        quality = _quality(seasons)
        if quality["detailed_finale_gaps"]:
            warnings.append("Detailed plot unavailable for prior-season finale(s): "
                            + ", ".join(quality["detailed_finale_gaps"])
                            + ". Episode overviews may omit twists and endings; complete story coverage is not verified.")
        if not quality["all_episodes_have_detailed_plots"]:
            warnings.append(f"Detailed plot coverage: {quality['episodes_with_detailed_plot']}/{quality['aired_episodes']} aired episodes; "
                            "remaining summaries are overviews or explicit gaps. Never infer missing plot.")


def _episode_source(show, episode):
    episode_id = episode["id"]
    return {
        "id": f"tvmaze:episode:{episode_id}",
        "url": _safe_url(episode.get("url"), {"www.tvmaze.com", "tvmaze.com"})
               or f"https://www.tvmaze.com/episodes/{episode_id}",
        "title": f"{show['name']} — S{episode['season']:02}E{episode['number']:02} — {episode.get('name') or 'Untitled'}",
        "season": episode["season"], "episode": episode["number"],
    }


def _validate_cached(bundle, identity):
    """Reject stale/wrong-scope cache files rather than trusting a file name."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != _SCHEMA:
        return False
    show = bundle.get("show")
    if not isinstance(show, dict) or not _integer(show.get("id")):
        return False
    if _norm(show.get("title")) != identity["title"] or show.get("year") != identity["year"]:
        return False
    if identity["ids"].get("tvmaze_id") and show["id"] != identity["ids"]["tvmaze_id"]:
        return False
    if any(identity["ids"].get(key) and str(show.get(key)) != str(identity["ids"][key])
           for key in ("imdb_id", "tvdb_id")):
        return False
    if show.get("summary") != "" or type(show.get("ambiguous_title")) is not bool:
        return False
    sources = bundle.get("sources")
    warnings = bundle.get("warnings")
    seasons = bundle.get("seasons")
    if not isinstance(sources, list) or not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings):
        return False
    if not isinstance(seasons, list):
        return False
    expected = list(range(1, identity["upcoming_season"]))
    if [season.get("number") for season in seasons if isinstance(season, dict)] != expected:
        return False
    if not isinstance(bundle.get("research_quality"), dict):
        return False
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            return False
        if source.get("season") is not None and source["season"] not in expected:
            return False
        if source["id"] in source_ids:
            return False
        source_ids.add(source["id"])
    for season in seasons:
        episodes = season.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return False
        numbers = []
        for episode in episodes:
            if not isinstance(episode, dict) or episode.get("season") != season["number"]:
                return False
            number = episode.get("number")
            if not _integer(number) or number < 1:
                return False
            aired = _date(episode.get("airdate"))
            if not aired or aired > date.fromisoformat(identity["as_of"]):
                return False
            if not isinstance(episode.get("summary"), str) or not isinstance(episode.get("sources"), list):
                return False
            if episode.get("summary_depth") not in {"missing", "episode_overview", "detailed_plot"}:
                return False
            if type(episode.get("detailed_plot_available")) is not bool:
                return False
            if episode["detailed_plot_available"] != (episode["summary_depth"] == "detailed_plot"):
                return False
            if (not isinstance(episode.get("source_texts"), dict)
                    or not isinstance(episode.get("summary_source_ids"), list)
                    or not set(episode["summary_source_ids"]).issubset(set(episode["sources"]))
                    or any(episode["source_texts"].get(source_id) != episode["summary"]
                           for source_id in episode["summary_source_ids"])
                    or bool(episode["summary"]) != bool(episode["summary_source_ids"])):
                return False
            if not episode["sources"] or not set(episode["sources"]).issubset(source_ids):
                return False
            numbers.append(number)
        if numbers != sorted(set(numbers)) or not any(_useful(e["summary"]) for e in episodes):
            return False
    return bool(seasons and _useful(seasons[-1]["episodes"][-1]["summary"])
                and bundle["research_quality"] == _quality(seasons))


def _write_cache(path, payload):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def research_show(spec, cache_dir: Path, *, session=None, today: date | None = None) -> dict:
    """Resolve one show and research seasons 1..upcoming_season-1.

    Optional identity fields: tvmaze_id, imdb_id, tvdb_id, tmdb_id. A Requests
    session and UTC date can be injected for offline tests. Empty episode
    summaries are explicit gaps, never inferred plot. No season or episode is
    truncated to meet a request limit.
    """
    title, year, incoming = _get(spec, "show_title"), _get(spec, "premiere_year"), _get(spec, "upcoming_season")
    if not isinstance(title, str) or not title.strip() or len(title) > 300:
        raise PipelineError("Provide one show title.")
    title = title.strip()
    if not _integer(year) or not 1900 <= year <= 2100:
        raise PipelineError("Provide the show's original premiere year.")
    if not _integer(incoming) or not 2 <= incoming <= 101:
        raise PipelineError("Upcoming season must be between 2 and 101; season 1 has no earlier seasons to recap.")
    today = today or datetime.now(timezone.utc).date()
    ids = {name: _get(spec, name) for name in ("tvmaze_id", "tvdb_id", "tmdb_id", "imdb_id") if _get(spec, name) is not None}
    for name in ("tvmaze_id", "tvdb_id", "tmdb_id"):
        if name in ids and (not _integer(ids[name]) or ids[name] < 1):
            raise PipelineError(f"{name} must be a positive integer.")
    if "imdb_id" in ids and not re.fullmatch(r"tt\d{5,12}", str(ids["imdb_id"])):
        raise PipelineError("imdb_id must be an IMDb title ID such as tt0944947.")
    token = os.getenv("TMDB_READ_TOKEN", "").strip()
    identity = {"schema": _SCHEMA, "revision": _CACHE_REVISION, "title": _norm(title), "year": year,
                "upcoming_season": incoming, "ids": ids, "as_of": today.isoformat(),
                "tmdb_available": bool(token)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir) / f"research-{digest}.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("identity") == identity and _validate_cached(cached.get("bundle"), identity):
            return cached["bundle"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    owned = session is None
    session = session or requests.Session()
    http = _HTTP(session)
    warnings = []
    try:
        show = _resolve_show(http, title, year, ids)
        if show.get("_recap_ambiguity_unknown"):
            warnings.append("Title ambiguity lookup unavailable; footage identity requires positive production evidence.")
        raw = http.json(f"{_TVMAZE}/shows/{show['id']}/episodes", params={"specials": 0})
        if not isinstance(raw, list):
            raise PipelineError("TVMaze returned an invalid episode list.")
        grouped = {number: [] for number in range(1, incoming)}
        excluded = 0
        excluded_ids = []
        incomplete_latest = []
        seen = set()
        seen_ids = set()
        for episode in raw:
            if not isinstance(episode, dict):
                continue
            season, number = episode.get("season"), episode.get("number")
            # Check the season boundary before reading summaries or collecting sources.
            if not _integer(season) or season not in grouped:
                continue
            if not _integer(number) or number < 1 or episode.get("type", "regular") != "regular":
                continue
            if not _aired(episode, today):
                excluded += 1
                excluded_ids.append(f"S{season:02}E{number:02}")
                if season == incoming - 1:
                    incomplete_latest.append(f"E{number:02}")
                continue
            if not _integer(episode.get("id")):
                raise PipelineError("TVMaze episode metadata is missing an episode ID.")
            if episode["id"] in seen_ids:
                raise PipelineError("TVMaze repeats an episode ID across episode records; resolve the source ambiguity.")
            seen_ids.add(episode["id"])
            identity_pair = (season, number)
            if identity_pair in seen:
                raise PipelineError(f"TVMaze lists duplicate episode number S{season:02}E{number:02}; resolve the source ambiguity.")
            seen.add(identity_pair)
            grouped[season].append(episode)
        if incomplete_latest:
            raise PipelineError(f"Latest required prior season {incoming - 1} is not confirmed complete: "
                                "regular episode(s) " + ", ".join(incomplete_latest)
                                + " have future, missing, or unconfirmed air times. "
                                "Wait for the completed season or correct the episode metadata; "
                                "the last aired episode cannot be treated as the finale.")
        if excluded:
            warnings.append(f"Excluded {excluded} prior-season episodes with future, missing, or unconfirmed air times: "
                            + ", ".join(excluded_ids) + ". Older-season metadata gaps remain explicit.")
        missing_seasons = [number for number, episodes in grouped.items() if not episodes]
        if missing_seasons:
            raise PipelineError("No aired regular episodes found for required prior season(s): "
                                + ", ".join(map(str, missing_seasons)) + ". Check the upcoming season.")
        for episodes in grouped.values():
            episodes.sort(key=lambda episode: episode["number"])

        sources = [{"id": f"tvmaze:show:{show['id']}",
                    "url": _safe_url(show.get("url"), {"www.tvmaze.com", "tvmaze.com"})
                           or f"https://www.tvmaze.com/shows/{show['id']}",
                    "title": f"{show['name']} ({year}) — show identity",
                    "season": None, "episode": None}]
        seasons = []
        episode_map = {}
        for season_number, episodes in grouped.items():
            output = []
            for episode in episodes:
                source = _episode_source(show, episode)
                sources.append(source)
                summary = _plain(episode.get("summary"))
                item = {"id": str(episode["id"]), "season": season_number, "number": episode["number"],
                        "title": _plain(episode.get("name")) or f"Episode {episode['number']}",
                        "airdate": episode["airdate"], "summary": summary if _useful(summary) else "",
                        "image": _image(episode.get("image")), "sources": [source["id"]],
                        "summary_depth": "episode_overview" if _useful(summary) else "missing",
                        "detailed_plot_available": False,
                        "summary_source_type": "tvmaze_overview",
                        "summary_source_ids": [source["id"]] if _useful(summary) else [],
                        "source_texts": {source["id"]: summary} if _useful(summary) else {}}
                output.append(item)
                episode_map[episode["id"]] = item
            seasons.append({"number": season_number,
                            "year": min(_year(item["airdate"]) for item in output),
                            "episodes": output})

        # Give detailed plot research priority over promotional overview gaps.
        _enrich_wikipedia(http, show, seasons, sources, warnings)

        # A single list response covers arbitrarily long shows. Extra detail calls
        # are only for gaps, prioritising each season's finale and the latest season.
        finals = [episodes[-1] for episodes in reversed(list(grouped.values()))]
        priority = finals + [episode for episodes in grouped.values() for episode in episodes]
        pending = []
        selected = set()
        for episode in priority:
            if not episode_map[episode["id"]]["summary"] and episode["id"] not in selected:
                pending.append(episode)
                selected.add(episode["id"])
        attempted = 0
        for episode in pending[:_DETAIL_LIMIT]:
            attempted += 1
            try:
                detail = http.json(f"{_TVMAZE}/episodes/{episode['id']}")
                # Detail must belong to the exact prior episode we requested.
                if (not isinstance(detail, dict) or detail.get("id") != episode["id"]
                        or detail.get("season") != episode["season"] or detail.get("number") != episode["number"]
                        or _norm(detail.get("name")) != _norm(episode.get("name"))):
                    continue
                if _useful(detail.get("summary")):
                    episode_map[episode["id"]]["summary"] = _plain(detail["summary"])
                    episode_map[episode["id"]]["summary_depth"] = "episode_overview"
                    episode_map[episode["id"]]["summary_source_ids"] = [f"tvmaze:episode:{episode['id']}"]
                    episode_map[episode["id"]]["source_texts"][f"tvmaze:episode:{episode['id']}"] = _plain(detail["summary"])
            except PipelineError:
                warnings.append(f"TVMaze detail lookup failed for S{episode['season']:02}E{episode['number']:02}.")
        if len(pending) > attempted:
            warnings.append(f"TVMaze detail lookups capped at {_DETAIL_LIMIT}; all episode records are retained.")

        gaps = [season for season in seasons if any(not item["summary"] for item in season["episodes"])]
        if gaps and token:
            try:
                tmdb_id = _resolve_tmdb(http, show, title, year, ids, token)
                fallback_seasons = sorted(gaps, key=lambda season: -season["number"])
                for season in fallback_seasons[:_TMDB_SEASON_LIMIT]:
                    number = season["number"]
                    try:
                        data = http.json(f"{_TMDB}/tv/{tmdb_id}/season/{number}", token=token)
                        if not isinstance(data, dict) or data.get("season_number") != number:
                            continue
                        entries = data.get("episodes", [])
                        if not isinstance(entries, list):
                            continue
                        by_number = {}
                        for entry in entries:
                            if isinstance(entry, dict) and _integer(entry.get("episode_number")):
                                by_number.setdefault(entry["episode_number"], []).append(entry)
                        source_id = f"tmdb:tv:{tmdb_id}:season:{number}"
                        used = False
                        for item in season["episodes"]:
                            candidates = by_number.get(item["number"], [])
                            if item["summary"] or len(candidates) != 1:
                                continue
                            entry = candidates[0]
                            # Number plus title and matching airdate prevent crossed
                            # episode orders from attaching the wrong plot.
                            if (_norm(entry.get("name")) != _norm(item["title"])
                                    or entry.get("air_date") != item["airdate"]
                                    or entry.get("season_number", number) != number
                                    or not _useful(entry.get("overview"))):
                                continue
                            item["summary"] = _plain(entry["overview"])
                            item["summary_depth"] = "episode_overview"
                            item["summary_source_type"] = "tmdb_overview"
                            item["summary_source_ids"] = [source_id]
                            item["source_texts"][source_id] = item["summary"]
                            item["sources"].append(source_id)
                            still = entry.get("still_path")
                            if not item["image"] and isinstance(still, str) and re.fullmatch(r"/[A-Za-z0-9_.-]+", still):
                                item["image"] = "https://image.tmdb.org/t/p/original" + still
                            used = True
                        if used:
                            sources.append({"id": source_id, "url": f"https://www.themoviedb.org/tv/{tmdb_id}/season/{number}",
                                            "title": f"{show['name']} — season {number} episode overviews",
                                            "season": number, "episode": None})
                    except PipelineError:
                        warnings.append(f"Verified TMDB fallback unavailable for season {number}.")
                if len(fallback_seasons) > _TMDB_SEASON_LIMIT:
                    warnings.append(f"TMDB fallback capped at {_TMDB_SEASON_LIMIT} seasons; all seasons are retained.")
            except PipelineError:
                warnings.append("TMDB fallback could not be verified or was unavailable; no fallback plot was used.")
        elif gaps:
            warnings.append("TMDB fallback unavailable without TMDB_READ_TOKEN; unresolved episode plots are left empty.")

        for season in seasons:
            episodes = season["episodes"]
            covered = sum(bool(item["summary"]) for item in episodes)
            if covered != len(episodes):
                missing = ", ".join(f"E{item['number']:02}" for item in episodes if not item["summary"])
                warnings.append(f"Season {season['number']} summary coverage: {covered}/{len(episodes)} aired episodes; missing {missing}.")
            if not covered:
                raise PipelineError(f"Season {season['number']} has no useful sourced episode summaries. "
                                    "Provide TMDB_READ_TOKEN or correct the source metadata; no plot can be invented.")
        if not seasons[-1]["episodes"][-1]["summary"]:
            last = seasons[-1]["episodes"][-1]
            raise PipelineError(f"The latest prior-season finale S{last['season']:02}E{last['number']:02} "
                                "has no useful sourced summary; cannot produce a reliable recap.")

        external = show.get("externals") or {}
        prior_images = [item["image"] for season in seasons for item in season["episodes"] if item["image"]]
        bundle = {"schema_version": _SCHEMA,
                  "show": {"id": show["id"], "title": show["name"], "year": year,
                           "ambiguous_title": show["_recap_ambiguous_title"],
                           "genres": [value for value in (show.get("genres") or []) if isinstance(value, str)],
                           "summary": "", "url": sources[0]["url"],
                           "image": prior_images[-1] if prior_images else None,
                           "imdb_id": external.get("imdb") if isinstance(external.get("imdb"), str) else None,
                           "tvdb_id": external.get("thetvdb") if _integer(external.get("thetvdb")) else None},
                  "seasons": seasons, "sources": sources, "warnings": warnings,
                  "research_quality": _quality(seasons)}
        try:
            _write_cache(path, {"identity": identity, "bundle": bundle})
        except OSError:
            bundle["warnings"].append("Research cache could not be written; this run uses the fetched facts.")
        return bundle
    finally:
        if owned:
            session.close()
