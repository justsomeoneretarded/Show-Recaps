"""Render complete, season-scoped television recaps with the existing audio chain.

Only synthesis and recognition are injectable. Production exports still pass the
same lossless assembly, final-recording review and strict full media decode.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageDraw

from movie_pipeline import audio_review, render, voice
from movie_pipeline.core import PipelineError, duration, ffmpeg, probe, save_json, voice_reference, write_srt
from movie_pipeline.sound_design import delivery_speed
from movie_pipeline.soundtrack import music_credit, narration_rate, theme_for
from movie_pipeline.speech_recovery import SpeechRecovery
from movie_pipeline.subtitles import script_cues
from movie_pipeline.timing import DurationSpec

from .editing import (UsageTracker, analyse_assets, choose_resolution, plan_source_audio,
                      plan_visuals, prefer_sharp_sources, scene_catalogue)
from .gemini import shared_client
from .gemini_media import pick_sentence_shots
from .sync import segments as sentence_segments, sentences as split_sentences, word_times
from .models import voice_speed_value
from .models import validate_script_scope
from .thumbnails import build_thumbnails

END_SCREEN_SECONDS = 10.0
ENDING_HOLD_SECONDS = 3.0
DELIVERIES = {"normal", "calm", "hype", "dramatic"}


@contextmanager
def _environment(values):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _value(spec, name, default=None):
    return spec.get(name, default) if isinstance(spec, dict) else getattr(spec, name, default)


def _episode_seasons(bundle):
    result = {}
    for season in bundle.get("seasons", []):
        for episode in season.get("episodes", []):
            if episode.get("id") is not None:
                result[str(episode["id"])] = int(episode["season"])
    return result


def _paragraph_episodes(section, index, episode_seasons):
    """Only claim an exact episode when paragraph provenance establishes it."""
    explicit = section.get("paragraph_episode_ids")
    if isinstance(explicit, list) and index < len(explicit) and isinstance(explicit[index], list):
        ids = explicit[index]
    else:
        ids = section.get("episode_ids", [])
    current = [str(value) for value in ids
               if episode_seasons.get(str(value)) == section.get("season")]
    # A multi-episode paragraph has no single current episode for native audio.
    return list(dict.fromkeys(current)) if explicit is not None or len(set(current)) == 1 else []


def _part_seconds(path):
    metadata = probe(path)
    ends = [float(stream["duration"]) + max(0.0, float(stream.get("start_time") or 0))
            for stream in metadata.get("streams", [])
            if stream.get("codec_type") in {"audio", "video"} and stream.get("duration")]
    if not ends or not all(math.isfinite(value) and value > 0 for value in ends):
        raise PipelineError("A rendered recap part has no complete media duration.")
    return max(ends)


def _chapter_title(section, show_title=""):
    title = " ".join(str(section.get("title") or "The story").split())
    show = " ".join(str(show_title or "").split())
    # "The Terminal List: A Mission of Revenge" reads as "A Mission of Revenge" in a chapter.
    if show and title.casefold().startswith(show.casefold()):
        rest = title[len(show):].lstrip(" :–-—")
        title = rest or title
    season = section.get("season")
    if season is not None and not title.casefold().startswith(f"season {season}".casefold()):
        return f"Season {season}: {title}"
    return title


def _label(path, section, show_title=""):
    """Season/chapter lower third, sharing the existing restrained animation."""
    image = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    title = _chapter_title(section, show_title)
    face = render.brand_font("body-medium", 34)
    lines = render.wrapped(draw, title, face, 1090)
    while len(lines) > 2 and face.size > 20:
        face = render.brand_font("body-medium", face.size - 2)
        lines = render.wrapped(draw, title, face, 1090)
    top = 662 - len(lines) * (face.size + 7) - 18
    draw.rounded_rectangle((48, top, 1232, 680), radius=14, fill=(10, 16, 25, 232))
    draw.rounded_rectangle((48, top, 55, 680), radius=3, fill=(114, 183, 180, 255))
    for index, line in enumerate(lines):
        draw.text((78, top + 14 + index * (face.size + 7)), line, font=face, fill="white")
    image.save(path)


def _end_screen(path, show_title):
    """Keep space clear for the platform's end-screen video and subscribe links."""
    canvas = Image.new("RGB", (1280, 720), "#101923")
    draw = ImageDraw.Draw(canvas)
    draw.text((640, 95), "THANKS FOR WATCHING", anchor="mm",
              font=render.brand_font("display", 58), fill="white")
    draw.rounded_rectangle((580, 135, 700, 141), radius=3, fill="#72b7b4")
    face = render.brand_font("body-medium", 28)
    title = " ".join(str(show_title).split())
    while draw.textlength(title, font=face) > 1120 and face.size > 16:
        face = render.brand_font("body-medium", face.size - 2)
    draw.text((640, 182), title, anchor="mm", font=face, fill="#cdd2dc")
    draw.text((640, 654), "More stories. More recaps.", anchor="mm",
              font=render.brand_font("body-medium", 23), fill="#9eafbf")
    canvas.save(path)


class _DeliverySpeaker:
    """One selected model and cloned reference across all delivery labels."""
    def __init__(self, reference, cache):
        self.reference, self.cache = reference, cache
        self.delivery = "normal"
        self.model = None

    def speak(self, text, attempt_offset=0, *, force=False):
        if self.model is None:
            self.model = voice.Narrator(self.reference, self.cache)
        return self.model.speak(text, attempt_offset=attempt_offset,
                                force=force, delivery=self.delivery)

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None


def _paragraph_events(section, index):
    groups = section.get("paragraph_event_ids")
    if isinstance(groups, list) and index < len(groups) and isinstance(groups[index], list):
        return [str(value) for value in groups[index]]
    return [str(value) for value in section.get("event_ids") or []]


def _sentence_shots(gemini, section, text, aligned, seconds, pool, usage, episode_ids, events, context):
    """One shot per narrated sentence, cut so each key word lands on its new shot.

    Gemini picks shots and the key word for every sentence in one request per
    paragraph; without it, scenes linked to the paragraph's events still lead.
    """
    sentences = split_sentences(text)
    picks = pick_sentence_shots(gemini, sentences, scene_catalogue(
        section, text, pool, usage, episode_ids=episode_ids), context=context)
    words = word_times(text, aligned, seconds)
    plan = sentence_segments(text, words, seconds, {i: p["key_word"] for i, p in picks.items() if p.get("key_word")})
    chosen, cuts = [], []
    for segment in plan:
        span = segment["end"] - segment["start"]
        keys = (picks.get(segment["sentence"]) or {}).get("keys") or []
        shots = plan_visuals(section, segment["text"], span, pool, usage, episode_ids=episode_ids,
                             preferred=keys, events=events)
        chosen.extend(shots)
        cuts.append({"start": round(segment["start"], 3), "end": round(segment["end"], 3),
                     "sentence": segment["sentence"],
                     "key_word": (picks.get(segment["sentence"]) or {}).get("key_word", ""),
                     "linked_event": any(shot.get("event_id") in events for shot in shots)})
    if not chosen:
        raise PipelineError("No scoped visual plan covers the complete recap paragraph.")
    return chosen, cuts


def _make_recovery(spec, work):
    # Imported lazily so planner and synthetic FFmpeg tests download no models.
    from faster_whisper import WhisperModel

    names = {"warguy": "warguy", "kitlaser": "kit-laser", "kit laser": "kit-laser",
             "kit-laser": "kit-laser", "female": "female"}
    choice = str(_value(spec, "narrator", "Warguy")).casefold()
    if choice not in names:
        raise PipelineError("Choose Warguy, KitLaser, or Female for recap narration.")
    reference = voice_reference(names[choice], Path(os.getenv("VOICE_DIR", "assets/voices")))
    speaker = _DeliverySpeaker(reference, work / "speech")
    recognizer = WhisperModel("base.en", device="cpu", compute_type="int8",
                              cpu_threads=min(os.cpu_count() or 2, 4),
                              download_root=os.getenv("SPEECH_VERIFY_MODEL_DIR") or None)
    recovery = SpeechRecovery(recognizer, lambda: speaker, work / "speech-checks",
        fallback_factory=lambda: WhisperModel("small.en", device="cpu", compute_type="int8",
            cpu_threads=min(os.cpu_count() or 2, 4),
            download_root=os.getenv("SPEECH_VERIFY_MODEL_DIR") or None))
    return recovery, speaker, hashlib.sha256(reference.read_bytes()).hexdigest()


def _validate_timeline(timeline, script, spec, bundle):
    allowed = validate_script_scope(script, spec, bundle)
    episodes = _episode_seasons(bundle)
    sections = {section["id"]: section for section in script["sections"]}
    for event in timeline:
        if event["type"] == "end-screen":
            continue
        section = sections.get(event.get("section_id"))
        if section is None:
            raise PipelineError("A rendered event does not belong to a verified recap section.")
        for shot in event.get("shots", []):
            season = shot.get("season")
            ids = shot.get("episode_ids") or ([shot["episode_id"]] if shot.get("episode_id") else [])
            if season is None:
                raise PipelineError("Unscoped recap footage was selected.")
            if season not in allowed:
                raise PipelineError("Recap footage escaped its current section's season.")
            orientation = section.get("season") is None
            if not orientation and season != section.get("season") and (
                    shot.get("selection") != "other-season-fallback" or ids):
                raise PipelineError("Recap footage escaped its current section's season.")
            if any(episodes.get(str(value)) != season for value in ids):
                raise PipelineError("Selected footage has an unknown or wrong-season episode.")
            current = set(event.get("episode_ids") or [])
            # Intro/outro footage illustrates the show, not a paragraph's episode.
            if ids and not orientation and (not current or not set(map(str, ids)).issubset(current)):
                raise PipelineError("Selected footage does not belong to the current paragraph episode.")
            if event["type"] == "trailer-audio" and (len(current) != 1 or set(map(str, ids)) != current):
                raise PipelineError("Native source audio requires one known current episode.")


def _strict_decode(path):
    ffmpeg("-xerror", "-err_detect", "explode", "-i", path,
           "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-")


def _youtube_time(seconds):
    seconds = max(0, int(seconds))
    return (f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"
            if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02}")


def _description_chapters(chapters, total):
    """YouTube needs ten-second chapters; merge short labels without padding media."""
    result = []
    for chapter in chapters:
        item = {"start": float(chapter["start"]), "title": str(chapter["title"])}
        if not result:
            item["start"] = 0.0
            result.append(item)
        elif item["start"] - result[-1]["start"] < 10:
            result[-1]["title"] += " / " + item["title"]
        else:
            result.append(item)
    if len(result) > 1 and total-result[-1]["start"] < 10:
        result[-2]["title"] += " / " + result[-1]["title"]
        result.pop()
    return result


def _credits(bundle, assets, script, theme):
    source_ids = set(script.get("source_ids") or [])
    research = [source for source in bundle.get("sources", [])
                if not source_ids or source.get("id") in source_ids]
    records = [*research, *assets.get("sources", [])]
    lines, seen = [], set()
    for item in records:
        url = str(item.get("url") or item.get("source_url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = " ".join(str(item.get("title") or item.get("name") or "Source").split())
        lines.append(f"- {title}: {url}")
    lines.append(music_credit(theme))
    return "\n".join(lines) + "\n"


def render_recap(script, spec, bundle, assets, output: Path, work: Path, *, recovery=None) -> dict:
    """Export one fully verified recap package; runtime guidance never cuts speech."""
    output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    final = output / "final-video.mp4"
    # A failed retry must never leave a stale video that appears to be this run.
    final.unlink(missing_ok=True)
    report = {"status": "rendering", "kind": script.get("kind"), "warnings": [],
              "speech": {"narrator": _value(spec, "narrator"), "tts_model": _value(spec, "tts_model", "original")},
              "audio_chain": "existing movie_pipeline narration, ALAC 48 kHz stereo intermediates and final mix",
              "end_screen_seconds": END_SCREEN_SECONDS, "ending_hold_seconds": ENDING_HOLD_SECONDS}
    owned_recovery = recovery is None
    speaker = None
    try:
        allowed = validate_script_scope(script, spec, bundle)
        episodes = _episode_seasons(bundle)
        scoped_assets = {**assets, "seasons": {str(season): assets.get("seasons", {}).get(str(season), [])
                                             for season in sorted(allowed)},
                         "sources": [source for source in assets.get("sources", [])
                                     if source.get("season") in allowed]}
        scoped_assets, sharp_warnings = prefer_sharp_sources(scoped_assets)
        width, height, warnings = choose_resolution(scoped_assets, bool(_value(spec, "try_4k", False)))
        warnings = [*sharp_warnings, *warnings]
        report["warnings"].extend([*script.get("warnings", []), *assets.get("warnings", []), *warnings])
        report["resolution"] = {"width": width, "height": height, "ceiling": 2160 if _value(spec, "try_4k", False) else 1080}
        genres = " ".join(map(str, bundle.get("show", {}).get("genres") or []))
        theme = theme_for(genres + " " + str(script.get("show_title") or ""), _value(spec, "brief", ""))
        report["music_theme"] = theme
        model = str(_value(spec, "tts_model", "original")).casefold()
        if model not in {"original", "turbo", "nano"}:
            raise PipelineError("TTS_MODEL must be original, turbo or nano.")
        with _environment({"TTS_MODEL": model, "VIDEO_RESOLUTION": f"{width}x{height}", "VIDEO_GENRE": theme}):
            if recovery is None:
                recovery, speaker, fingerprint = _make_recovery(spec, work)
                report["speech"]["reference_sha256"] = fingerprint
            usage = UsageTracker(int(_value(spec, "upcoming_season")), allowed_seasons=set(allowed),
                                 episode_seasons=episodes)
            analyses = analyse_assets(scoped_assets, work / "source-analysis", usage)
            # Picture-level matching and repeat detection across every clip.
            from .visual_match import VisualIndex
            usage.visual = VisualIndex.build(usage.shots)
            report["visual_matching"] = ("off" if usage.visual is None else
                                         f"{len(usage.visual.vectors)} scenes matched by picture, "
                                         f"{len(usage.visual.hashes)} fingerprinted")
            gemini = shared_client()
            voice_speed = voice_speed_value(_value(spec, "voice_speed", "1.0"))
            report["voice_speed"] = voice_speed
            if analyses:
                scoped_assets = {**scoped_assets, "_analyses": analyses}
            parts, timeline, cues, chapters, clock = [], [], [], [], 0.0
            for section_index, section in enumerate(script["sections"]):
                paragraphs = section.get("paragraphs") or []
                if not paragraphs:
                    continue
                show_title = script.get("show_title") or _value(spec, "show_title")
                chapters.append({"start": clock, "title": _chapter_title(section, show_title),
                                 "section_id": section["id"]})
                label = work / f"chapter-{section_index:03}.png"
                _label(label, section, show_title)
                pool = scoped_assets
                native_inserted = False
                for paragraph_index, text in enumerate(paragraphs):
                    if not isinstance(text, str) or not text.strip():
                        raise PipelineError("Recap narration must contain complete nonempty paragraphs.")
                    deliveries = section.get("delivery") or []
                    delivery = deliveries[paragraph_index] if paragraph_index < len(deliveries) else "normal"
                    delivery = delivery if delivery in DELIVERIES else "normal"
                    if speaker is not None:
                        speaker.delivery = delivery
                    label_text = f"{section['id']} paragraph {paragraph_index + 1}"
                    audio, aligned = recovery.recover(text, label=label_text)
                    raw_seconds = duration(audio)
                    pace = delivery_speed(narration_rate(theme), delivery,
                                          words=len(text.split()), seconds=raw_seconds)
                    # The viewer-chosen voice-over speed multiplies the natural delivery pace.
                    pace = round(min(1.6, max(0.75, pace * voice_speed)), 3)
                    audio = voice.paced_take(Path(audio), pace)
                    aligned = [{**cue, "start": cue["start"] / pace, "end": cue["end"] / pace} for cue in aligned]
                    # Save a unique complete verified recording for the final review.
                    retained = work / "verified" / f"{section_index:03}-{paragraph_index:03}.wav"
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    if Path(audio).resolve() != retained.resolve():
                        shutil.copyfile(audio, retained)
                    current_episodes = _paragraph_episodes(section, paragraph_index, episodes)
                    paragraph_events = _paragraph_events(section, paragraph_index)
                    chosen, cut_plan = _sentence_shots(
                        gemini, section, text, aligned, duration(retained), pool, usage, current_episodes,
                        paragraph_events, f"{show_title}: {_chapter_title(section, show_title)}")
                    if not chosen:
                        raise PipelineError("No scoped visual plan covers the complete recap paragraph.")
                    part_work = work / "parts" / f"{section_index:03}-{paragraph_index:03}"
                    part_work.mkdir(parents=True, exist_ok=True)
                    part = part_work / "narration.mp4"
                    _validate_timeline([{"type": "narration", "section_id": section["id"],
                                         "episode_ids": current_episodes, "shots": chosen}], script, spec, bundle)
                    render.speech_video(Path(chosen[0]["source"]), retained, chosen, part,
                                        label if paragraph_index == 0 else None, part_work / "render")
                    seconds = _part_seconds(part)
                    event = {"type": "narration", "section_kind": section["kind"], "section_id": section["id"],
                             "cuts": cut_plan,
                             "start": clock, "duration": seconds, "text": text, "source_audio": str(retained.resolve()),
                             "delivery": delivery, "pace": pace, "original_speech_seconds": raw_seconds,
                             "season": section.get("season"), "episode_ids": current_episodes, "shots": chosen}
                    timeline.append(event)
                    cues.extend({**cue, "start": clock + cue["start"], "end": clock + cue["end"]}
                                for cue in script_cues(text, aligned))
                    parts.append(part)
                    clock += seconds
                    # Native audio is a discretionary complete beat; never a quota.
                    if not native_inserted and len(current_episodes) == 1 and section.get("kind") in {"story", "ending"}:
                        recognizer = getattr(recovery, "recognizer", None)
                        analyser = (lambda source: voice.captions(Path(source), recognizer)) if recognizer is not None else None
                        beat = plan_source_audio(section, text, pool, usage,
                                                 episode_ids=current_episodes, analyser=analyser,
                                                 events=paragraph_events)
                        if beat:
                            pending = {"type": "trailer-audio", "section_kind": section["kind"],
                                "section_id": section["id"], "start": clock, "duration": beat["duration"],
                                "season": section["season"], "episode_ids": current_episodes,
                                "text": beat.get("text", ""), "shots": [beat]}
                            # Optional media failure may skip a beat; wrong scope must fail closed.
                            _validate_timeline([pending], script, spec, bundle)
                            native = part_work / "source-audio.mp4"
                            replacement = None
                            try:
                                from movie_pipeline.dialogue import isolate_dialogue
                                replacement = isolate_dialogue(Path(beat["source"]), beat["start"], beat["duration"],
                                                               part_work / "dialogue")
                            except (OSError, ValueError, PipelineError):
                                report["warnings"].append("Optional source-dialogue separation was unavailable; the released source audio was retained.")
                            try:
                                render.trailer_video(Path(beat["source"]), beat["start"], beat["duration"], native, audio=replacement)
                                render.validate_audio_video_duration(native)
                                beat_seconds = _part_seconds(native)
                            except (OSError, ValueError, PipelineError) as exc:
                                native.unlink(missing_ok=True)
                                report["warnings"].append(f"Optional source-audio beat was skipped after a media failure ({type(exc).__name__}).")
                            else:
                                timeline.append({**pending, "duration": beat_seconds, "separated": replacement is not None})
                                if beat.get("text"):
                                    cues.append({"start": clock, "end": clock + min(beat_seconds, beat["duration"]), "text": beat["text"]})
                                parts.append(native)
                                clock += beat_seconds
                                native_inserted = True
            if not any(event["type"] == "narration" for event in timeline):
                raise PipelineError("A recap cannot export without complete verified narration.")
            end_plate = work / "end-screen.png"
            _end_screen(end_plate, script.get("show_title") or _value(spec, "show_title"))
            end_part = work / "end-screen.mp4"
            render.end_screen_video(end_plate, END_SCREEN_SECONDS, end_part)
            chapters.append({"start": clock, "title": "Thanks for watching", "section_id": "end-screen"})
            timeline.append({"type": "end-screen", "start": clock, "duration": _part_seconds(end_part), "shots": []})
            parts.append(end_part)
            _validate_timeline(timeline, script, spec, bundle)
            if hasattr(recovery, "release_voice"):
                recovery.release_voice()
            staged = work / "verified-export.mp4"
            minutes = _value(spec, "season_minutes" if script["kind"] == "season" else "series_minutes", "Auto")
            goal = None if str(minutes).casefold() == "auto" else float(minutes)*60
            timing = (DurationSpec(automatic=True) if goal is None else
                      DurationSpec(target=goal, minimum=goal*.75, maximum=goal*1.25, automatic=False))
            total, speed = render.finish(parts, staged, work / "finish", theme in {"action", "adventure"},
                timing, ending_hold=ENDING_HOLD_SECONDS, music_theme=theme, timeline=timeline)
            _strict_decode(staged)
            review = audio_review.verify_export_audio(staged, timeline, speed, recovery, work)
            if not review.get("accepted"):
                raise PipelineError("The final recap mix did not preserve complete verified narration.")
            report["audio_review"] = review
            if review.get("warning"):
                report["warnings"].append(review["warning"])
            actual_chapters = [{**chapter, "start": chapter["start"] / speed} for chapter in chapters]
            render.embed_chapters(staged, actual_chapters, total, work)
            _strict_decode(staged)
            render.validate_audio_video_duration(staged, total)
            write_srt(output / "subtitles.srt", [{**cue, "start": cue["start"] / speed, "end": cue["end"] / speed}
                                                for cue in cues])
            thumbnails = build_thumbnails(script, bundle, scoped_assets, output)
            if len(thumbnails) != 3 or any(not Path(path).is_file() for path in thumbnails):
                raise PipelineError("A recap package requires three completed thumbnails.")
            credits = _credits(bundle, scoped_assets, script, theme)
            yt_chapters = _description_chapters(actual_chapters, total)
            chapter_text = "\n".join(f"{_youtube_time(chapter['start'])} {chapter['title']}" for chapter in yt_chapters)
            opening = [p for section in script["sections"] for p in section["paragraphs"]][:2]
            description = str(script.get("description") or "") + "\n\n" + "\n\n".join(opening)
            description += "\n\nChapters\n" + chapter_text
            description += "\n\nResearch and visual sources\n" + credits
            show_tag = re.sub(r"[^A-Za-z0-9]", "", str(script.get("show_title") or ""))[:70] or "TVRecap"
            scope_tag = f"Season{int(_value(spec,'upcoming_season'))-1}Recap" if script["kind"]=="season" else "SeriesRecap"
            description += f"\n\n#{show_tag} #{scope_tag} #PreviouslyOn"
            report["warnings"].extend(usage.warnings)
            report.update(status="passed", duration=total, final_speed=speed, chapters=actual_chapters,
                          youtube_chapters=yt_chapters, youtube_chapters_eligible=len(yt_chapters)>=3,
                          target_seconds=goal, inside_soft_range=goal is None or goal*.75<=total<=goal*1.25,
                          research_quality=bundle.get("research_quality",{}), coverage=script.get("coverage",{}),
                          full_media_decode="passed", asset_analysis_sources=len(analyses),
                          thumbnails=[Path(path).name for path in thumbnails])
            report["warnings"] = list(dict.fromkeys(report["warnings"]))
            (output / "recommended-title.txt").write_text(str(script["title"]).strip() + "\n", encoding="utf-8")
            (output / "description.txt").write_text(description.strip() + "\n", encoding="utf-8")
            (output / "source-credits.txt").write_text(credits, encoding="utf-8")
            save_json(output / "script.json", script)
            save_json(output / "timeline.json", {"events": timeline, "speed": speed, "chapters": actual_chapters,
                                               "ending_hold_seconds": ENDING_HOLD_SECONDS})
            save_json(output / "production-report.json", report)
            shutil.copyfile(staged, final)
            return report
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__, detail=str(exc)[-1000:])
        save_json(output / "production-report.json", report)
        final.unlink(missing_ok=True)
        raise
    finally:
        if owned_recovery and recovery is not None:
            recovery.close()
