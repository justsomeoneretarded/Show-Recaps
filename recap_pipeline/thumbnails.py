"""Finished recap thumbnails made only from verified, scoped local source assets."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat

from movie_pipeline.core import PipelineError, ffmpeg, normalized, save_json

SIZE = (1280, 720)
MAX_SOURCE_ASSETS = 12
ACCENTS = ("#F4C95D", "#65D6CB", "#F09F8C")


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise PipelineError(f"Invalid {label} for recap thumbnails.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Invalid {label} for recap thumbnails.") from exc
    if result < 1 or str(value) not in {str(result), f"{result}.0"}:
        raise PipelineError(f"Invalid {label} for recap thumbnails.")
    return result


def _scope(script: dict, bundle: dict) -> tuple[list[int], int, str]:
    incoming = _positive_int(script.get("upcoming_season"), "upcoming season")
    raw = script.get("seasons")
    if not isinstance(raw, list) or not raw:
        raise PipelineError("Recap thumbnails require the script's explicit season scope.")
    seasons = sorted({_positive_int(value, "season") for value in raw})
    available = {int(row["number"]) for row in bundle.get("seasons", [])
                 if isinstance(row, dict) and str(row.get("number", "")).isdigit()}
    if any(season >= incoming or season not in available for season in seasons):
        raise PipelineError("Recap thumbnail season scope includes an incoming or unresearched season.")
    if script.get("kind") not in {"season", "series"}:
        raise PipelineError("Recap thumbnails require kind 'season' or 'series'.")
    if script["kind"] == "season" and seasons != [incoming - 1]:
        raise PipelineError("The season recap thumbnail must cover the immediately previous season.")
    show = bundle.get("show") or {}
    title = str(show.get("title") or "").strip()
    if not title:
        raise PipelineError("No verified show title is available for recap thumbnails.")
    if script.get("show_title") and normalized(str(script["show_title"])) != normalized(title):
        raise PipelineError("The recap script and researched show do not match.")
    return seasons, incoming, title


def _read_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    if min(image.size) < 120:
        raise PipelineError("Thumbnail source image is too small to identify the show.")
    # Blank source frames should not displace usable episode stills.
    if max(ImageStat.Stat(image.resize((160, 90))).stddev) < 3:
        raise PipelineError("Thumbnail source image has no useful visible detail.")
    return image


def _source_images(script: dict, bundle: dict, assets: dict, work: Path) -> tuple[list[tuple], list[str]]:
    seasons, _incoming, _title = _scope(script, bundle)
    episode_seasons = {
        str(episode["id"]): int(row["number"])
        for row in bundle.get("seasons", []) for episode in row.get("episodes", [])
        if (episode.get("id") is not None and str(row.get("number", "")).isdigit()
            and episode.get("season") == int(row["number"]))
    }
    result, warnings, seen = [], [], set()
    tried = 0
    for season in seasons:
        for asset in (assets.get("seasons") or {}).get(str(season), []):
            if tried >= MAX_SOURCE_ASSETS:
                break
            if not isinstance(asset, dict) or asset.get("kind") not in {"image", "video"}:
                continue
            try:
                if _positive_int(asset.get("season"), "asset season") != season:
                    continue
            except PipelineError:
                continue
            ids = [str(value) for value in asset.get("episode_ids") or []]
            if any(episode_seasons.get(value) != season for value in ids):
                continue
            path = Path(str(asset.get("path") or ""))
            if not path.is_file() or str(path.resolve()) in seen:
                continue
            seen.add(str(path.resolve()))
            tried += 1
            try:
                if asset["kind"] == "image":
                    images = [(_read_image(path), None, False)]
                else:
                    length = float(asset.get("duration") or 0)
                    if not math.isfinite(length) or length <= 0:
                        continue
                    images = []
                    # Gemini's picked moments (sharp, expressive, no text or logo) come first.
                    picked = [float(t) for t in asset.get("thumbnail_times") or []
                              if isinstance(t, (int, float)) and 0 <= t < length][:2]
                    moments = picked or [length * fraction for fraction in (0.28, 0.62)]
                    for offset, second in enumerate(moments):
                        frame = work / f"source-{tried:02}-{offset}.jpg"
                        try:
                            ffmpeg("-ss", f"{max(0, min(length - 0.05, second)):.3f}",
                                   "-i", path, "-frames:v", "1", "-vf", "scale=1280:-2", frame)
                            images.append((_read_image(frame), round(second, 3), bool(picked)))
                        except (OSError, ValueError, PipelineError):
                            continue
                for image, second, chosen in images:
                    result.append((image, {
                        "gemini_pick": chosen,
                        "path": str(path), "source_url": str(asset.get("source_url") or ""),
                        "source_title": str(asset.get("title") or ""), "season": season,
                        "episode_ids": ids, "illustrative": bool(asset.get("illustrative")),
                        "frame_second": second, "scope": asset.get("scope", "season"),
                        "identity_evidence": asset.get("identity_evidence"),
                    }))
            except (OSError, ValueError, PipelineError):
                warnings.append(f"Could not read a verified thumbnail source for season {season}.")
    # A true episode image/frame takes priority over an illustrative show poster,
    # and Gemini-picked moments over fixed positions in a clip.
    result.sort(key=lambda row: (row[1]["illustrative"], not row[1].get("gemini_pick")))
    if any(not row[1]["illustrative"] for row in result):
        result = [row for row in result if not row[1]["illustrative"]]
    return result, warnings


def _lines(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    rows, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), trial, font=font)[2] > width:
            rows.append(current)
            current = word
        else:
            current = trial
    if current:
        rows.append(current)
    return rows


def _text(draw: ImageDraw.ImageDraw, text: str, xy: tuple, width: int, *,
          maximum: int, minimum: int = 28, lines: int = 2, fill="white") -> int:
    chosen = _font(minimum)
    rows = [text]
    for size in range(maximum, minimum - 1, -2):
        candidate = _font(size)
        wrapped = _lines(draw, text, candidate, width)
        if len(wrapped) <= lines and all(draw.textbbox((0, 0), row, font=candidate)[2] <= width for row in wrapped):
            chosen, rows = candidate, wrapped
            break
    else:
        # Long names remain complete and measured rather than silently truncated.
        for size in range(minimum - 1, 11, -1):
            candidate = _font(size)
            wrapped = _lines(draw, text, candidate, width)
            if len(wrapped) <= lines and all(draw.textbbox((0, 0), row, font=candidate)[2] <= width for row in wrapped):
                chosen, rows = candidate, wrapped
                break
        else:
            raise PipelineError("The show title is too long for a readable recap thumbnail.")
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=chosen)[3] + 11
    for row in rows:
        draw.text((x, y), row, font=chosen, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0))
        y += line_height
    return y


def _canvas(image: Image.Image, variant: int) -> Image.Image:
    # Keep portrait posters complete; the blurred background contains no invented imagery.
    if image.width / image.height < 1.35:
        background = ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(26))
        background = ImageEnhance.Brightness(background).enhance(0.43)
        subject = ImageOps.contain(image, (720, 670), method=Image.Resampling.LANCZOS)
        background.paste(subject, (SIZE[0] - subject.width - 42, (SIZE[1] - subject.height) // 2))
    else:
        background = ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS, centering=(0.58, 0.42))
        # Dark trailer frames read as murky at thumbnail size: lift them before the shade.
        light = ImageStat.Stat(background.convert("L")).mean[0]
        if light < 95:
            background = ImageEnhance.Brightness(background).enhance(min(1.6, 95 / max(light, 30)))
        background = ImageEnhance.Contrast(background).enhance(1.15)
        background = ImageEnhance.Color(background).enhance(1.12)
    canvas = background.convert("RGBA")
    shade = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(shade)
    if variant == 1:
        for y in range(SIZE[1]):
            alpha = int(220 * max(0, min(1, (y - 235) / 420)))
            draw.line((0, y, SIZE[0], y), fill=(4, 9, 16, alpha))
    else:
        # A smooth left shade leaves the genuine focal image readable on the right.
        for x in range(SIZE[0]):
            alpha = int(228 * max(0, 1 - x / 970))
            draw.line((x, 0, x, SIZE[1]), fill=(4, 9, 16, alpha))
    return Image.alpha_composite(canvas, shade)


def build_thumbnails(script: dict, bundle: dict, assets: dict, output: Path) -> list[Path]:
    """Create three 1280x720 JPEGs; fail explicitly when no scoped genuine image exists.

    The collector's provenance is retained in thumbnail-report.json. A research-verified
    season poster or neutral title card is labelled illustrative, never a season event.
    """
    seasons, upcoming, title = _scope(script, bundle)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recap-thumbnails-", dir=output) as directory:
        candidates, warnings = _source_images(script, bundle, assets, Path(directory))
        warnings = [str(message) for message in assets.get("warnings") or []] + warnings
        if not candidates:
            raise PipelineError("No readable, verified image or video frame exists within this recap's season scope; thumbnails cannot invent the show's identity.")
        coverage = f"Season {seasons[0]} Recap" if len(seasons) == 1 else f"Seasons {seasons[0]}-{seasons[-1]} Recap"
        before = f"Before Season {upcoming}"
        concepts = script.get("thumbnail_concepts") or [{}]
        hook = str(concepts[0].get("headline") or "") if isinstance(concepts[0], dict) else ""
        if hook.upper() in {"WHERE WE LEFT OFF", "THE STORY SO FAR", ""}:
            hook = ""
        results, report = [], []
        for variant in range(3):
            image, provenance = candidates[variant % len(candidates)]
            canvas = _canvas(image, variant)
            draw = ImageDraw.Draw(canvas)
            accent = ACCENTS[variant]
            draw.rectangle((18, 18, 1261, 701), outline=accent, width=4)
            if variant == 1:
                y = _text(draw, title.upper(), (58, 370), 1130, maximum=73, lines=2)
                y = _text(draw, before, (58, min(y + 8, 574)), 1120, maximum=57, lines=1, fill=accent)
                _text(draw, coverage, (61, 651), 1080, maximum=25, minimum=20, lines=1)
                headline, subline = before, coverage
            elif variant == 2:
                _text(draw, title.upper(), (58, 96), 840, maximum=68, lines=2)
                y = _text(draw, before, (58, 458), 1040, maximum=67, lines=1, fill=accent)
                _text(draw, coverage, (61, min(610, y + 15)), 1040, maximum=35, lines=1)
                headline, subline = before, coverage
            elif hook:
                # Gemini's story headline leads; the show and scope stay readable below it.
                y = _text(draw, hook.upper(), (58, 90), 760, maximum=84, lines=2, fill=accent)
                y = _text(draw, title.upper(), (58, max(300, y + 20)), 900, maximum=52, lines=2)
                _text(draw, f"{coverage} · {before}", (61, min(600, y + 14)), 1000, maximum=34, lines=1)
                headline, subline = hook.upper(), f"{coverage} · {before}"
            else:
                y = _text(draw, title.upper(), (58, 96), 700, maximum=77, lines=2)
                y = _text(draw, coverage, (58, max(320, y + 20)), 900, maximum=65, lines=1, fill=accent)
                _text(draw, before, (61, min(566, y + 18)), 900, maximum=36, lines=1)
                headline, subline = coverage, before
            if provenance["illustrative"]:
                draw.rounded_rectangle((48, 36, 702, 73), radius=7, fill=(0, 0, 0, 205))
                draw.text((61, 43), "ILLUSTRATIVE VISUAL - NOT AN EXACT SEASON SCENE",
                          font=_font(18), fill="white")
            target = output / f"thumbnail-{variant + 1:02}.jpg"
            canvas.convert("RGB").save(target, format="JPEG", quality=92, optimize=True, subsampling=0)
            if target.stat().st_size >= 2_000_000:
                canvas.convert("RGB").save(target, format="JPEG", quality=85, optimize=True)
            with Image.open(target) as completed:
                if completed.size != SIZE or completed.format != "JPEG":
                    raise PipelineError("Recap thumbnail failed its JPEG dimensions check.")
            results.append(target)
            report.append({"file": target.name, "show_title": title, "headline": headline,
                           "subline": subline, "scope": seasons, "upcoming_season": upcoming,
                           "width": SIZE[0], "height": SIZE[1], "source": provenance})
        save_json(output / "thumbnail-report.json", {"thumbnails": report, "warnings": warnings})
    return results
