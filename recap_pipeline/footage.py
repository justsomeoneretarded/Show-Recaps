"""Frame-accurate shot analysis for recap footage.

Trailers and clips cut every one to three seconds. Sampling every half second
(the older analyser) blurs those cuts, so a recap shot could start or end a
few frames into the neighbouring trailer shot: a visible flash frame. Here one
FFmpeg decode finds every cut and every fade to black, and keeps a small frame
from each shot, so the editor can

- start every recap shot exactly on a source cut and never end a few frames
  after one,
- read every single source shot for burned-in captions, subscribe screens,
  title and logo cards, and drop dark or empty frames, and
- group consecutive clean shots into "runs" that can be cut as one sequence
  (a trailer's own editing), never across a removed shot or a fade.

Gemini's scene notes (descriptions, picture quality, story-event links) are
attached to the precise shots they overlap. Everything is optional: callers fall
back to the older analysis when FFmpeg or OCR is unavailable.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CUT_SCORE = 0.3          # FFmpeg scene score that marks a hard cut
FRAME_RATE = 4           # frames kept per second for checks (one decode for everything)
MIN_SHOT = 0.3           # shorter fragments are flashes or dissolves, never used
FLASH = 0.4              # a recap shot never ends less than this after a source cut
EDGE = 0.06              # stay this far inside a cut (frame-accurate, plus one frame of margin)
DARK_MEAN = 18.0         # 0-255 luma: fades, black title gaps
BRIGHT_MEAN = 232.0      # white flashes and white cards
FLAT_STD = 14.0          # almost uniform frame: a plain card
VERSION = 1


def _parse(log: str):
    cuts = sorted({float(t) for t in re.findall(r"pts_time:([0-9.]+)", log)})
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)", log)]
    return cuts, blacks


def scan(path: Path, frames_dir: Path, *, runner=subprocess.run):
    """One decode: (cut times, black intervals, {time: frame path}) at FRAME_RATE."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    graph = (f"[0:v]split=2[a][b];[a]scale=320:-2,blackdetect=d=0.08:pic_th=0.92:pix_th=0.08,"
             f"select='gt(scene\\,{CUT_SCORE})',showinfo[c];[b]fps={FRAME_RATE},scale=960:-2[d]")
    command = ["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(path), "-an", "-filter_complex", graph,
               "-map", "[c]", "-f", "null", "-", "-map", "[d]", "-q:v", "4", str(frames_dir / "%05d.jpg")]
    result = runner(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError("cut detection failed")
    cuts, blacks = _parse(result.stderr)
    frames = {round((int(p.stem) - 1) / FRAME_RATE, 3): p for p in sorted(frames_dir.glob("*.jpg"))}
    if not frames:
        raise RuntimeError("no frames decoded")
    return cuts, blacks, frames


def _luma(image):
    import numpy as np
    gray = np.asarray(image.convert("L").resize((160, 90)), dtype=float)
    return gray, float(gray.mean()), float(gray.std())


def _intervals(length, cuts, blacks, start, end):
    """Shots between consecutive cuts, with fades to black cut out of them."""
    points = [start] + [c for c in cuts if start + MIN_SHOT < c < end - MIN_SHOT] + [end]
    shots = []
    for low, high in zip(points, points[1:]):
        pieces = [(low, high)]
        for b_low, b_high in blacks:
            next_pieces = []
            for a, b in pieces:
                if b_high <= a or b_low >= b:
                    next_pieces.append((a, b))
                    continue
                if b_low - a >= MIN_SHOT:
                    next_pieces.append((a, b_low))
                if b - b_high >= MIN_SHOT:
                    next_pieces.append((b_high, b))
            pieces = next_pieces
        shots.extend({"start": round(a, 3), "end": round(b, 3), "cut_in": a == low}
                     for a, b in pieces if b - a >= MIN_SHOT)
    return shots


# The broadcaster's own corner logo is normal on official footage.
CORNER_LOGOS = ("prime", "video", "amazon", "netflix", "hbo", "max", "hulu", "disney", "apple",
                "peacock", "paramount", "showtime", "starz", "amc", "fx", "original", "originals")
MIN_CONFIDENCE = 75


def ocr_words(image) -> list[str]:
    """Words Tesseract reads with confidence; film grain and texture read as noise are ignored."""
    command = shutil.which("tesseract")
    if not command:
        return []
    import io
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    try:
        result = subprocess.run([command, "stdin", "stdout", "--psm", "11", "tsv"], input=buffer.getvalue(),
                                capture_output=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    words = []
    for line in result.stdout.decode("utf-8", errors="ignore").splitlines()[1:]:
        cells = line.split("\t")
        if len(cells) < 12:
            continue
        try:
            confidence = float(cells[10])
        except ValueError:
            continue
        word = cells[11].strip()
        if confidence >= MIN_CONFIDENCE and re.search(r"[A-Za-z0-9]{2,}", word):
            words.append(word)
    return words


def junk_text(words, *, flat: bool = False) -> bool:
    """On-screen writing that must never reach a recap: promos, cards, captions, channel marks."""
    from movie_pipeline.shots import CARD_TEXT, PROMO_TEXT
    from .editing import NETWORK_CARD, RELEASE_DATE, SOUND_CAPTION
    if isinstance(words, str):
        words = words.split()
    text = " ".join(words)
    letters = [re.sub(r"[^A-Za-z]", "", word) for word in words]
    foreign = [word for word in letters if len(word) >= 3
               and not any(word.casefold() in logo or logo in word.casefold() for logo in CORNER_LOGOS)]
    network_card = bool(NETWORK_CARD.search(text)) and ("credit" in text.casefold() or flat)
    return bool(PROMO_TEXT.search(text) or CARD_TEXT.search(text) or RELEASE_DATE.search(text)
                or SOUND_CAPTION.search(text) or network_card
                # Any readable word other than the network's own logo: a title, caption,
                # credit or another channel's watermark. A plain frame with any word is a card.
                or any(len(word) >= 4 for word in foreign) or len(foreign) >= 2
                or (flat and any(len(word) >= 2 for word in letters)))


TEXT_STEP = 3   # text is read on every third frame (0.75 s at 4 fps), so a card dissolved into a shot is seen


def _check(frame_paths, reader, junk=None):
    """(clean, reason, middle image path, motion) for one shot from its kept frames.

    When ``junk`` is a list, the indices of frames carrying on-screen text are added to it
    (every sampled frame is read), so the caller can cut a title card out of a shot whose
    edges no cut detector found (a dissolve into and out of the card).
    """
    from PIL import Image
    import numpy as np
    grays, means = [], []
    found_text = False
    last = len(frame_paths) - 1
    for index, frame in enumerate(frame_paths):
        with Image.open(frame) as image:
            image.load()
            gray, mean, std = _luma(image)
            grays.append(gray)
            means.append(mean)
            # Night scenes stay; a frame with almost nothing lit is a fade or a black card.
            flat = std < FLAT_STD or float((gray > 40).mean()) < 0.08
            if index % TEXT_STEP == 0 or index in (last, len(frame_paths) // 2):
                if junk_text(reader(image), flat=flat):
                    found_text = True
                    if junk is None:
                        return False, "text", None, 0.0
                    junk.append(index)
            if flat and mean >= DARK_MEAN and not found_text:
                return False, "flat", None, 0.0
    if found_text:
        return False, "text", None, 0.0
    if max(means) < DARK_MEAN or sorted(means)[len(means) // 2] < DARK_MEAN * 0.7:
        return False, "dark", None, 0.0
    if sorted(means)[len(means) // 2] > BRIGHT_MEAN:
        return False, "flash", None, 0.0
    motion = (float(np.mean([np.abs(a - b).mean() for a, b in zip(grays, grays[1:])]))
              if len(grays) > 1 else 0.0)
    return True, "", frame_paths[len(frame_paths) // 2], motion


def _ocr(image):
    return ocr_words(image)


def analyse(path: Path, length: float, work: Path, *, reader=None, scanner=None, promo_cutoff=None):
    """Precise, clean shots of one clip: [{start, end, run, motion, frame}].

    ``run`` numbers stretches of consecutive clean shots: footage inside one run
    plays as the source's own edit; a removed shot, fade or slate ends a run.
    """
    path, work = Path(path), Path(work)
    if not shutil.which("ffmpeg") and scanner is None:
        raise RuntimeError("ffmpeg is unavailable")
    reader = reader or _ocr
    # Several OCR processes run side by side; each must stay single-threaded.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    keep = work / (path.stem + ".frames")
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        cuts, blacks, frames = (scanner or scan)(path, Path(temporary))
        # Studio slates at the end (subscribe, social, release date) and the opening logo.
        end = length * 0.93
        if promo_cutoff is None:
            try:
                from movie_pipeline.shots import promo_cutoff as find_slate
                promo_cutoff = find_slate(path, length)
            except Exception:
                promo_cutoff = None
        if promo_cutoff is not None:
            end = min(end, max(promo_cutoff - 0.5, length * 0.3))
        start = min(1.0, length * 0.02)
        raw = _intervals(length, cuts, blacks, start, end)
        times = sorted(frames)

        def frames_of(shot):
            inside = [t for t in times if shot["start"] + EDGE <= t <= shot["end"] - EDGE]
            if not inside:
                middle = (shot["start"] + shot["end"]) / 2
                inside = [min(times, key=lambda t: abs(t - middle))]
            return inside

        def inspect(shot):
            inside = frames_of(shot)
            junk = []
            result = _check([frames[t] for t in inside], reader, junk)
            if result[0] or result[1] != "text" or not junk:
                return [(shot, result)]
            # Text on some frames only: a card dissolved into the shot. Cut it out with a
            # margin of one sampling step and keep what is left, as new (uncut) shots.
            margin = TEXT_STEP / FRAME_RATE
            spans = [(inside[i] - margin, inside[i] + margin) for i in junk]
            pieces, cursor = [], shot["start"]
            for low, high in sorted(spans):
                if low > cursor:
                    pieces.append((cursor, low))
                cursor = max(cursor, high)
            pieces.append((cursor, shot["end"]))
            kept = []
            for low, high in pieces:
                low, high = max(low, shot["start"]), min(high, shot["end"])
                if high - low < MIN_SHOT * 2:
                    continue
                part = {**shot, "start": round(low, 3), "end": round(high, 3),
                        "cut_in": shot["cut_in"] and low == shot["start"]}
                checked = _check([frames[t] for t in frames_of(part)], reader)
                kept.append((part, checked))
            return kept or [(shot, result)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            inspected = [item for items in pool.map(inspect, raw) for item in items]
        raw = [shot for shot, _ in inspected]
        checks = [check for _, check in inspected]
        keep.mkdir(parents=True, exist_ok=True)
        shots, run, previous_end, dropped = [], 0, None, {}
        for shot, (clean, reason, frame, motion) in zip(raw, checks):
            if not clean:
                dropped[reason] = dropped.get(reason, 0) + 1
                previous_end = None
                continue
            # Only a hard cut directly after the previous clean shot continues the run.
            if previous_end is None or shot["start"] - previous_end > 0.12 or not shot["cut_in"]:
                run += 1
            target = keep / f"{len(shots):04}.jpg"
            shutil.copyfile(frame, target)
            shots.append({"start": shot["start"], "end": shot["end"], "run": run,
                          "motion": round(motion, 2), "frame": str(target), "precise": True})
            previous_end = shot["end"]
    overlay = static_overlay([shot["frame"] for shot in shots])
    if overlay == "drop":
        dropped["overlay"] = dropped.get("overlay", 0) + len(shots)
        return [], dropped
    if overlay:
        for shot in shots:
            shot["crop_anchor"] = list(overlay)
    return shots, dropped


OVERLAY_SPAN = 0.08   # share of a band's width covered by fixed strokes: a burned-in title or credit
OVERLAY_BAND = 0.14   # top and bottom bands where channels put titles, credits and captions
CROP = 0.84           # the crop that removes an overlay band while keeping 16:9


def static_overlay(frame_paths):
    """Where a clip carries text that never changes while the picture does.

    Footage changes from shot to shot; a channel's burned-in title or "Credit:"
    line does not. In the per-pixel median of every shot's frame the picture
    blurs away and fixed strokes stay sharp, with almost no change over time.
    Returns None, "drop" (overlays at top and bottom), or the (x, y) anchor of
    a CROP that leaves the overlay band out.
    """
    import numpy as np
    from PIL import Image
    paths = list(frame_paths)
    if len(paths) < 6:
        return None
    paths = paths[::max(1, len(paths) // 40)]
    stack = np.stack([np.asarray(Image.open(path).convert("L").resize((480, 270)), dtype=np.float32)
                      for path in paths])
    median, spread = np.median(stack, axis=0), stack.std(axis=0)
    strokes = np.zeros_like(median)
    strokes[:, 1:] = np.abs(np.diff(median, axis=1))
    fixed = (spread < 14) & (strokes > 35)
    height, width = median.shape
    band = int(height * OVERLAY_BAND)
    found = {}
    for name, rows in (("top", fixed[:band]), ("bottom", fixed[height - band:])):
        columns = rows.sum(axis=0) >= 2
        thirds = [float(columns[i * width // 3:(i + 1) * width // 3].mean()) for i in range(3)]
        if max(thirds) > OVERLAY_SPAN:
            found[name] = thirds
    if not found:
        return None
    if len(found) == 2:
        return "drop"
    name, thirds = next(iter(found.items()))
    # Keep the side of the frame away from the overlay; a full-width title keeps the centre.
    x = 1.0 if thirds[0] > OVERLAY_SPAN and thirds[2] <= OVERLAY_SPAN else \
        0.0 if thirds[2] > OVERLAY_SPAN and thirds[0] <= OVERLAY_SPAN else 0.5
    return (x, 1.0 if name == "top" else 0.0)


def merge_notes(shots: list[dict], scenes: list[dict], source: str, *, logged: bool) -> list[dict]:
    """Attach Gemini's notes to the precise shots they cover.

    When Gemini logged the clip, it left out scenes it judged unusable (on-screen
    text, interviews, poor picture); a precise shot outside every kept scene is
    dropped too, and the run is split there.
    """
    if not logged:
        return shots
    result, run_shift, broken = [], 0, False
    for shot in shots:
        span = shot["end"] - shot["start"]
        best, overlap = None, 0.0
        for scene in scenes:
            # Gemini watches at one frame per second: allow half a second either side.
            low, high = float(scene["start"]) - 0.5, float(scene["end"]) + 0.5
            common = max(0.0, min(shot["end"], high) - max(shot["start"], low))
            if common > overlap:
                best, overlap = scene, common
        if best is None or overlap < 0.5 * span:
            broken = True
            continue
        if broken:
            run_shift += 1000  # a dropped shot splits the run
            broken = False
        notes = {name: best[name] for name in ("description", "quality", "watermark", "letterbox", "dark",
                                                "event_id", "episode_id", "event_summary", "characters") if name in best}
        from .gemini_media import scene_key
        result.append({**shot, **notes, "run": shot["run"] + run_shift,
                       "gemini_key": scene_key(source, best["start"])})
    return result
