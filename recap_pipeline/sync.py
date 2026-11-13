"""Sentence-level edit timing: where each narrated sentence starts and where its cut lands.

The narration is recognised word by word after recording. Each sentence gets
its own shot, and when a key word is known the cut is delayed so the new shot
appears just as that word is spoken ("cut on the word").
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

CUT_LEAD = 0.3          # the new shot appears this long before its key word
MAX_CUT_DELAY = 0.6     # never delay a cut past this share of its sentence
MIN_SEGMENT = 2.0       # shorter segments merge into the previous one (no flicker)


def _norm(word):
    return re.sub(r"[^a-z0-9]", "", str(word).casefold())


def word_times(text, aligned, total):
    """(script word, start, end) for every narrated word.

    Recognised words supply timing; unmatched words are interpolated. Without
    usable recognition the words are spread evenly over the recording.
    """
    script = str(text).split()
    if not script:
        return []
    heard = []
    for cue in aligned or []:
        tokens = str(cue.get("text", "")).split()
        try:
            start, end = float(cue["start"]), float(cue["end"])
        except (KeyError, TypeError, ValueError):
            continue
        weight = sum(len(t) + 1 for t in tokens) or 1
        position = start
        for token in tokens:
            span = (end - start) * (len(token) + 1) / weight
            heard.append((token, position, position + span))
            position += span
    times = [None] * len(script)
    if heard:
        matcher = SequenceMatcher(None, [_norm(w) for w in script], [_norm(w) for w, _, _ in heard], autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                times[block.a + offset] = heard[block.b + offset][1:]
    if sum(t is not None for t in times) < max(1, len(script) // 2):
        step = total / len(script)
        return [(word, i * step, (i + 1) * step) for i, word in enumerate(script)]
    first, last = heard[0][1], heard[-1][2]
    index = 0
    while index < len(script):
        if times[index] is not None:
            index += 1
            continue
        gap = index
        while gap < len(script) and times[gap] is None:
            gap += 1
        low = times[index - 1][1] if index else first
        high = max(low, times[gap][0] if gap < len(script) else last)
        for k in range(gap - index):
            times[index + k] = (low + (high - low) * k / (gap - index), low + (high - low) * (k + 1) / (gap - index))
        index = gap
    return [(word, *times[i]) for i, word in enumerate(script)]


def sentences(text):
    """Split narration into sentences, keeping every word exactly once."""
    words = str(text).split()
    result, current = [], []
    for word in words:
        current.append(word)
        if re.search(r"[.!?]['\")\]”’]*$", word):
            result.append(current)
            current = []
    if current:
        result.append(current)
    return [" ".join(group) for group in result]


def segments(text, words, total, key_words=None):
    """One edit segment per sentence: {text, start, end, sentence}.

    ``key_words`` maps sentence index to the word whose action should be on screen
    as it is spoken. Segments tile 0..total exactly.
    """
    parts = sentences(text)
    if not parts or not words:
        return [{"text": str(text), "start": 0.0, "end": float(total), "sentence": 0}]
    key_words = key_words or {}
    cuts, cursor = [], 0
    for index, sentence in enumerate(parts):
        count = len(sentence.split())
        span = words[cursor:cursor + count] or words[-1:]
        start, end = span[0][1], span[-1][2]
        cut = start
        key = _norm(key_words.get(index, ""))
        if index and key:
            hit = next((w for w in span if _norm(w[0]) == key), None)
            if hit and hit[1] - CUT_LEAD > start:
                cut = min(hit[1] - CUT_LEAD, start + (end - start) * MAX_CUT_DELAY)
        cuts.append((0.0 if index == 0 else max(0.0, min(cut, total)), sentence, index))
        cursor += count
    result = []
    for position, (cut, sentence, index) in enumerate(cuts):
        stop = cuts[position + 1][0] if position + 1 < len(cuts) else float(total)
        if result and (stop - cut < MIN_SEGMENT or cut - result[-1]["start"] < MIN_SEGMENT):
            result[-1]["end"] = stop
            result[-1]["text"] += " " + sentence
            continue
        result.append({"text": sentence, "start": cut, "end": stop, "sentence": index})
    if len(result) > 1 and result[-1]["end"] - result[-1]["start"] < MIN_SEGMENT:
        last = result.pop()
        result[-1]["end"], result[-1]["text"] = last["end"], result[-1]["text"] + " " + last["text"]
    result[-1]["end"] = float(total)
    return [segment for segment in result if segment["end"] - segment["start"] > 0.05]
