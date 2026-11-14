"""Local picture checks for every candidate scene: what it shows, and whether we already showed it.

Each scene's middle frame gets
- a SigLIP 2 image embedding, so a narrated sentence ("throws Langston from a
  helicopter", "an MRI at a clinic") can be matched to the frames that show it,
  with or without Gemini's scene notes, and
- a perceptual fingerprint, so the same shot re-uploaded in another trailer or
  channel's clip is recognised as a repeat even though it comes from a new file.

Both are optional: without the model only fingerprints are used; without
OpenCV nothing changes.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

REPEAT_DISTANCE = 10      # of 64 fingerprint bits: closer means the same picture
BATCH = 16


def _key(source, start):
    from .gemini_media import scene_key
    return scene_key(source, start)


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


class VisualIndex:
    def __init__(self, model=None):
        self.model = model
        self.vectors = {}       # scene key -> embedding row
        self.hashes = {}        # scene key -> 64-bit fingerprint
        self._text = {}

    @classmethod
    def build(cls, shots_by_source, *, loader=None, frame_reader=None, hasher=None):
        """Index every scene of every clip. Returns None when frames cannot be read at all."""
        if os.getenv("RECAP_VISUAL_MATCH", "on").strip().lower() == "off":
            return None
        try:
            if frame_reader is None or hasher is None:
                from movie_pipeline.shots import get_frame, scene_hash
                frame_reader = frame_reader or (lambda path, second: get_frame(Path(path), second))
                hasher = hasher or scene_hash
        except ImportError:
            return None
        model = None
        if loader is not None:
            model = loader()
        elif os.getenv("RECAP_VISUAL_MODEL", "siglip2").strip().lower() != "off":
            try:
                import torch
                from movie_pipeline.shots import load_siglip
                model = load_siglip(torch)
            except Exception as exc:  # the model is an improvement, never a requirement
                print(f"Visual matching model unavailable ({type(exc).__name__}); using fingerprints only.", flush=True)
        index = cls(model)
        pending = []
        for source, shots in shots_by_source.items():
            for shot in shots:
                try:
                    image = frame_reader(source, (float(shot["start"]) + float(shot["end"])) / 2)
                except Exception:
                    continue
                key = _key(source, shot["start"])
                fingerprint = hasher(image)
                if fingerprint is not None:
                    index.hashes[key] = int(fingerprint)
                if model is not None:
                    pending.append((key, image))
                if len(pending) >= BATCH:
                    index._embed(pending)
                    pending = []
        index._embed(pending)
        print(f"Visual index: {len(index.hashes)} scene fingerprints, {len(index.vectors)} matched images.", flush=True)
        return index

    def _embed(self, pending):
        if not pending or self.model is None:
            return
        try:
            vectors = self.model.images([image for _, image in pending])
        except Exception as exc:
            print(f"Visual matching paused ({type(exc).__name__}).", flush=True)
            self.model = None
            return
        for (key, _), vector in zip(pending, vectors):
            self.vectors[key] = vector

    def similarity(self, text):
        """{scene key: standardised similarity to the sentence} (mean 0, unit spread)."""
        if self.model is None or not self.vectors or not str(text).strip():
            return {}
        if text in self._text:
            return self._text[text]
        try:
            query = self.model.text(str(text)[:300])
            keys = list(self.vectors)
            raw = {key: float((self.vectors[key] * query).sum()) for key in keys}
        except Exception:
            return {}
        values = list(raw.values())
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        result = {key: (value - mean) / spread for key, value in raw.items()}
        self._text[text] = result
        return result

    def repeats(self, key, used):
        """True when this scene's picture matches one already shown."""
        fingerprint = self.hashes.get(key)
        return fingerprint is not None and any(hamming(fingerprint, other) <= REPEAT_DISTANCE for other in used)
