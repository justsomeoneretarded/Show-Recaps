"""Picture-to-sentence matching and repeat detection, with a fake model (no downloads)."""
from recap_pipeline.editing import UsageTracker, plan_visuals
from recap_pipeline.visual_match import VisualIndex, hamming


class Vector(list):
    def __mul__(self, other):
        return Vector(a * b for a, b in zip(self, other))

    def sum(self):
        return sum(self)


class FakeModel:
    """Two-dimensional 'meaning': (helicopter, bar)."""
    WORDS = {"helicopter": Vector([1.0, 0.0]), "bar": Vector([0.0, 1.0])}

    def images(self, images):
        return [self.WORDS[image] for image in images]

    def text(self, text):
        return Vector([float("helicopter" in text), float("bar" in text)])


def _index(frames, hashes):
    shots = {"/a.mp4": [{"start": 0, "end": 8}, {"start": 10, "end": 18}],
             "/b.mp4": [{"start": 0, "end": 8}]}
    return VisualIndex.build(shots, loader=FakeModel, frame_reader=lambda path, second: frames[(path, second)],
                             hasher=lambda image: hashes[image])


def _clip(name, shots):
    return {"kind": "video", "path": name, "season": 2, "episode_ids": [], "scope": "season",
            "duration": 30.0, "width": 1920, "height": 1080, "shots": shots}


def test_sentence_lands_on_the_scene_whose_picture_matches():
    frames = {("/a.mp4", 4.0): "bar", ("/a.mp4", 14.0): "helicopter", ("/b.mp4", 4.0): "bar"}
    index = _index(frames, {"bar": 0b1111, "helicopter": 0xFFFF0000})
    usage = UsageTracker(3, episode_seasons={})
    usage.visual = index
    data = {"seasons": {"2": [_clip("/a.mp4", [{"start": 0, "end": 8}, {"start": 10, "end": 18}])]}}
    shot = plan_visuals({"id": "s", "kind": "story", "season": 2, "episode_ids": []},
                        "Reacher throws Langston from a helicopter.", 6, data, usage)[0]
    assert 10 <= shot["start"] < 18


def test_same_picture_from_another_upload_is_not_shown_twice():
    frames = {("/a.mp4", 4.0): "bar", ("/a.mp4", 14.0): "helicopter", ("/b.mp4", 4.0): "bar"}
    index = _index(frames, {"bar": 0b1111, "helicopter": 0xFFFF0000})
    usage = UsageTracker(3, episode_seasons={})
    usage.visual = index
    data = {"seasons": {"2": [_clip("/a.mp4", [{"start": 0, "end": 8}, {"start": 10, "end": 18}]),
                              _clip("/b.mp4", [{"start": 0, "end": 8}])]}}
    section = {"id": "s", "kind": "story", "season": 2, "episode_ids": []}
    first = plan_visuals(section, "They meet in a bar.", 6, data, usage)[0]
    second = plan_visuals(section, "Later, back in the bar.", 6, data, usage)[0]
    # Both "bar" scenes are the same picture: the second sentence gets the other scene.
    assert first["source"] in {"/a.mp4", "/b.mp4"} and first["start"] < 8
    assert second["source"] == "/a.mp4" and second["start"] >= 10


def test_index_without_a_model_still_fingerprints():
    shots = {"/a.mp4": [{"start": 0, "end": 8}]}
    index = VisualIndex.build(shots, loader=lambda: None, frame_reader=lambda p, s: "x", hasher=lambda image: 7)
    assert index.hashes and not index.vectors and index.similarity("anything") == {}
    assert hamming(0b1010, 0b0101) == 4
