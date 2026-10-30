import io
import json
import zipfile

import numpy as np
from PIL import Image

from monovision.data import bdd100k


def _jpg(value):
    buf = io.BytesIO()
    Image.fromarray(np.full((90, 160, 3), value, np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _zip(tmp_path):
    p = tmp_path / "b.zip"
    with zipfile.ZipFile(p, "w") as z:
        for name, v in [("train/n", 10), ("train/d", 120), ("train/m", 65), ("val/d", 130)]:
            z.writestr(f"10k/{name}.jpg", _jpg(v))
    return p


def test_brightness_split(tmp_path):
    out = tmp_path / "out"
    rows = bdd100k.build(_zip(tmp_path), out)
    assert {(s, i, k) for s, i, k, _ in rows} == {
        ("train", "n", "night"), ("train", "d", "day"), ("train", "m", "skip"), ("test", "d", "day")}
    assert (out / "trainA" / "n.jpg").exists() and (out / "trainB" / "d.jpg").exists()
    assert (out / "testB" / "d.jpg").exists() and not (out / "trainA" / "m.jpg").exists()


def test_label_split(tmp_path):
    lab = tmp_path / "l.json"
    lab.write_text(json.dumps([{"name": "n.jpg", "attributes": {"timeofday": "night"}},
                               {"name": "d.jpg", "attributes": {"timeofday": "daytime"}},
                               {"name": "m.jpg", "attributes": {"timeofday": "dawn/dusk"}}]))
    rows = bdd100k.build(_zip(tmp_path), tmp_path / "out", labels=[lab])
    assert {(s, i): k for s, i, k, _ in rows}[("train", "m")] == "skip"
    assert {(s, i): k for s, i, k, _ in rows}[("train", "n")] == "night"
