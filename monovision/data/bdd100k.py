"""Build the night (A) / day (B) split in the ForkGAN/AU-GAN layout from a BDD100K images zip.

    <out>/{trainA,trainB,testA,testB}/*.jpg      A = night, B = day

Labels: BDD100K marks each image with attributes.timeofday ('night', 'daytime', 'dawn/dusk').
Pass --labels <bdd100k_labels_images_{train,val}.json ...> when you have them. The 10k images
zip ships without labels, so the fallback is a brightness heuristic on the mean luma of a small
thumbnail; it is bimodal on the 10k set, with dawn/dusk and ambiguous frames left out.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = {"train": "train", "val": "test"}  # BDD100K split -> our split name


def mean_luma(data, size=(160, 90)):
    im = Image.open(io.BytesIO(data))
    im.draft("L", size)
    return float(np.asarray(im.convert("L")).mean())


def classify_luma(luma, night_max, day_min):
    if luma < night_max:
        return "night"
    if luma >= day_min:
        return "day"
    return None


def load_timeofday(label_files):
    tod = {}
    for f in label_files:
        for item in json.load(open(f)):
            tod[Path(item["name"]).stem] = item.get("attributes", {}).get("timeofday")
    return tod


def classify_label(tod):
    return {"night": "night", "daytime": "day"}.get(tod)


def build(zip_path, out, labels=None, night_max=50.0, day_min=80.0, limit=None, extract=True):
    out = Path(out)
    tod = load_timeofday(labels) if labels else None
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            parts = Path(name).parts
            if not name.endswith(".jpg") or len(parts) < 2 or parts[-2] not in SPLITS:
                continue
            split, stem = SPLITS[parts[-2]], Path(name).stem
            data = z.read(name)
            if tod is not None:
                luma, kind = None, classify_label(tod.get(stem))
            else:
                luma = mean_luma(data)
                kind = classify_luma(luma, night_max, day_min)
            rows.append((split, stem, kind or "skip", "" if luma is None else f"{luma:.1f}"))
            if kind and extract:
                d = out / f"{split}{'A' if kind == 'night' else 'B'}"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{stem}.jpg").write_bytes(data)
            if limit and len(rows) >= limit:
                break
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default="raw_data/bdd100k_images_10k.zip")
    p.add_argument("--out", default="datasets/bdd100k")
    p.add_argument("--labels", nargs="*", help="BDD100K label JSON files (uses timeofday instead of brightness)")
    p.add_argument("--night-max", type=float, default=50.0, help="mean luma below this is night")
    p.add_argument("--day-min", type=float, default=80.0, help="mean luma at or above this is day")
    p.add_argument("--manifest", default="data_splits/bdd100k_10k_split.csv")
    p.add_argument("--dry-run", action="store_true", help="classify and write the manifest only")
    a = p.parse_args(argv)
    rows = build(a.zip, a.out, a.labels, a.night_max, a.day_min, extract=not a.dry_run)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "class", "mean_luma"])
        w.writerows(rows)
    counts = {}
    for split, _, kind, _ in rows:
        counts[(split, kind)] = counts.get((split, kind), 0) + 1
    for k in sorted(counts):
        print(*k, counts[k])


if __name__ == "__main__":
    main()
