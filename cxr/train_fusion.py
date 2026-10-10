"""Train only the fusion model on a capped NIH ChestX-ray subset.

The four agents stay frozen. This script downloads 400 train images and 80 test
images, asks Agent 1, Agent 2, and Agent 3 for probabilities, and fits one
logistic regression per finding.
"""

from __future__ import annotations

from cxr.console import configure_stdio

configure_stdio()

import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from cxr.db import init_db, insert_training_run
from cxr.fusion import feature_vector, fit_models, macro_f1
from cxr.labels import DATASET_ID, LABELS, NIH_NAMES, NIH_TO_LABEL, TEST_COUNT, TRAIN_COUNT
from cxr.paths import FUSION_PATH, NIH_SMALL, ensure_dirs
from cxr.pipeline import embed_image_agents

MANIFEST = NIH_SMALL / "manifest.csv"


def main() -> None:
    ensure_dirs()
    init_db()
    rows = ensure_subset()
    features, targets, splits = embed_rows(rows)
    train_mask = splits == "train"
    test_mask = splits == "test"
    payload = fit_models(features[train_mask], targets[train_mask])
    score = macro_f1(payload, features[test_mask], targets[test_mask])
    joblib.dump(payload, FUSION_PATH)
    note = (
        "Logistic regression on frozen Agent 1, Agent 2, and Agent 3. "
        "Agent 4 is not part of this training run."
    )
    insert_training_run(DATASET_ID, int(train_mask.sum()), int(test_mask.sum()), score, note)
    print(f"Saved {FUSION_PATH}")
    print(f"Test macro-F1: {score:.3f}")


def ensure_subset() -> list[dict]:
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) >= TRAIN_COUNT + TEST_COUNT:
            print(f"Using saved subset in {NIH_SMALL}")
            return rows
    print(f"Downloading a capped subset of {DATASET_ID}")
    from datasets import load_dataset

    train_rows = _take("train", TRAIN_COUNT, load_dataset)
    test_rows = _take("test", TEST_COUNT, load_dataset)
    rows = train_rows + test_rows
    _write_manifest(rows)
    return rows


def _take(split: str, count: int, load_dataset) -> list[dict]:
    dataset = load_dataset(DATASET_ID, split=split, streaming=True)
    names = list(NIH_NAMES)
    feature = getattr(dataset, "features", None)
    if feature and "labels" in feature:
        label_feature = feature["labels"]
        inner = getattr(label_feature, "feature", None)
        if inner is not None and getattr(inner, "names", None):
            names = list(inner.names)
    folder = NIH_SMALL / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, example in enumerate(dataset):
        if index >= count:
            break
        image: Image.Image = example["image"].convert("RGB")
        path = folder / f"{index:04d}.jpg"
        image.save(path, format="JPEG", quality=90)
        target = _target_from_example(example["labels"], names)
        row = {"split": split, "file": str(path.relative_to(NIH_SMALL))}
        row.update({label: str(value) for label, value in target.items()})
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"  saved {split} {index + 1}/{count}")
    if len(rows) < count:
        raise RuntimeError(f"{split} split only yielded {len(rows)} images, needed {count}.")
    return rows


def _target_from_example(raw_labels, names: list[str]) -> dict[str, int]:
    target = {label: 0 for label in LABELS}
    values = raw_labels if isinstance(raw_labels, (list, tuple)) else [raw_labels]
    for value in values:
        if isinstance(value, str):
            name = value
        else:
            index = int(value)
            if index < 0 or index >= len(names):
                continue
            name = names[index]
        label = NIH_TO_LABEL.get(name)
        if label:
            target[label] = 1
    return target


def _write_manifest(rows: list[dict]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "file", *LABELS])
        writer.writeheader()
        writer.writerows(rows)


def embed_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = NIH_SMALL / "features.npz"
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if len(stored["features"]) == len(rows):
            print(f"Using cached agent outputs in {cache.name}")
            return stored["features"], stored["targets"], stored["splits"]
    features = []
    targets = []
    splits = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        image = Image.open(NIH_SMALL / row["file"]).convert("RGB")
        votes = embed_image_agents(image)
        if all(all(value is None for value in agent.values()) for agent in votes.values()):
            raise RuntimeError("All three image agents failed. Check the error above and rerun.")
        features.append(feature_vector(votes))
        targets.append([int(row[label]) for label in LABELS])
        splits.append(row["split"])
        if index % 10 == 0 or index == total:
            print(f"  embedded {index}/{total}")
    feature_array = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64)
    split_array = np.asarray(splits)
    np.savez(cache, features=feature_array, targets=target_array, splits=split_array)
    return feature_array, target_array, split_array


if __name__ == "__main__":
    main()
