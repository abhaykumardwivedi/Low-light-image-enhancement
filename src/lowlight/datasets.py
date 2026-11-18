"""Dataset loaders for ACDC night, Cityscapes val, Dark Zurich val and plain image folders.

IMAGE RULE (PID 11.1): inside this package every image is an RGB uint8 NumPy array of shape
(H, W, 3), and every label mask is a uint8 (H, W) array of trainIds with 255 = ignore.
OpenCV reads and writes images in BGR order, so `read_image` converts BGR -> RGB immediately
on load, and `write_image` converts back only at the moment of saving.

Each loader first builds a sorted list of `Sample`s (image path + matching label path) without
reading any pixels, so listing and checking a dataset is fast. Pixels are read only when an item
is requested.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lowlight.config import resolve_path
from lowlight.labels import IGNORE_INDEX, NUM_CLASSES, labelid_to_trainid

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# Labelled image counts from the dataset papers (PID Section 8). check-data compares against these.
EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("acdc_night", "train"): 400,
    ("acdc_night", "val"): 106,
    ("cityscapes", "val"): 500,
    ("darkzurich", "val"): 50,
    # From the NightCity paper (Tan et al., 2021); confirmed on the first Colab run.
    ("nightcity", "train"): 2998,
    ("nightcity", "val"): 1299,
}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset folder does not exist (usually: not downloaded or unzipped yet)."""


@dataclass(frozen=True)
class Sample:
    """One image and (optionally) its label file. No pixels are loaded yet.

    Attributes:
        image_id: short unique name, e.g. "GOPR0351_frame_000159" (file name minus suffix).
        image_path: path to the RGB image.
        label_path: path to the label mask, or None if there is no label (e.g. a demo folder).
        label_format: "trainid" (values are already 0-18/255) or "labelid" (Cityscapes raw ids,
            converted to trainIds when read).
    """

    image_id: str
    image_path: Path
    label_path: Path | None = None
    label_format: str = "trainid"


# --------------------------------------------------------------------------------------------
# Reading and writing files
# --------------------------------------------------------------------------------------------


def _imread(path: str | Path, flags: int) -> np.ndarray:
    """Read an image file with OpenCV, also for paths with non-English characters.

    cv2.imread cannot open such paths on Windows, so we read the raw bytes with NumPy and let
    OpenCV decode them from memory instead.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:  # OpenCV returns None instead of raising when the bytes are not an image
        raise ValueError(f"Could not decode image file: {path}")
    return img


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an image decoded by OpenCV (gray, BGR or BGRA; 8 or 16 bit) to RGB uint8."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)  # keep the top 8 bits of a 16-bit image
    elif img.dtype != np.uint8:
        raise ValueError(f"Unsupported image data type: {img.dtype}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # drop the transparency channel
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    return np.ascontiguousarray(img)


def read_image(path: str | Path) -> np.ndarray:
    """Read any image file and return it as RGB uint8 (H, W, 3)."""
    return to_rgb_uint8(_imread(path, cv2.IMREAD_UNCHANGED))


def write_image(path: str | Path, img_rgb: np.ndarray) -> None:
    """Save an RGB uint8 image. The file type comes from the extension (.png, .jpg, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Could not encode image as {path.suffix}")
    buf.tofile(path)  # works for non-English paths, unlike cv2.imwrite on Windows


def write_label(path: str | Path, mask: np.ndarray) -> None:
    """Save a single-channel uint8 label mask as PNG (lossless)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Could not encode label mask")
    buf.tofile(path)


def read_label(path: str | Path, label_format: str = "trainid") -> np.ndarray:
    """Read a label mask and return trainIds as uint8 (H, W), 255 = ignore.

    Args:
        path: label PNG.
        label_format: "trainid" if the file already holds 0-18/255, or "labelid" for Cityscapes
            `*_gtFine_labelIds.png` files, which are converted with the lookup table.
    """
    mask = _imread(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim != 2:
        # A 3-channel label is almost always the coloured preview (e.g. *_color.png), not ids.
        raise ValueError(f"Label must be a single-channel id mask, got shape {mask.shape}: {path}")
    if mask.dtype != np.uint8:
        raise ValueError(f"Label must be uint8, got {mask.dtype}: {path}")
    if label_format == "labelid":
        return labelid_to_trainid(mask)
    if label_format == "trainid":
        return mask
    raise ValueError(f"Unknown label_format {label_format!r}")


# --------------------------------------------------------------------------------------------
# Finding image/label pairs
# --------------------------------------------------------------------------------------------


def _find_pairs(
    image_dir: Path,
    label_dir: Path,
    image_suffix: str,
    label_suffix: str,
    label_format: str,
) -> list[Sample]:
    """Find `<image_dir>/<subfolder>/<id><image_suffix>` and the matching label path.

    The label is expected at `<label_dir>/<same subfolder>/<id><label_suffix>`. The label path is
    filled in even if that file does not exist, so `missing_labels` can report it.
    """
    if not image_dir.is_dir():
        raise DatasetNotFoundError(f"Image folder not found: {image_dir}")
    samples = []
    # One level of subfolders: ACDC/Dark Zurich use sequence folders, Cityscapes uses city folders.
    for img_path in sorted(image_dir.glob(f"*/*{image_suffix}")):
        image_id = img_path.name[: -len(image_suffix)]
        label_path = label_dir / img_path.parent.name / f"{image_id}{label_suffix}"
        samples.append(Sample(image_id, img_path, label_path, label_format))
    return samples


def find_acdc(root: str | Path, split: str, condition: str = "night") -> list[Sample]:
    """List ACDC samples, e.g. `rgb_anon/night/val/<seq>/<id>_rgb_anon.png`.

    Only "train" and "val" are allowed: ACDC test labels are not public (PID Section 8).
    """
    if split not in ("train", "val"):
        raise ValueError(f"ACDC split must be 'train' or 'val' (test has no labels), got {split!r}")
    root = Path(root)
    return _find_pairs(
        root / "rgb_anon" / condition / split,
        root / "gt" / condition / split,
        "_rgb_anon.png",
        "_gt_labelTrainIds.png",
        "trainid",
    )


def find_cityscapes(root: str | Path, split: str = "val") -> list[Sample]:
    """List Cityscapes samples: `leftImg8bit/val/<city>/<id>_leftImg8bit.png` + labelIds mask."""
    root = Path(root)
    return _find_pairs(
        root / "leftImg8bit" / split,
        root / "gtFine" / split,
        "_leftImg8bit.png",
        "_gtFine_labelIds.png",
        "labelid",
    )


def find_darkzurich(root: str | Path, split: str = "val") -> list[Sample]:
    """List Dark Zurich samples: `rgb_anon/val/night/<seq>/<id>_rgb_anon.png` + trainId mask."""
    root = Path(root)
    return _find_pairs(
        root / "rgb_anon" / split / "night",
        root / "gt" / split / "night",
        "_rgb_anon.png",
        "_gt_labelTrainIds.png",
        "trainid",
    )


def find_folder(image_dir: str | Path, label_dir: str | Path | None = None) -> list[Sample]:
    """List every image in a plain folder (not recursive), e.g. demo photos.

    If `label_dir` is given, a label named `<image stem>.png` (trainIds) is attached when it
    exists; images without a label get `label_path=None`.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise DatasetNotFoundError(f"Image folder not found: {image_dir}")
    samples = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            label = Path(label_dir) / f"{p.stem}.png" if label_dir is not None else None
            samples.append(Sample(p.stem, p, label if label and label.is_file() else None))
    return samples


# Name endings that are not part of the id shared by an image and its label.
_IMAGE_TAILS = ("_leftImg8bit", "_rgb_anon", "_image", "_img")
_LABEL_TAILS = (
    "_gtFine_labelTrainIds", "_gtFine_labelIds", "_gt_labelTrainIds", "_labelTrainIds",
    "_labelIds", "_trainIds", "_trainId", "_labelId", "_label", "_mask", "_gt",
)  # fmt: skip
_LABEL_DIR_PREFIXES = ("label", "gt", "annot", "mask")
_SKIP_NAME_WORDS = ("color", "colour", "instance", "polygon", "vis")


def _strip_tail(stem: str, tails: Sequence[str]) -> str:
    for tail in tails:
        if stem.lower().endswith(tail.lower()):
            return stem[: -len(tail)]
    return stem


def _is_label_dir(dirs: Sequence[str]) -> bool:
    """True if any folder name contains a word like 'label', 'gtFine' or 'annotations'."""
    tokens = (t for d in dirs for t in re.split(r"[^a-z0-9]+", d.lower()) if t)
    return any(t.startswith(_LABEL_DIR_PREFIXES) for t in tokens)


def detect_label_format(label_paths: Sequence[Path], max_files: int = 10) -> str:
    """Guess whether label files hold trainIds (0-18, 255) or Cityscapes labelIds (0-33).

    The file name decides when it says so ("trainIds" / "labelIds"). Otherwise a few files are
    read: Cityscapes labelIds almost always contain values above 18 (e.g. 21 vegetation,
    23 sky, 26 car), which never appear in trainId masks.
    """
    names = " ".join(p.name.lower() for p in label_paths[:max_files])
    if "trainid" in names:
        return "trainid"
    if "labelid" in names:
        return "labelid"
    for p in label_paths[:max_files]:
        values = np.unique(read_label(p, "trainid"))
        if ((values > 18) & (values < 255)).any():
            return "labelid"
    return "trainid"


def find_paired_tree(
    root: str | Path,
    split: str,
    split_aliases: Sequence[str] | None = None,
) -> list[Sample]:
    """Find images and labels anywhere under `root` when the exact layout is not known.

    Used for NightCity, whose zip layout is not documented. A file belongs to `split` if one of
    its folders is named like the split (or one of `split_aliases`). It is a label if a folder
    name contains "label"/"gt"/"annot"/"mask" or its name ends with a label suffix such as
    `_labelIds`; otherwise it is an image. Colour previews and instance masks are skipped.
    Images and labels are matched by their file name without those suffixes.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset folder not found: {root}")
    wanted = {s.lower() for s in (split_aliases or (split,))}
    images: dict[str, Path] = {}
    labels: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dirs = [d.lower() for d in p.relative_to(root).parts[:-1]]
        if p.name.startswith("._") or "__macosx" in dirs:  # hidden junk files from macOS zips
            continue
        if not wanted & set(dirs) or any(w in p.stem.lower() for w in _SKIP_NAME_WORDS):
            continue
        if _is_label_dir(dirs) or _strip_tail(p.stem, _LABEL_TAILS) != p.stem:
            labels.setdefault(_strip_tail(p.stem, _LABEL_TAILS), []).append(p)
        else:
            images[_strip_tail(p.stem, _IMAGE_TAILS)] = p

    # If both a trainIds and a labelIds file exist for one image, prefer the trainIds file.
    chosen = {
        key: next((c for c in cands if "trainid" in c.name.lower()), cands[0])
        for key, cands in labels.items()
    }
    fmt = detect_label_format(list(chosen.values())) if chosen else "trainid"
    samples = []
    for key in sorted(images):
        # A missing label gets a path that does not exist, so check-data reports it.
        label = chosen.get(key, root / "_missing_label" / f"{key}.png")
        samples.append(Sample(key, images[key], label, fmt))
    return samples


def find_nightcity(root: str | Path, split: str) -> list[Sample]:
    """List NightCity samples. Its 'val' set may be stored in a folder called 'test'."""
    if split not in ("train", "val"):
        raise ValueError(f"NightCity split must be 'train' or 'val', got {split!r}")
    aliases = ("train",) if split == "train" else ("val", "validation", "test")
    return find_paired_tree(root, split, aliases)


def missing_labels(samples: Sequence[Sample]) -> list[Sample]:
    """Return the samples whose label file should exist but does not."""
    return [s for s in samples if s.label_path is not None and not s.label_path.is_file()]


def select_subset(samples: Sequence[Sample], limit: int | None, seed: int) -> list[Sample]:
    """Pick a fixed random subset of `limit` samples (all samples if limit is None or too big).

    The same (samples, limit, seed) always gives the same subset, so a 20-image development run
    always uses the same 20 images. The subset keeps the original (sorted) order.
    """
    if limit is None or limit >= len(samples):
        return list(samples)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(samples), size=limit, replace=False))
    return [samples[i] for i in chosen]


# --------------------------------------------------------------------------------------------
# Dataset object and registry
# --------------------------------------------------------------------------------------------


@dataclass
class SegDataset:
    """A list of samples that loads pixels on demand.

    `dataset[i]` returns `(image_rgb, label_trainids_or_None, sample)`.
    """

    name: str
    split: str
    samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray | None, Sample]:
        s = self.samples[i]
        image = read_image(s.image_path)
        label = read_label(s.label_path, s.label_format) if s.label_path is not None else None
        return image, label, s

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray | None, Sample]]:
        for i in range(len(self)):
            yield self[i]

    @property
    def image_ids(self) -> list[str]:
        return [s.image_id for s in self.samples]


@dataclass(frozen=True)
class DatasetSpec:
    """How to find one dataset: which config key holds its root folder, and the finder."""

    root_key: str
    finder: Callable[[Path, str], list[Sample]]
    splits: tuple[str, ...]
    optional: bool = False  # optional datasets (stretch goals) may be absent without failing


DATASETS: dict[str, DatasetSpec] = {
    "acdc_night": DatasetSpec("acdc_root", find_acdc, ("train", "val")),
    "cityscapes": DatasetSpec("cityscapes_root", find_cityscapes, ("val",)),
    "darkzurich": DatasetSpec("darkzurich_root", find_darkzurich, ("val",), optional=True),
    # Real-night alternative that needs no account (decision D28).
    "nightcity": DatasetSpec("nightcity_root", find_nightcity, ("train", "val"), optional=True),
}


def dataset_root(cfg: dict[str, Any], name: str) -> Path:
    """Absolute root folder of a registered dataset, from the config."""
    return resolve_path(cfg["datasets"][DATASETS[name].root_key])


def build_dataset(cfg: dict[str, Any], name: str, split: str) -> SegDataset:
    """Create a dataset from the config, applying `limit` and `seed` for development subsets.

    Raises:
        KeyError: unknown dataset name.
        ValueError: split not allowed, or some labels are missing.
        DatasetNotFoundError: the dataset folder does not exist.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    if split not in spec.splits:
        raise ValueError(f"{name}: split must be one of {spec.splits}, got {split!r}")

    samples = spec.finder(dataset_root(cfg, name), split)
    missing = missing_labels(samples)
    if missing:
        examples = ", ".join(str(s.label_path) for s in missing[:3])
        raise ValueError(f"{name}/{split}: {len(missing)} labels missing, e.g. {examples}")

    samples = select_subset(samples, cfg.get("limit"), cfg["seed"])
    logger.info("Loaded %s/%s: %d images", name, split, len(samples))
    return SegDataset(name, split, samples)


# --------------------------------------------------------------------------------------------
# Checking datasets (used by `python -m lowlight.cli check-data`)
# --------------------------------------------------------------------------------------------


@dataclass
class DatasetCheck:
    """Result of checking one dataset split."""

    name: str
    split: str
    root: str
    optional: bool
    expected: int | None
    found: int = 0
    labels_missing: int = 0
    images_read: int = 0
    status: str = "ok"  # "ok", "missing" (folder not there) or "problem"
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if this check should not fail the whole data check."""
        return self.status == "ok" or (self.status == "missing" and self.optional)


def _check_pixels(sample: Sample) -> list[str]:
    """Read one image and its label; return a list of problems (empty if fine)."""
    try:
        image = read_image(sample.image_path)
        label = read_label(sample.label_path, sample.label_format) if sample.label_path else None
    except (OSError, ValueError) as e:
        return [f"{sample.image_id}: {e}"]
    problems = []
    if label is not None:
        if label.shape != image.shape[:2]:
            problems.append(f"{sample.image_id}: image {image.shape[:2]} vs label {label.shape}")
        bad = np.setdiff1d(np.unique(label), [*range(NUM_CLASSES), IGNORE_INDEX])
        if bad.size:
            problems.append(f"{sample.image_id}: unexpected label values {bad.tolist()}")
    return problems


def check_dataset(
    cfg: dict[str, Any],
    name: str,
    split: str,
    deep: bool = False,
    expected_counts: dict[tuple[str, str], int] | None = None,
) -> DatasetCheck:
    """Check one dataset split: folder exists, count, every image has a label, pixels readable.

    Args:
        deep: if True, read every image and label (slow; meant for Colab). If False, read only
            3 samples (first, middle, last) to catch format problems quickly.
        expected_counts: override EXPECTED_COUNTS (used by tests).
    """
    counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    spec = DATASETS[name]
    root = dataset_root(cfg, name)
    result = DatasetCheck(name, split, str(root), spec.optional, counts.get((name, split)))

    try:
        samples = spec.finder(root, split)
    except DatasetNotFoundError as e:
        result.status = "missing"
        result.problems.append(str(e))
        return result

    result.found = len(samples)
    if result.expected is not None and result.found != result.expected:
        result.problems.append(f"found {result.found} images, expected {result.expected}")

    missing = missing_labels(samples)
    result.labels_missing = len(missing)
    result.problems += [f"label missing: {s.label_path}" for s in missing[:5]]

    ids = [s.image_id for s in samples]
    if len(set(ids)) != len(ids):
        result.problems.append("duplicate image ids")

    missing_set = set(missing)
    readable = [s for s in samples if s not in missing_set]
    if not deep and readable:
        picks = {0, len(readable) // 2, len(readable) - 1}
        readable = [readable[i] for i in sorted(picks)]
    for s in readable:
        result.problems += _check_pixels(s)
    result.images_read = len(readable)

    if result.problems:
        result.status = "problem"
    return result


def check_datasets(
    cfg: dict[str, Any],
    names: Sequence[str] | None = None,
    deep: bool = False,
    expected_counts: dict[tuple[str, str], int] | None = None,
) -> list[DatasetCheck]:
    """Check every split of the given datasets (all registered datasets by default)."""
    results = []
    for name in names or DATASETS:
        for split in DATASETS[name].splits:
            results.append(check_dataset(cfg, name, split, deep, expected_counts))
    return results


def format_check_report(results: Sequence[DatasetCheck]) -> str:
    """Make a plain-text table of check results, followed by any problems."""
    lines = [f"{'dataset':<12}{'split':<7}{'found':>7}{'expected':>10}{'no label':>10}  status"]
    for r in results:
        expected = "-" if r.expected is None else str(r.expected)
        status = r.status + (" (optional)" if r.status == "missing" and r.optional else "")
        lines.append(
            f"{r.name:<12}{r.split:<7}{r.found:>7}{expected:>10}{r.labels_missing:>10}  {status}"
        )
    for r in results:
        if r.problems and r.passed:
            lines.append(f"\n{r.name}/{r.split}: not found (optional, fine to skip): {r.root}")
        elif r.problems:
            lines.append(f"\n{r.name}/{r.split} problems:")
            lines += [f"  - {p}" for p in r.problems[:10]]
            if len(r.problems) > 10:
                lines.append(f"  ... and {len(r.problems) - 10} more")
    overall = "ALL REQUIRED DATA OK" if all(r.passed for r in results) else "DATA CHECK FAILED"
    lines.append(f"\n{overall}")
    return "\n".join(lines)
