"""The 19 Cityscapes classes, their colours, and label conversions (PID Section 9.1).

Every dataset and model in this project is mapped to these 19 "trainIds" (0-18),
with 255 meaning "ignore this pixel".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

import numpy as np

logger = logging.getLogger(__name__)

IGNORE_INDEX = 255
NUM_CLASSES = 19

# Class names in trainId order: CLASSES[0] is trainId 0 ("road"), and so on.
CLASSES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Standard Cityscapes colours (RGB), same order as CLASSES.
PALETTE: np.ndarray = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = (0, 0, 0)

# Cityscapes labelId -> trainId. Every labelId not listed here becomes 255 (ignore).
LABELID_TO_TRAINID: dict[int, int] = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}  # fmt: skip


def _build_labelid_lut() -> np.ndarray:
    """Build a 256-entry lookup table: lut[labelId] = trainId (255 if not used)."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABELID_TO_TRAINID.items():
        lut[label_id] = train_id
    return lut


LABELID_LUT: np.ndarray = _build_labelid_lut()


def labelid_to_trainid(label_mask: np.ndarray) -> np.ndarray:
    """Convert a Cityscapes labelIds mask (uint8, H x W) to a trainIds mask.

    Uses the lookup table, so the whole image is converted in one indexing step:
    every pixel value v is replaced by LABELID_LUT[v].
    """
    if label_mask.dtype != np.uint8:
        raise TypeError(f"Expected a uint8 mask, got {label_mask.dtype}")
    return LABELID_LUT[label_mask]


# Pascal VOC class indices (used by torchvision's pretrained DeepLabV3, stretch goal).
VOC_CLASSES: tuple[str, ...] = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)  # fmt: skip

# VOC class index -> the Cityscapes trainIds it covers (PID Section 9.4).
# VOC "person" covers both Cityscapes "person" and "rider", so it maps to two trainIds.
# Only these 6 shared classes are scored in the separate "mapped-class mIoU" table.
VOC_TO_TRAINID: dict[int, tuple[int, ...]] = {
    VOC_CLASSES.index("person"): (CLASSES.index("person"), CLASSES.index("rider")),
    VOC_CLASSES.index("car"): (CLASSES.index("car"),),
    VOC_CLASSES.index("bus"): (CLASSES.index("bus"),),
    VOC_CLASSES.index("train"): (CLASSES.index("train"),),
    VOC_CLASSES.index("motorbike"): (CLASSES.index("motorcycle"),),
    VOC_CLASSES.index("bicycle"): (CLASSES.index("bicycle"),),
}


def normalise_name(name: str) -> str:
    """Lowercase a class name and turn underscores, hyphens and repeated spaces into one space.

    Example: "Traffic_Light" -> "traffic light".
    """
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


def check_id2label(id2label: Mapping[int | str, str], model_name: str = "model") -> np.ndarray:
    """Check a model's class list against CLASSES and return a remapping table.

    Hugging Face models store their classes as `config.id2label`, e.g. {0: "road", ...}.
    The returned array `remap` has one entry per model output index:
    remap[model_index] = our trainId. If the model already uses our exact order,
    remap is just [0, 1, ..., 18]; otherwise the difference is logged.

    Raises:
        ValueError: if the model has a class name we do not know, or misses one of ours.
    """
    # JSON configs store keys as strings ("0"), so convert them to int first.
    items = sorted((int(k), normalise_name(v)) for k, v in id2label.items())
    ours = {name: i for i, name in enumerate(CLASSES)}

    unknown = [name for _, name in items if name not in ours]
    if unknown:
        raise ValueError(f"{model_name}: unknown class names {unknown}")
    missing = sorted(set(CLASSES) - {name for _, name in items})
    if missing:
        raise ValueError(f"{model_name}: missing classes {missing}")

    remap = np.full(max(k for k, _ in items) + 1, IGNORE_INDEX, dtype=np.uint8)
    for k, name in items:
        remap[k] = ours[name]

    if not np.array_equal(remap, np.arange(NUM_CLASSES)):
        logger.warning("%s: class order differs from Cityscapes; remapping %s", model_name, remap)
    return remap
