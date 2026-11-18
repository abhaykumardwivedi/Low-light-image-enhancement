"""Tests for lowlight.labels."""

import numpy as np
import pytest

from lowlight.labels import (
    CLASSES,
    IGNORE_INDEX,
    LABELID_LUT,
    NUM_CLASSES,
    PALETTE,
    VOC_CLASSES,
    VOC_TO_TRAINID,
    check_id2label,
    labelid_to_trainid,
    normalise_name,
)


def test_class_list_and_palette_sizes():
    assert len(CLASSES) == NUM_CLASSES == 19
    assert PALETTE.shape == (19, 3)
    assert PALETTE.dtype == np.uint8
    assert tuple(PALETTE[CLASSES.index("car")]) == (0, 0, 142)


def test_lut_has_256_entries_and_19_used():
    assert LABELID_LUT.shape == (256,)
    used = LABELID_LUT[LABELID_LUT != IGNORE_INDEX]
    assert sorted(used.tolist()) == list(range(19))


def test_labelid_to_trainid_hand_made_example():
    # labelIds: 7 road, 26 car, 24 person, 0 unlabeled, 33 bicycle, 6 ground (not used)
    label_mask = np.array([[7, 26, 24], [0, 33, 6]], dtype=np.uint8)
    expected = np.array([[0, 13, 11], [255, 18, 255]], dtype=np.uint8)
    out = labelid_to_trainid(label_mask)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.uint8
    assert out.shape == label_mask.shape


def test_labelid_to_trainid_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        labelid_to_trainid(np.zeros((2, 2), dtype=np.int32))


def test_voc_mapping():
    assert VOC_TO_TRAINID[VOC_CLASSES.index("person")] == (11, 12)  # person + rider
    assert VOC_TO_TRAINID[VOC_CLASSES.index("motorbike")] == (17,)
    assert len(VOC_TO_TRAINID) == 6


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Traffic_Light", "traffic light"), ("traffic  sign", "traffic sign"), (" Road ", "road")],
)
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


def test_check_id2label_same_order_gives_identity():
    # Hugging Face config.json stores keys as strings and may use underscores.
    id2label = {str(i): name.replace(" ", "_") for i, name in enumerate(CLASSES)}
    remap = check_id2label(id2label)
    np.testing.assert_array_equal(remap, np.arange(19))


def test_check_id2label_builds_remap_for_different_order():
    swapped = list(CLASSES)
    swapped[0], swapped[1] = swapped[1], swapped[0]  # model index 0 = sidewalk
    remap = check_id2label(dict(enumerate(swapped)))
    assert remap[0] == 1 and remap[1] == 0
    assert remap[2] == 2


def test_check_id2label_rejects_unknown_or_missing():
    bad = dict(enumerate(CLASSES))
    bad[0] = "banana"
    with pytest.raises(ValueError, match="unknown"):
        check_id2label(bad)
    with pytest.raises(ValueError, match="missing"):
        check_id2label(dict(enumerate(CLASSES[:-1])))
