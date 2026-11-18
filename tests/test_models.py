"""Tests for lowlight.models. Fast tests run anywhere; slow ones load real models (Colab)."""

import numpy as np
import pytest

from lowlight.labels import CLASSES
from lowlight.models import MODEL_IDS, DummySegmenter, resize_short_side


@pytest.mark.parametrize(
    ("shape", "short", "expected"),
    [((1024, 2048, 3), 512, (512, 1024, 3)), ((1080, 1920, 3), 512, (512, 910, 3)),
     ((300, 200, 3), 400, (600, 400, 3)), ((512, 1024, 3), 512, (512, 1024, 3))],
)  # fmt: skip
def test_resize_short_side_keeps_aspect(shape, short, expected):
    assert resize_short_side(np.zeros(shape, np.uint8), short).shape == expected


def test_dummy_segmenter_resizes_fixed_mask(scene):
    seg = DummySegmenter(np.array([[0, 13]], np.uint8))
    out = seg.predict(scene)
    assert out.shape == scene.shape[:2] and set(np.unique(out)) == {0, 13}
    assert DummySegmenter(lambda img: np.full(img.shape[:2], 2, np.uint8)).predict(scene)[0, 0] == 2


def test_model_ids():
    assert set(MODEL_IDS) == {"segformer_b0", "segformer_b2", "mask2former_swin_t"}


# ---------- slow: real models (run on Colab with `pytest -m slow`) ----------


def _street_like_image():
    """512 x 1024 image: gray 'road' below, blue 'sky' above, like a very simple street scene."""
    img = np.zeros((512, 1024, 3), np.uint8)
    img[:200] = (70, 130, 180)
    img[200:] = (90, 90, 90)
    return img


@pytest.mark.slow
@pytest.mark.parametrize("name", ["segformer_b0", "mask2former_swin_t"])
def test_real_model_output_format(name):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from lowlight.models import HFSegmenter

    seg = HFSegmenter(name, short_side=256, device="auto")
    np.testing.assert_array_equal(seg.remap, np.arange(len(CLASSES)))  # Cityscapes order
    img = _street_like_image()[:300, :700]  # odd size on purpose (tests Mask2Former padding)
    mask = seg.predict(img)
    assert mask.shape == img.shape[:2] and mask.dtype == np.uint8
    assert mask.max() < 19


@pytest.mark.slow
def test_mask2former_scores_match_official_postprocessing():
    """Our manual Mask2Former post-processing must agree with Hugging Face's own function."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from lowlight.models import HFSegmenter, mask2former_class_scores

    seg = HFSegmenter("mask2former_swin_t", short_side=256, device="cpu")
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(seg.model_id)
    x = seg._to_tensor(_street_like_image())  # 256 x 512: already a multiple of 32, no padding
    with torch.inference_mode():
        out = seg.model(pixel_values=x)
        ours = mask2former_class_scores(
            out.class_queries_logits, out.masks_queries_logits, x.shape[-2:], x.shape[-2:]
        ).argmax(1)[0]
        theirs = processor.post_process_semantic_segmentation(out, target_sizes=[x.shape[-2:]])[0]
    agreement = (ours.cpu() == theirs.cpu()).float().mean().item()
    assert agreement > 0.98, agreement
