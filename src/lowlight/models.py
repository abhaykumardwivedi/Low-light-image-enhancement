"""Segmentation model wrappers (PID Section 9.4). Inference only: models are never trained here.

Every model sits behind the same tiny interface:

    seg = load_segmenter("segformer_b0", short_side=512)
    mask = seg.predict(img_rgb_uint8)      # (H, W) uint8 trainIds, same size as the input

Inference protocol, identical for every method, model and image:
1. resize so the SHORTER side equals `short_side`, keeping the aspect ratio (no square squashing);
2. scale to 0-1 and normalise with the model's own mean and std (read from its processor config);
3. run the model under torch.inference_mode();
4. SegFormer: upsample the class scores (1/4 resolution) bilinearly to the ORIGINAL size, argmax.
   Mask2Former: combine its mask and class predictions into per-class scores, upsample, argmax;
5. map the model's class order to our trainIds (checked against labels.CLASSES at load time).

PyTorch and transformers are imported only when a real model is loaded, so the rest of the project
(and the tests) work on a laptop without them.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from lowlight.labels import NUM_CLASSES, check_id2label

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "segformer_b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "mask2former_swin_t": "facebook/mask2former-swin-tiny-cityscapes-semantic",
}


class Segmenter(Protocol):
    """Anything with a name, a class count and predict(image) -> trainId mask."""

    name: str
    num_classes: int

    def predict(self, img_rgb_uint8: np.ndarray) -> np.ndarray: ...


class DummySegmenter:
    """A fake model for tests: returns a fixed mask (resized to the input) or calls a function."""

    def __init__(self, mask: np.ndarray | Callable[[np.ndarray], np.ndarray], name: str = "dummy"):
        self.name = name
        self.num_classes = NUM_CLASSES
        self._mask = mask

    def predict(self, img_rgb_uint8: np.ndarray) -> np.ndarray:
        if callable(self._mask):
            return self._mask(img_rgb_uint8)
        h, w = img_rgb_uint8.shape[:2]
        return cv2.resize(self._mask, (w, h), interpolation=cv2.INTER_NEAREST)


def resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    """Resize so min(height, width) == short_side, keeping the aspect ratio.

    A 2048 x 1024 Cityscapes image with short_side 512 becomes 1024 x 512. INTER_AREA is used for
    shrinking (it averages pixels and avoids jagged edges), INTER_LINEAR for enlarging.
    """
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if scale == 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))  # OpenCV wants (width, height)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp)


def select_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if a GPU is available, else 'cpu'. Asking for 'cuda' without one fails."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda was requested but no GPU is available")
    return device


def mask2former_class_scores(class_logits, mask_logits, padded_hw, valid_hw):
    """Turn Mask2Former's outputs into per-class scores, shape (1, C, h, w).

    Mask2Former predicts Q "queries"; each has a class probability vector (with one extra
    "no object" class, dropped here) and a mask. The score for class c at a pixel is
    sum over queries of P(class c | query) * sigmoid(mask logit of query at that pixel). This is
    the same formula as the processor's post_process_semantic_segmentation, but we also crop away
    the padding we added, so non-square images line up exactly (decision D36).
    """
    import torch
    import torch.nn.functional as F

    masks = F.interpolate(mask_logits, size=padded_hw, mode="bilinear", align_corners=False)
    masks = masks[..., : valid_hw[0], : valid_hw[1]].sigmoid()
    probs = class_logits.softmax(dim=-1)[..., :-1]
    return torch.einsum("bqc,bqhw->bchw", probs, masks)


class HFSegmenter:
    """A Hugging Face SegFormer or Mask2Former model following the protocol above."""

    def __init__(self, name: str, short_side: int = 512, device: str = "auto"):
        import torch
        from transformers import AutoImageProcessor

        if name not in MODEL_IDS:
            raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODEL_IDS)}")
        self.name = name
        self.model_id = MODEL_IDS[name]
        self.short_side = short_side
        self.device = select_device(device)
        self.kind = "segformer" if name.startswith("segformer") else "mask2former"

        processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.mean = np.array(processor.image_mean, dtype=np.float32)
        self.std = np.array(processor.image_std, dtype=np.float32)
        do_rescale = getattr(processor, "do_rescale", True)
        self.rescale = float(getattr(processor, "rescale_factor", 1 / 255)) if do_rescale else 1.0

        if self.kind == "segformer":
            from transformers import SegformerForSemanticSegmentation as ModelClass
        else:
            from transformers import Mask2FormerForUniversalSegmentation as ModelClass
        self.model = ModelClass.from_pretrained(self.model_id).to(self.device).eval()

        # Model output index -> our trainId (identity if the model uses the Cityscapes order).
        self.remap = check_id2label(self.model.config.id2label, name)
        self.num_classes = NUM_CLASSES
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self._torch = torch
        logger.info(
            "Loaded %s (%s): %.1fM parameters, short side %d, device %s",
            name, self.model_id, self.num_params / 1e6, short_side, self.device,
        )  # fmt: skip

    def _to_tensor(self, img: np.ndarray):
        """Resize, rescale to 0-1, normalise with the model's mean/std -> 1 x 3 x h x w tensor."""
        x = resize_short_side(img, self.short_side).astype(np.float32) * self.rescale
        x = (x - self.mean) / self.std
        t = self._torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        return t.to(self.device)

    def class_scores(self, img: np.ndarray):
        """Per-class scores at the resized resolution, shape (1, C, h, w)."""
        import torch.nn.functional as F

        x = self._to_tensor(img)
        if self.kind == "segformer":
            return self.model(pixel_values=x).logits  # 1/4 of the resized size
        # Mask2Former's backbone needs sides that are multiples of 32: pad right/bottom with zeros
        # (zero = the mean colour after normalisation) and tell the model which pixels are real.
        h, w = x.shape[-2:]
        ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
        x = F.pad(x, (0, pw - w, 0, ph - h))
        pixel_mask = self._torch.zeros((1, ph, pw), dtype=self._torch.long, device=self.device)
        pixel_mask[:, :h, :w] = 1
        out = self.model(pixel_values=x, pixel_mask=pixel_mask)
        return mask2former_class_scores(
            out.class_queries_logits, out.masks_queries_logits, (ph, pw), (h, w)
        )

    def predict(self, img_rgb_uint8: np.ndarray) -> np.ndarray:
        """Return the (H, W) uint8 trainId mask at the ORIGINAL image size."""
        import torch.nn.functional as F

        h, w = img_rgb_uint8.shape[:2]
        with self._torch.inference_mode():
            scores = self.class_scores(img_rgb_uint8)
            scores = F.interpolate(scores, size=(h, w), mode="bilinear", align_corners=False)
            pred = scores.argmax(dim=1)[0].to("cpu").numpy()
        return self.remap[pred]


@functools.lru_cache(maxsize=4)
def load_segmenter(name: str, short_side: int = 512, device: str = "auto") -> HFSegmenter:
    """Load a model once and reuse it (the only global state allowed by PID 11.1)."""
    return HFSegmenter(name, short_side, device)


def segmenter_from_config(cfg: dict, name: str) -> HFSegmenter:
    """Load a model with the inference size and device from the config."""
    if cfg.get("sliding_window"):
        raise NotImplementedError("sliding_window is not implemented yet (decision D35)")
    return load_segmenter(name, cfg["inference_short_side"], cfg["device"])
