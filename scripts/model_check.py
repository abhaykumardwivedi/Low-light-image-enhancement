"""M3 model check (run on Colab with a GPU): load each model, score it, save example pictures.

Sanity rule (PID Section 13, decision D11): SegFormer-B0 on clean Cityscapes val should reach about
65% mIoU or more at short side 512. If it does not, also try 1024 before suspecting a bug.

Usage:
    python scripts/model_check.py --dataset nightcity --split val --limit 50
    python scripts/model_check.py --dataset cityscapes --models segformer_b0 --short-side 512 1024

Outputs in results/m3_check/: one JSON per (dataset, model, size), summary.csv, and example grids
(none / gamma / clahe / gamma_clahe) for the first images.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform

from lowlight.config import load_config, resolve_path
from lowlight.datasets import build_dataset, write_image
from lowlight.enhance import enhance
from lowlight.evaluate import evaluate
from lowlight.metrics import image_scores
from lowlight.models import MODEL_IDS, load_segmenter
from lowlight.viz import comparison_grid

SANITY_THRESHOLD = 0.65
GRID_METHODS = ("none", "gamma", "clahe", "gamma_clahe")


def environment() -> dict:
    import torch
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": gpu}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="nightcity")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="images to use (default: all)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    parser.add_argument("--short-side", nargs="+", type=int, default=[512])
    parser.add_argument("--grids", type=int, default=3, help="example images to draw")
    args = parser.parse_args()

    cfg = load_config(overrides={"limit": args.limit})
    out_dir = resolve_path(cfg["paths"]["results_root"]) / "m3_check"
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    env = environment()
    print("Environment:", env)
    ds = build_dataset(cfg, args.dataset, args.split)
    print(f"{args.dataset}/{args.split}: {len(ds)} images")

    rows = []
    for name in args.models:
        for size in args.short_side:
            seg = load_segmenter(name, size, cfg["device"])
            result = evaluate(seg, ds, seed=cfg["seed"])
            s = result.summary
            row = {"dataset": args.dataset, "split": args.split, "model": name, "short_side": size,
                   "params_m": round(seg.num_params / 1e6, 2), "n_images": s["n_images"],
                   "miou": round(s["miou"], 4), "pixel_acc": round(s["pixel_acc"], 4),
                   "median_infer_ms": round(s["median_infer_ms"], 1)}  # fmt: skip
            rows.append(row)
            print(row)
            payload = {**row, "per_class_iou": s["per_class_iou"], "environment": env,
                       "image_ids": result.image_ids}  # fmt: skip
            path = out_dir / f"{args.dataset}_{args.split}_{name}_s{size}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            if args.dataset == "cityscapes" and name == "segformer_b0":
                ok = s["miou"] >= SANITY_THRESHOLD
                verdict = "PASS" if ok else "BELOW 0.65, check before running experiments"
                print(
                    f"SANITY CHECK (B0, Cityscapes val, short side {size}): "
                    f"mIoU {s['miou']:.3f} -> {verdict}"
                )

        # Example pictures with the first model size: how each enhancement changes the prediction.
        seg = load_segmenter(name, args.short_side[0], cfg["device"])
        for i in range(min(args.grids, len(ds))):
            img, gt, sample = ds[i]
            grid_rows = []
            for method in GRID_METHODS:
                enhanced = enhance(img, method)
                pred = seg.predict(enhanced)
                grid_rows.append((method, enhanced, pred, gt, image_scores(pred, gt)[1]["miou"]))
            write_image(
                out_dir / "grids" / f"{name}_{sample.image_id}.jpg", comparison_grid(grid_rows)
            )

    with (out_dir / "summary.csv").open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    print("Saved results to", out_dir)


if __name__ == "__main__":
    main()
