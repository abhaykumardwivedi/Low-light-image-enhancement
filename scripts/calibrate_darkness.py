"""Calibrate darkness levels L3/L4 against real night images and save a comparison figure.

Runs on Colab after the data setup notebook (needs the night dataset's TRAIN split and
Cityscapes val). Only brightness statistics are compared; no labels and no models are used.

Usage:
    python scripts/calibrate_darkness.py --night nightcity            # report only
    python scripts/calibrate_darkness.py --night nightcity --write    # also rewrite the YAML

Outputs (in results/calibration/): calibration.json, l_histograms.png and .pdf.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from lowlight.calibrate import calibrate_levels, image_stats, levels_to_yaml
from lowlight.config import DARKNESS_CONFIG, load_config, load_darkness_levels, resolve_path
from lowlight.darken import darken
from lowlight.datasets import build_dataset
from lowlight.features import l_histogram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--night", default="nightcity", choices=["nightcity", "acdc_night"])
    parser.add_argument("--n-night", type=int, default=100, help="night TRAIN images to use")
    parser.add_argument("--n-day", type=int, default=50, help="Cityscapes val images to use")
    parser.add_argument("--write", action="store_true", help="rewrite darkness_levels.yaml")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["results_root"]) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Night: the TRAIN split only (tuning data), never val.
    night = build_dataset({**cfg, "limit": args.n_night}, args.night, "train")
    night_images = [img for img, _, _ in night]
    night_stats = image_stats(night_images)
    print(
        f"Real night ({args.night} train, {len(night)} images): mean L = {night_stats.mean_l:.3f}"
    )

    day = build_dataset({**cfg, "limit": args.n_day}, "cityscapes", "val")
    day_images = [(s.image_id, img) for img, _, s in day]
    levels = load_darkness_levels()
    new_levels, report = calibrate_levels(day_images, night_stats.mean_l, levels, cfg["seed"])
    report.update(
        night_dataset=args.night, n_night=len(night), n_day=len(day), date=str(date.today())
    )

    for name in levels:
        b, a = report["before"][name]["mean_l"], report["after"][name]["mean_l"]
        print(
            f"{name}: mean L before {b:.3f} -> after {a:.3f}  (alpha {new_levels[name]['alpha']})"
        )
    print("L3/L4 bracket real night:", report["bracketed_after"])
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _save_histograms(night_images, day_images, new_levels, cfg["seed"], out_dir)

    if args.write:
        note = (
            f"Fitted on {report['date']} so L3/L4 bracket the mean lightness of {len(night)} "
            f"{args.night} train images ({night_stats.mean_l:.3f}); see results/calibration/."
        )
        Path(DARKNESS_CONFIG).write_text(levels_to_yaml(new_levels, note), encoding="utf-8")
        print("Rewrote", DARKNESS_CONFIG)


def _save_histograms(night_images, day_images, levels, seed, out_dir: Path) -> None:
    """Plot the L-channel histogram of real night images against each synthetic level."""
    import matplotlib

    matplotlib.use("Agg")  # no screen needed (Colab, servers)
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 256)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in levels:
        hist = np.mean(
            [
                l_histogram(darken(img, name, seed, image_id=i, levels=levels))
                for i, img in day_images
            ],
            axis=0,
        )
        ax.plot(x, hist, label=f"synthetic {name}", linewidth=1)
    ax.plot(
        x,
        np.mean([l_histogram(img) for img in night_images], axis=0),
        label="real night",
        color="black",
        linewidth=2.5,
    )
    ax.set_xlabel("lightness L (0 = black, 1 = white)")
    ax.set_ylabel("share of pixels")
    ax.set_yscale("log")
    ax.set_title("Lightness histograms: real night vs synthetic darkness levels")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"l_histograms.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
