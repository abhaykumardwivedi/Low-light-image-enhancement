"""Command-line entry points: `python -m lowlight.cli <command>` (PID Section 9.11).

Commands so far: `check-data` (M1), `enhance` and `segment` (M3). `run`, `analyse` and
`report-assets` are added in later milestones.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from lowlight.config import load_config
from lowlight.datasets import DATASETS, check_datasets, format_check_report


def _cmd_check_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    results = check_datasets(cfg, names=args.datasets, deep=args.deep)
    print(format_check_report(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"deep": args.deep, "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report to {args.json}")
    return 0 if all(r.passed for r in results) else 1


def _parse_params(items: list[str] | None) -> dict[str, float]:
    """Turn ["gamma=0.4", "tile_grid=8"] into {"gamma": 0.4, "tile_grid": 8}."""
    params: dict[str, float] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param must look like name=value, got {item!r}")
        number = float(value)
        params[key] = int(number) if number.is_integer() and "." not in value else number
    return params


def _cmd_enhance(args: argparse.Namespace) -> int:
    from lowlight.datasets import read_image, write_image
    from lowlight.enhance import enhance

    out = enhance(read_image(args.input), args.method, **_parse_params(args.param))
    write_image(args.out, out)
    print(f"Saved {args.out}")
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    from lowlight.datasets import read_image, write_image, write_label
    from lowlight.enhance import enhance
    from lowlight.models import segmenter_from_config
    from lowlight.viz import colorize, overlay

    cfg = load_config(args.config)
    img = read_image(args.input)
    if args.method:
        img = enhance(img, args.method, **_parse_params(args.param))
    mask = segmenter_from_config(cfg, args.model).predict(img)
    write_image(args.out, colorize(mask))
    print(f"Saved coloured mask to {args.out}")
    if args.overlay:
        write_image(args.overlay, overlay(img, mask))
        print(f"Saved overlay to {args.overlay}")
    if args.mask_out:
        write_label(args.mask_out, mask)
        print(f"Saved trainId mask to {args.mask_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with one sub-command per task."""
    parser = argparse.ArgumentParser(prog="python -m lowlight.cli")
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed log messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-data", help="check dataset folders, counts and image/label pairs")
    p.add_argument("--config", type=Path, default=None, help="experiment YAML (default: defaults)")
    p.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), help="only these datasets")
    p.add_argument("--deep", action="store_true", help="read every image and label (slow)")
    p.add_argument("--json", type=Path, help="also save the report as a JSON file")
    p.set_defaults(func=_cmd_check_data)

    p = sub.add_parser("enhance", help="enhance one image")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--method", required=True, help="e.g. gamma, he, clahe, gamma_clahe")
    p.add_argument("--param", action="append", help="method parameter, e.g. --param gamma=0.4")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_enhance)

    p = sub.add_parser("segment", help="segment one image (optionally enhance it first)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", default="segformer_b0", help="segformer_b0 | segformer_b2 | ...")
    p.add_argument("--method", help="enhancement to apply first (default: none)")
    p.add_argument("--param", action="append", help="enhancement parameter, e.g. gamma=0.4")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="coloured mask PNG")
    p.add_argument("--overlay", type=Path, help="also save the mask blended over the image")
    p.add_argument("--mask-out", type=Path, help="also save the raw trainId mask PNG")
    p.set_defaults(func=_cmd_segment)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code (0 = success)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
