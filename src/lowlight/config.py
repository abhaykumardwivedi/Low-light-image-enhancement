"""Load, merge and validate the YAML config files.

Every experiment config (configs/experiments/*.yaml) is merged on top of
configs/default.yaml, so an experiment file only needs the settings it changes.
Validation happens once, here, so the rest of the code can trust the values.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# src/lowlight/config.py -> parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
DARKNESS_CONFIG = CONFIG_DIR / "darkness_levels.yaml"

VALID_DEVICES = ("auto", "cpu", "cuda")
REQUIRED_PATH_KEYS = ("data_root", "results_root", "weights_root")
DARKNESS_LEVEL_NAMES = ("L0", "L1", "L2", "L3", "L4")


class ConfigError(ValueError):
    """Raised when a config file is missing a setting or has a wrong value."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file and return it as a dict (an empty file gives {})."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)  # safe_load never runs code hidden in the YAML
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping (key: value pairs) at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: `base` with `override` merged in.

    Nested dicts are merged key by key; any other value in `override` replaces the
    one in `base`. Neither input is modified.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Turn a config path into an absolute Path. Relative paths are relative to `root`."""
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def validate_config(cfg: dict[str, Any]) -> None:
    """Check the settings every part of the code relies on. Raise ConfigError if wrong."""
    seed = cfg.get("seed")
    # bool is a subclass of int in Python, so exclude it explicitly.
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError(f"'seed' must be a non-negative integer, got {seed!r}")

    if cfg.get("device") not in VALID_DEVICES:
        raise ConfigError(f"'device' must be one of {VALID_DEVICES}, got {cfg.get('device')!r}")

    size = cfg.get("inference_short_side")
    if not isinstance(size, int) or isinstance(size, bool) or size < 32:
        raise ConfigError(f"'inference_short_side' must be an integer >= 32, got {size!r}")

    if not isinstance(cfg.get("sliding_window"), bool):
        raise ConfigError("'sliding_window' must be true or false")

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("'paths' section is missing")
    for key in REQUIRED_PATH_KEYS:
        if not paths.get(key):
            raise ConfigError(f"'paths.{key}' is missing")

    if "limit" in cfg and cfg["limit"] is not None:
        limit = cfg["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ConfigError(f"'limit' must be a positive integer or null, got {limit!r}")


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    default_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Load default.yaml, merge an experiment file and overrides on top, then validate.

    Args:
        path: optional experiment YAML to merge on top of the defaults.
        overrides: optional dict merged last (used by the CLI, e.g. {"limit": 20}).
        default_path: the defaults file (tests pass a temporary one).

    Returns:
        The merged, validated config dict.
    """
    cfg = load_yaml(default_path)
    if path is not None:
        cfg = deep_merge(cfg, load_yaml(path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def load_darkness_levels(path: str | Path = DARKNESS_CONFIG) -> dict[str, dict[str, Any]]:
    """Load and validate the L0-L4 darkening settings (PID Section 9.2)."""
    levels = load_yaml(path).get("levels")
    if not isinstance(levels, dict):
        raise ConfigError(f"{path} must have a 'levels' section")

    missing = [name for name in DARKNESS_LEVEL_NAMES if name not in levels]
    if missing:
        raise ConfigError(f"Darkness levels missing: {missing}")

    for name, p in levels.items():
        if not (0 < p.get("alpha", -1) <= 1):
            raise ConfigError(f"{name}: 'alpha' must be in (0, 1]")
        if not p.get("g", 0) >= 1:
            raise ConfigError(f"{name}: 'g' must be >= 1 (values below 1 would brighten)")
        # peak and sigma must be both set (noise on) or both null (no noise, e.g. L0).
        if (p.get("peak") is None) != (p.get("sigma") is None):
            raise ConfigError(f"{name}: set both 'peak' and 'sigma', or neither")
        if p.get("peak") is not None and (p["peak"] <= 0 or p["sigma"] < 0):
            raise ConfigError(f"{name}: 'peak' must be > 0 and 'sigma' >= 0")
    return levels
