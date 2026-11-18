"""Run scripts/model_check.py end to end with a fake model (the real models run on Colab)."""

import importlib.util
import json
import sys

import numpy as np
import yaml

from lowlight.config import REPO_ROOT
from lowlight.models import DummySegmenter


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "model_check", REPO_ROOT / "scripts/model_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_check_script_with_fake_model(fake_data, tmp_path, monkeypatch, capsys):
    script = _load_script()

    def fake_load(name, size, device):
        seg = DummySegmenter(np.zeros((1, 1), np.uint8), name=name)
        seg.num_params = 3_700_000
        return seg

    monkeypatch.setattr(script, "load_segmenter", fake_load)
    monkeypatch.setattr(script, "environment", lambda: {"torch": "fake"})
    # Point the default config at the fake data and a temporary results folder.
    default = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    default["datasets"].update({k: str(v) for k, v in fake_data.items()})
    default["paths"]["results_root"] = str(tmp_path / "results")
    cfg_path = tmp_path / "default.yaml"
    cfg_path.write_text(yaml.safe_dump(default), encoding="utf-8")
    from lowlight import config

    monkeypatch.setattr(config, "DEFAULT_CONFIG", cfg_path)
    original = script.load_config
    monkeypatch.setattr(script, "load_config", lambda **kw: original(default_path=cfg_path, **kw))

    argv = ["model_check.py", "--dataset", "cityscapes", "--models", "segformer_b0",
            "--short-side", "512", "--grids", "1"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    script.main()

    out = capsys.readouterr().out
    assert "SANITY CHECK" in out and "BELOW 0.65" in out  # an all-road model must fail the check
    result_dir = tmp_path / "results" / "m3_check"
    saved = json.loads((result_dir / "cityscapes_val_segformer_b0_s512.json").read_text())
    assert saved["n_images"] == 2 and saved["params_m"] == 3.7
    assert (result_dir / "summary.csv").read_text().startswith("dataset,split,model")
    assert len(list((result_dir / "grids").glob("*.jpg"))) == 1
