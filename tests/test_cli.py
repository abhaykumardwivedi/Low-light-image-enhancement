"""Tests for the command-line interface."""

import json

import pytest
from conftest import FAKE_COUNTS

import lowlight.datasets as datasets
from lowlight.cli import main


def test_check_data_passes_on_complete_fake_data(fake_cfg_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(datasets, "EXPECTED_COUNTS", FAKE_COUNTS)
    report = tmp_path / "report.json"
    code = main(["check-data", "--config", str(fake_cfg_path), "--deep", "--json", str(report)])
    assert code == 0
    assert "ALL REQUIRED DATA OK" in capsys.readouterr().out
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["deep"] is True and len(saved["results"]) == 6


def test_check_data_fails_with_real_counts(fake_cfg_path, capsys):
    # The fake data has 3 ACDC train images, not the real 400, so the check must fail.
    assert main(["check-data", "--config", str(fake_cfg_path)]) == 1
    assert "DATA CHECK FAILED" in capsys.readouterr().out


def test_enhance_command(scene, tmp_path):
    from lowlight.datasets import read_image, write_image
    from lowlight.enhance import enhance

    write_image(tmp_path / "in.png", scene)
    args = ["enhance", "--input", str(tmp_path / "in.png"), "--method", "clahe"]
    args += ["--param", "clip_limit=3", "--param", "tile_grid=4", "--out", str(tmp_path / "o.png")]
    assert main(args) == 0
    expected = enhance(scene, "clahe", clip_limit=3.0, tile_grid=4)
    assert (read_image(tmp_path / "o.png") == expected).all()


def test_bad_param_format(scene, tmp_path):
    from lowlight.datasets import write_image

    write_image(tmp_path / "in.png", scene)
    with pytest.raises(SystemExit):
        main(["enhance", "--input", str(tmp_path / "in.png"), "--method", "gamma",
              "--param", "gamma", "--out", str(tmp_path / "o.png")])  # fmt: skip


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["fly-to-moon"])
