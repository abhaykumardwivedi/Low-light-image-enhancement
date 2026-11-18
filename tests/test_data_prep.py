"""Tests for lowlight.data_prep (filtering and zipping; no network)."""

import zipfile

import pytest

from lowlight.data_prep import (
    ACDC_NIGHT_PATTERNS,
    CITYSCAPES_VAL_PATTERNS,
    download_acdc,
    extract_matching,
    make_zip,
    tree_summary,
    unzip_to,
)


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, f"content of {n}")
    return path


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_extract_acdc_keeps_only_night_train_val(tmp_path):
    z = _zip(
        tmp_path / "acdc.zip",
        [
            "rgb_anon/night/train/GOPR1/a_rgb_anon.png",  # keep
            "rgb_anon/fog/train/GOPR9/f_rgb_anon.png",  # other condition: skip
            "rgb_anon/night/test/GOPR2/t_rgb_anon.png",  # test split: skip
            "ACDC/rgb_anon/night/val/GOPR3/b_rgb_anon.png",  # extra top folder: keep, strip it
            "gt/night/val/GOPR3/b_gt_labelTrainIds.png",  # keep
            "gt/night/val/GOPR3/b_gt_labelColor.png",  # colour preview: skip
            "gt/night/val/GOPR3/b_gt_invGray.png",  # invalid-region mask: skip
            "README.txt",
        ],
    )
    out = tmp_path / "out"
    assert extract_matching([z], out, ACDC_NIGHT_PATTERNS) == 3
    assert _files(out) == [
        "gt/night/val/GOPR3/b_gt_labelTrainIds.png",
        "rgb_anon/night/train/GOPR1/a_rgb_anon.png",
        "rgb_anon/night/val/GOPR3/b_rgb_anon.png",
    ]
    # Running again finds the same files and skips the ones already extracted.
    assert extract_matching([z], out, ACDC_NIGHT_PATTERNS) == 3


def test_extract_cityscapes_val_only(tmp_path):
    z = _zip(
        tmp_path / "cs.zip",
        [
            "leftImg8bit/val/lindau/x_leftImg8bit.png",
            "leftImg8bit/train/aachen/y_leftImg8bit.png",
            "gtFine/val/lindau/x_gtFine_labelIds.png",
            "gtFine/val/lindau/x_gtFine_color.png",
            "gtFine/val/lindau/x_gtFine_polygons.json",
        ],
    )
    out = tmp_path / "out"
    assert extract_matching([z], out, CITYSCAPES_VAL_PATTERNS) == 2
    assert _files(out) == [
        "gtFine/val/lindau/x_gtFine_labelIds.png",
        "leftImg8bit/val/lindau/x_leftImg8bit.png",
    ]


def test_zip_round_trip(tmp_path):
    src = tmp_path / "data" / "acdc"
    (src / "rgb_anon").mkdir(parents=True)
    (src / "rgb_anon" / "a.png").write_bytes(b"123")
    assert make_zip(src, tmp_path / "acdc_night.zip") == 1
    dest = tmp_path / "restored"
    assert unzip_to(tmp_path / "acdc_night.zip", dest) == 1
    assert (dest / "acdc" / "rgb_anon" / "a.png").read_bytes() == b"123"


class _FakeResponse:
    def __init__(self, status=200, payload=None, body=b""):
        self.status_code, self._payload, self._body = status, payload, body
        self.headers = {"Content-Length": str(len(body))} if body else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAcdc:
    """Pretends to be the ACDC API, following the steps the real website uses."""

    def __init__(self, zip_bytes, licence_agreed=True, password="secret"):
        self.zip_bytes, self.licence_agreed, self.password = zip_bytes, licence_agreed, password
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(url)
        if json["password"] != self.password:
            return _FakeResponse(401, {"msg": "wrong"})
        return _FakeResponse(200, {"token": "T", "user": {"licenseAgreed": self.licence_agreed}})

    def get(self, url, headers=None, stream=False, timeout=None):
        self.calls.append(url)
        assert headers == {"Authorization": "Bearer T"}
        if url.endswith("/packages"):  # nested, like a list of datasets with packages inside
            return _FakeResponse(200, [{"name": "ACDC", "packages": [
                {"packageId": 7, "downloadTitle": "rgb_anon_trainvaltest.zip"},
                {"packageId": 8, "downloadTitle": "gt_trainval.zip"},
            ]}])  # fmt: skip
        if "/getPackageUri/" in url:
            return _FakeResponse(200, {"token": f"dl{url.rsplit('/', 1)[1]}"})
        return _FakeResponse(200, body=self.zip_bytes)


def _zip_bytes(tmp_path):
    return _zip(tmp_path / "z.zip", ["rgb_anon/night/val/S/a_rgb_anon.png"]).read_bytes()


def test_download_acdc_follows_website_steps(tmp_path):
    fake = _FakeAcdc(_zip_bytes(tmp_path))
    paths = download_acdc("me", "secret", tmp_path / "raw", session=fake)
    assert [p.name for p in paths] == ["rgb_anon_trainvaltest.zip", "gt_trainval.zip"]
    assert all(zipfile.is_zipfile(p) for p in paths)
    assert fake.calls[0].endswith("/api/login")
    assert any(c.endswith("/downloadPackage/dl7/rgb_anon_trainvaltest.zip") for c in fake.calls)


def test_download_acdc_requires_licence_and_password(tmp_path):
    with pytest.raises(RuntimeError, match="licence"):
        download_acdc("me", "secret", tmp_path, session=_FakeAcdc(b"", licence_agreed=False))
    with pytest.raises(RuntimeError, match="login failed"):
        download_acdc("me", "wrong", tmp_path, session=_FakeAcdc(b""))


def test_download_acdc_reports_unknown_package(tmp_path):
    with pytest.raises(RuntimeError, match="Available"):
        download_acdc("me", "secret", tmp_path, ["night_only.zip"], session=_FakeAcdc(b""))


def test_tree_summary(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.png").write_bytes(b"1")
    (tmp_path / "a" / "b" / "y.png").write_bytes(b"1")
    assert tree_summary(tmp_path) == "a/b: 2 files (e.g. x.png)"


def test_unzip_refuses_paths_outside_destination(tmp_path):
    z = _zip(tmp_path / "evil.zip", ["../escape.txt"])
    with pytest.raises(ValueError, match="Unsafe"):
        unzip_to(z, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()
