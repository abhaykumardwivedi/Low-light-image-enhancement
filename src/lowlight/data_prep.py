"""Helpers for getting the datasets into the cloud (used by notebooks/colab_data_setup.ipynb).

The big official zips contain much more than we need (all ACDC weather conditions, all
Cityscapes splits, colour previews, polygons...). These helpers extract only the files the
project uses, pack them into small zips for Google Drive, and unpack them again at the start of
each Colab session. The filtering and zipping functions are pure Python and are unit-tested;
`download_cityscapes` needs the internet and an approved Cityscapes account, so it runs on Colab.
"""

from __future__ import annotations

import logging
import re
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

# Each pattern finds a wanted file anywhere inside a zip and captures its path starting at the
# dataset's own top folder. Anything before that (e.g. an extra "ACDC/" folder) is dropped, so
# the output layout is always the one the loaders expect (docs/DATA.md).
ACDC_NIGHT_PATTERNS = (
    re.compile(r"(?:^|/)(rgb_anon/night/(?:train|val)/[^/]+/[^/]+_rgb_anon\.png)$"),
    re.compile(r"(?:^|/)(gt/night/(?:train|val)/[^/]+/[^/]+_gt_labelTrainIds\.png)$"),
)
CITYSCAPES_VAL_PATTERNS = (
    re.compile(r"(?:^|/)(leftImg8bit/val/[^/]+/[^/]+_leftImg8bit\.png)$"),
    re.compile(r"(?:^|/)(gtFine/val/[^/]+/[^/]+_gtFine_labelIds\.png)$"),
)
DARKZURICH_VAL_PATTERNS = (
    re.compile(r"(?:^|/)(rgb_anon/val/night/[^/]+/[^/]+_rgb_anon\.png)$"),
    re.compile(r"(?:^|/)(gt/val/night/[^/]+/[^/]+_gt_labelTrainIds\.png)$"),
)

# Cityscapes download package ids (as listed by the official cityscapesScripts downloader).
CITYSCAPES_PACKAGES = {
    "gtFine_trainvaltest.zip": 1,
    "leftImg8bit_trainvaltest.zip": 3,
}
# ACDC's website talks to this JSON API (read from the site's own JavaScript, 2026-11-19).
ACDC_API = "https://acdc.vision.ee.ethz.ch/api"
ACDC_PACKAGES = ("rgb_anon_trainvaltest.zip", "gt_trainval.zip")

# NightCity (Tan et al., 2021): public Google Drive files, no account needed (decision D28).
# "fine" = NightCity-fine, the corrected labels from the DTP project (Wei et al., ICCV 2023).
NIGHTCITY_FILES = {
    "nightcity_fine.zip": "1Ilj99NMAmkZIPQcVOd6cJebnKXjJ-Sit",
    "nightcity_images.zip": "1hZu6SwjloLLyOAZT23KdGTt_Rg1y63o4",
}

CITYSCAPES_LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
CITYSCAPES_FILE_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID={}"


def _safe_relpath(rel: str) -> PurePosixPath:
    """Reject paths that could write outside the output folder (e.g. '../x' or '/x')."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe path inside zip: {rel!r}")
    return p


def extract_matching(
    zip_paths: Iterable[str | Path],
    out_dir: str | Path,
    patterns: Sequence[re.Pattern[str]],
) -> int:
    """Extract only the zip members matching one of `patterns` into `out_dir`.

    Each matching member is written to `out_dir / <captured path>`. Files that already exist with
    the same size are skipped, so re-running after a disconnect is cheap.

    Returns:
        The number of matching files (extracted or already present).
    """
    out_dir = Path(out_dir)
    count = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/")  # some zips made on Windows use "\"
                match = next((m for p in patterns if (m := p.search(name))), None)
                if match is None:
                    continue
                target = out_dir / _safe_relpath(match.group(1))
                count += 1
                if target.is_file() and target.stat().st_size == info.file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        logger.info("Scanned %s", zip_path)
    return count


def make_zip(folder: str | Path, zip_path: str | Path) -> int:
    """Zip `folder` so that unzipping into its parent recreates it. Returns the file count.

    Example: make_zip("data/acdc", "acdc_night.zip") stores "acdc/rgb_anon/...". PNG files are
    already compressed, so they are stored without recompressing (much faster, same size).
    """
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, f.relative_to(folder.parent).as_posix())
    return len(files)


def unzip_to(zip_path: str | Path, dest: str | Path) -> int:
    """Unzip everything into `dest` (with the same safety check). Returns the file count."""
    dest = Path(dest)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = dest / _safe_relpath(info.filename.replace("\\", "/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    return count


def _stream_to_file(response, target: Path, name: str) -> None:
    """Write a streamed HTTP response to `target`, logging progress about once per GB."""
    done = 0
    with target.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            done += len(chunk)
            if done % (1024**3) < len(chunk):
                logger.info("%s: %.1f GB", name, done / 1024**3)


def download_gdrive(file_id: str, target: str | Path) -> Path:
    """Download a public Google Drive file with `gdown` (preinstalled on Colab).

    Raises:
        RuntimeError: Google refused the download (popular public files sometimes hit a daily
            download quota; the message explains the manual fallback).
    """
    import gdown  # only needed on Colab

    target = Path(target)
    if target.is_file() and zipfile.is_zipfile(target):
        logger.info("%s already downloaded", target.name)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(id=file_id, output=str(target), quiet=False)
    if out is None or not zipfile.is_zipfile(target):
        raise RuntimeError(
            f"Google Drive did not send {target.name} (often a temporary download limit). "
            "Try again in a few hours, or open "
            f"https://drive.google.com/file/d/{file_id}/view, download it, "
            "and put it in My Drive/lowlight-seg/data/raw/."
        )
    return target


def tree_summary(root: str | Path, max_depth: int = 4) -> str:
    """Describe a folder tree as 'folder: N files (example name)' lines, to check a new layout."""
    root = Path(root)
    lines = []
    for d in sorted([root, *(p for p in root.rglob("*") if p.is_dir())]):
        depth = len(d.relative_to(root).parts)
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        if depth <= max_depth and files:
            lines.append(
                f"{d.relative_to(root).as_posix() or '.'}: {len(files)} files (e.g. {files[0]})"
            )
    return "\n".join(lines)


def _find_packages(data: object) -> list[dict]:
    """Collect every dict with 'packageId' and 'downloadTitle' from the API's JSON (any nesting)."""
    found = []
    if isinstance(data, dict):
        if "packageId" in data and "downloadTitle" in data:
            found.append(data)
        for value in data.values():
            found += _find_packages(value)
    elif isinstance(data, list):
        for item in data:
            found += _find_packages(item)
    return found


def download_acdc(
    username: str,
    password: str,
    out_dir: str | Path,
    package_names: Sequence[str] = ACDC_PACKAGES,
    session=None,
) -> list[Path]:
    """Download ACDC packages with your own ACDC account (run on Colab).

    Uses the same steps as the ACDC website: log in, list the packages, ask for a download link,
    download. You must have accepted the ACDC licence once on the website yourself; this function
    never accepts it for you.

    Args:
        session: an HTTP session (tests pass a fake one); a new `requests.Session` by default.

    Raises:
        RuntimeError: login failed, licence not accepted yet, or a package was not found.
    """
    if session is None:
        import requests  # preinstalled on Colab

        session = requests.Session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = session.post(
        f"{ACDC_API}/login", json={"username": username, "password": password}, timeout=60
    )
    if resp.status_code != 200:
        raise RuntimeError(
            "ACDC login failed. Check ACDC_USERNAME / ACDC_PASSWORD in Colab Secrets, and that you "
            "clicked the activation link in the ACDC email."
        )
    login = resp.json()
    token, user = login.get("token"), login.get("user") or {}
    if not token:
        raise RuntimeError(f"ACDC login gave no token (response keys: {sorted(login)})")
    if user.get("licenseAgreed") is False:
        raise RuntimeError(
            "You have not accepted the ACDC licence yet. Log in at https://acdc.vision.ee.ethz.ch, "
            "open Download, click any package and accept the licence once, then run this again."
        )
    headers = {"Authorization": f"Bearer {token}"}

    r = session.get(f"{ACDC_API}/packages", headers=headers, timeout=60)
    r.raise_for_status()
    packages = {p["downloadTitle"].lower(): p for p in _find_packages(r.json())}

    paths = []
    for name in package_names:
        pkg = packages.get(name.lower())
        if pkg is None:
            raise RuntimeError(
                f"ACDC package {name!r} not found. Available: {sorted(packages)}. "
                "Update the package names in the code to match this list."
            )
        target = out_dir / pkg["downloadTitle"]
        uri = session.get(
            f"{ACDC_API}/getPackageUri/{pkg['packageId']}", headers=headers, timeout=60
        )
        uri.raise_for_status()
        url = f"{ACDC_API}/downloadPackage/{uri.json()['token']}/{pkg['downloadTitle']}"
        with session.get(url, headers=headers, stream=True, timeout=60) as d:
            d.raise_for_status()
            size = int(d.headers.get("Content-Length", 0))
            if target.is_file() and size and target.stat().st_size == size:
                logger.info("%s already downloaded", name)
            else:
                logger.info("Downloading %s (%.1f GB)", name, size / 1024**3)
                _stream_to_file(d, target, name)
        if not zipfile.is_zipfile(target):
            raise RuntimeError(f"{target} is not a valid zip file (download incomplete?)")
        paths.append(target)
    return paths


def download_cityscapes(
    username: str,
    password: str,
    package_names: Sequence[str],
    out_dir: str | Path,
) -> list[Path]:
    """Download official Cityscapes packages with your own account (run on Colab).

    This logs in on the official website exactly like a browser does, then downloads each
    package. Files that are already fully downloaded are skipped.

    Raises:
        RuntimeError: login failed, the account is not approved yet, or the download is not a zip.
    """
    import requests  # only needed on Colab, where it is preinstalled

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    resp = session.post(
        CITYSCAPES_LOGIN_URL,
        data={"username": username, "password": password, "submit": "Login"},
        timeout=60,
    )
    resp.raise_for_status()

    paths = []
    for name in package_names:
        target = out_dir / name
        url = CITYSCAPES_FILE_URL.format(CITYSCAPES_PACKAGES[name])
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            size = int(r.headers.get("Content-Length", 0))
            if target.is_file() and size and target.stat().st_size == size:
                logger.info("%s already downloaded", name)
                paths.append(target)
                continue
            if "zip" not in r.headers.get("Content-Type", "") and size < 1_000_000:
                raise RuntimeError(
                    "Cityscapes did not send a zip file. Check the username and password in "
                    "Colab Secrets, and that your account has been approved (approval email)."
                )
            _stream_to_file(r, target, name)
        if not zipfile.is_zipfile(target):
            raise RuntimeError(f"{target} is not a valid zip file (download incomplete?)")
        paths.append(target)
    return paths
