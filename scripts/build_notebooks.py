"""Generate the Colab notebooks in notebooks/ from the cell texts below.

The .ipynb files are JSON and hard to review or edit by hand, so the source of truth is this
script. After changing it, run:  python scripts/build_notebooks.py
"""

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

cells = []


def md(text):
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(r"""
# Data setup on Google Colab (run once)

This notebook puts the datasets into **your private Google Drive**, ready for every later experiment.
Nothing is stored on your laptop. It:

1. unpacks the project code from Drive,
2. downloads **NightCity** (real night street images, **no account needed**),
3. downloads **Cityscapes** with your own account and keeps only the 500 val images + labels,
4. downloads **ACDC** with your own account and keeps only the night train/val images + labels,
5. (optional) does the same for **Dark Zurich** val,
6. saves small zips to `My Drive/lowlight-seg/data/` and runs `check-data` on everything.

**This notebook does not need a GPU.** Use *Runtime → Change runtime type → CPU* to save your free GPU time.
Every step can be re-run safely: finished steps are skipped.

## Before you start (one time, about 5 minutes)
NightCity needs nothing from you: you can press *Run all* right away. The ACDC and Cityscapes steps are
skipped (with a message) until you have accounts.

1. When you have them: make sure your ACDC and Cityscapes accounts are active, and that you accepted the ACDC licence once on
   its website (log in, open **Download**, click a package, accept).
2. In this Colab notebook, click the **key icon** (Secrets) in the left sidebar and add four secrets,
   each with *Notebook access* turned on:
   `ACDC_USERNAME`, `ACDC_PASSWORD`, `CITYSCAPES_USERNAME`, `CITYSCAPES_PASSWORD`.
   Never type a password into a cell. Colab remembers these secrets for your Google account.
3. The project code must be at `My Drive/lowlight-seg/code/lowlight-code.zip` (made by `scripts/pack_code.ps1`).

Run the cells from top to bottom (*Runtime → Run all* also works).
""")

md("## 1. Connect Google Drive")
code(r"""
from google.colab import drive

drive.mount("/content/drive")

from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/lowlight-seg")  # your private project folder
CODE_ZIP = DRIVE / "code" / "lowlight-code.zip"
DATA_DRIVE = DRIVE / "data"            # prepared dataset zips live here
RAW_DRIVE = DATA_DRIVE / "raw"         # browser-downloaded original zips go here
RESULTS_DRIVE = DRIVE / "results"
REPO = Path("/content/lowlight-seg")   # the code is unpacked here (fast local disk)

for folder in (DATA_DRIVE, RAW_DRIVE, RESULTS_DRIVE):
    folder.mkdir(parents=True, exist_ok=True)

if not CODE_ZIP.is_file():
    raise FileNotFoundError(f"Upload lowlight-code.zip to {CODE_ZIP.parent} first (see 'Before you start').")
print("Drive connected:", DRIVE)
""")

md("## 2. Unpack the code and record the environment")
code(r"""
import json, platform, shutil, subprocess, sys, zipfile

if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
with zipfile.ZipFile(CODE_ZIP) as zf:
    zf.extractall(REPO)
    COMMIT = zf.comment.decode() or "unknown"  # pack_code.ps1 (git archive) stores the commit id here

# Install the project package (dependencies used here are already preinstalled on Colab).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "--no-deps"], check=True)
sys.path.insert(0, str(REPO / "src"))  # make `import lowlight` work in this running notebook too

import logging
import cv2, numpy, yaml
import lowlight.data_prep as dp

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

env = {
    "code_commit": COMMIT,
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "pyyaml": yaml.__version__,
}
(RESULTS_DRIVE / "colab_env_data_setup.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))
""")


md(r"""
## 3. NightCity (no account needed)

Downloads NightCity from its public Google Drive links: the corrected **NightCity-fine** labels (Wei et al.,
ICCV 2023), plus the original images if the labels zip does not already contain them. Everything is unpacked
into `data/nightcity`, a short summary of the folder layout is printed (and saved to
`results/nightcity_layout.txt`), and `nightcity.zip` is saved to Drive.
""")
code(r"""
from lowlight.datasets import find_nightcity

NC_ZIP = DATA_DRIVE / "nightcity.zip"
nc_root = REPO / "data" / "nightcity"


def nightcity_count():
    try:
        return len(find_nightcity(nc_root, "train")) + len(find_nightcity(nc_root, "val"))
    except Exception:
        return 0


if NC_ZIP.is_file():
    print("nightcity.zip is already on Drive - skipping.")
else:
    raw = Path("/content/raw_nightcity")
    manual = sorted(RAW_DRIVE.glob("*ight*ity*.zip"))  # fallback: zips you uploaded by hand
    try:
        fine = dp.download_gdrive(dp.NIGHTCITY_FILES["nightcity_fine.zip"], raw / "nightcity_fine.zip")
        dp.unzip_to(fine, nc_root / "fine")
        if nightcity_count() < 1000:  # the fine zip holds only labels: fetch the images too
            imgs = dp.download_gdrive(dp.NIGHTCITY_FILES["nightcity_images.zip"], raw / "nightcity_images.zip")
            dp.unzip_to(imgs, nc_root / "images")
    except RuntimeError as e:
        if not manual:
            raise
        print(f"Automatic download failed ({e}); using the zips in data/raw instead.")
        for z in manual:
            dp.unzip_to(z, nc_root / z.stem)
    layout = dp.tree_summary(nc_root)
    print(layout)
    (RESULTS_DRIVE / "nightcity_layout.txt").write_text(layout)
    print(f"Found {nightcity_count()} NightCity images with the expected folder names.")
    local_zip = Path("/content/nightcity.zip")
    dp.make_zip(nc_root, local_zip)
    shutil.copy(local_zip, NC_ZIP)
    shutil.rmtree(raw, ignore_errors=True)
    print("Saved", NC_ZIP, f"({NC_ZIP.stat().st_size / 1e9:.2f} GB)")
""")

md(r"""
## 4. Cityscapes val (download inside Colab)

Downloads about **11.5 GB** from the official website onto Colab's temporary disk (often 10–30 minutes),
keeps only the 500 val images and their labels (~1–1.5 GB), and saves `cityscapes_val.zip` to Drive.
The big zip never reaches your laptop or your Drive. If your account is not approved yet, this cell
just prints a message and skips; run it again later.
""")
code(r"""
CS_ZIP = DATA_DRIVE / "cityscapes_val.zip"

if CS_ZIP.is_file():
    print("cityscapes_val.zip is already on Drive - skipping.")
else:
    from google.colab import userdata
    try:
        cs_user = userdata.get("CITYSCAPES_USERNAME")
        cs_pass = userdata.get("CITYSCAPES_PASSWORD")
    except Exception as e:  # secret missing or notebook access not granted
        cs_user = cs_pass = None
        print("Skipping Cityscapes: add CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD in Colab Secrets",
              f"(key icon) with 'Notebook access' on. Details: {e}")
    if cs_user and cs_pass:
        raw = Path("/content/raw_cityscapes")
        zips = dp.download_cityscapes(
            cs_user, cs_pass, ["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"], raw
        )
        n = dp.extract_matching(zips, REPO / "data" / "cityscapes", dp.CITYSCAPES_VAL_PATTERNS)
        print(f"Kept {n} files (expected 1000 = 500 images + 500 labels).")
        local_zip = Path("/content/cityscapes_val.zip")
        dp.make_zip(REPO / "data" / "cityscapes", local_zip)
        shutil.copy(local_zip, CS_ZIP)
        shutil.rmtree(raw)  # free Colab disk space
        print("Saved", CS_ZIP, f"({CS_ZIP.stat().st_size / 1e9:.2f} GB)")
""")

md(r"""
## 5. ACDC night (download inside Colab)

Logs in to ACDC with your secrets, downloads the images and labels zips onto Colab's temporary disk, keeps only
the **night** train and val images with their `*_gt_labelTrainIds.png` labels, and saves `acdc_night.zip` to Drive.
If the automatic download does not work, upload the ACDC zips to `My Drive/lowlight-seg/data/raw/` by hand and
run this cell again; it then uses those instead.
""")
code(r"""
ACDC_ZIP = DATA_DRIVE / "acdc_night.zip"


def copy_raw_to_local(zips, folder):
    # Reading a big zip directly from Drive is slow, so copy it to Colab's local disk first.
    folder.mkdir(parents=True, exist_ok=True)
    local = []
    for z in zips:
        dst = folder / z.name
        if not (dst.is_file() and dst.stat().st_size == z.stat().st_size):
            print("Copying", z.name, f"({z.stat().st_size / 1e9:.2f} GB) to local disk...")
            shutil.copy(z, dst)
        local.append(dst)
    return local


from google.colab import userdata

if ACDC_ZIP.is_file():
    print("acdc_night.zip is already on Drive - skipping.")
else:
    local = []
    try:
        acdc_user = userdata.get("ACDC_USERNAME")
        acdc_pass = userdata.get("ACDC_PASSWORD")
    except Exception as e:
        acdc_user = acdc_pass = None
        print(f"No ACDC secrets ({e}); looking for uploaded zips in {RAW_DRIVE} instead.")
    if acdc_user and acdc_pass:
        local = dp.download_acdc(acdc_user, acdc_pass, Path("/content/raw_acdc"))
    else:
        acdc_raw = [z for z in sorted(RAW_DRIVE.glob("*.zip")) if not any(w in z.name.lower() for w in ("zurich", "nightcity"))]
        if acdc_raw:
            print("Using:", [z.name for z in acdc_raw])
            local = copy_raw_to_local(acdc_raw, Path("/content/raw_acdc"))
    if not local:
        print("ACDC skipped: add the ACDC secrets (or upload the zips) and run this cell again.")
    else:
        n = dp.extract_matching(local, REPO / "data" / "acdc", dp.ACDC_NIGHT_PATTERNS)
        print(f"Kept {n} files (expected 1012 = 506 images + 506 labels).")
        if n == 0:
            raise RuntimeError("No ACDC night files found inside these zips. Compare the zip contents "
                               "below with ACDC_NIGHT_PATTERNS in data_prep.py:\n" +
                               str(zipfile.ZipFile(local[0]).namelist()[:20]))
        local_zip = Path("/content/acdc_night.zip")
        dp.make_zip(REPO / "data" / "acdc", local_zip)
        shutil.copy(local_zip, ACDC_ZIP)
        shutil.rmtree("/content/raw_acdc")
        print("Saved", ACDC_ZIP, f"({ACDC_ZIP.stat().st_size / 1e9:.2f} GB)")
""")

md(r"""
## 6. Dark Zurich val (optional stretch goal)

Only runs if a zip with "zurich" in its name is in `data/raw/`.
""")
code(r"""
DZ_ZIP = DATA_DRIVE / "darkzurich_val.zip"

if DZ_ZIP.is_file():
    print("darkzurich_val.zip is already on Drive - skipping.")
else:
    dz_raw = [z for z in sorted(RAW_DRIVE.glob("*.zip")) if "zurich" in z.name.lower()]
    if not dz_raw:
        print("No Dark Zurich zip in data/raw/ - skipping (optional).")
    else:
        local = copy_raw_to_local(dz_raw, Path("/content/raw_dz"))
        n = dp.extract_matching(local, REPO / "data" / "darkzurich", dp.DARKZURICH_VAL_PATTERNS)
        print(f"Kept {n} files (expected 100 = 50 images + 50 labels).")
        local_zip = Path("/content/darkzurich_val.zip")
        dp.make_zip(REPO / "data" / "darkzurich", local_zip)
        shutil.copy(local_zip, DZ_ZIP)
        shutil.rmtree("/content/raw_dz")
        print("Saved", DZ_ZIP)
""")

md(r"""
## 7. Check all data

Unpacks the prepared zips (if this is a fresh session) and runs `check-data --deep`, which opens every
image and label. The report is also saved to `My Drive/lowlight-seg/results/data_check.json`.
**Save the printed table** (copy it into `docs/PROGRESS.md`) for the M1 checkpoint.
""")
code(r"""
for z in (DATA_DRIVE / "nightcity.zip", DATA_DRIVE / "acdc_night.zip", DATA_DRIVE / "cityscapes_val.zip",
          DATA_DRIVE / "darkzurich_val.zip"):
    marker = REPO / "data" / f".{z.stem}_unpacked"
    if z.is_file() and not marker.exists():
        print("Unpacking", z.name)
        dp.unzip_to(z, REPO / "data")
        marker.touch()

result = subprocess.run(
    [sys.executable, "-m", "lowlight.cli", "check-data", "--deep",
     "--json", str(RESULTS_DRIVE / "data_check.json")],
    cwd=REPO, capture_output=True, text=True,
)
print(result.stdout)
print(result.stderr)
print("Code commit:", COMMIT)
""")

md(r"""
## 8. (Optional) Free Drive space

After check-data says **ALL REQUIRED DATA OK**, the original ACDC/Dark Zurich zips in `data/raw/` are no
longer needed. Set `DELETE_RAW = True` and run this cell to delete them.
""")
code(r"""
DELETE_RAW = False  # change to True only after the data check passed

if DELETE_RAW:
    for z in sorted(RAW_DRIVE.glob("*.zip")):
        print("Deleting", z.name)
        z.unlink()
else:
    print("Nothing deleted. Raw zips:", [z.name for z in RAW_DRIVE.glob("*.zip")])
""")


def save(filename):
    """Write the collected cells as a notebook and start a new, empty cell list."""
    nb = {
        "cells": list(cells),
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path = NOTEBOOK_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path, len(cells), "cells")
    cells.clear()


# The "connect Drive" and "unpack code" cells (with their headings) are shared by every notebook.
SETUP_CELLS = [dict(c) for c in cells[1:5]]
save("colab_data_setup.ipynb")


# =============================================================================================
# notebooks/colab_model_check.ipynb (M3)
# =============================================================================================

md(r"""
# Model check on Google Colab (M3)

Checks that the segmentation models work correctly before any experiment is run:

1. installs the pinned model library (`transformers`),
2. runs **all** tests, including the slow ones that download and run the real models,
3. scores SegFormer-B0, SegFormer-B2 and Mask2Former on **50 NightCity val images** and saves example
   pictures (none / gamma / CLAHE / gamma+CLAHE),
4. if Cityscapes is available: the **sanity check** - SegFormer-B0 on all 500 clean Cityscapes val images at
   short side 512 and 1024 (should be about 65% mIoU or more),
5. copies everything to `My Drive/lowlight-seg/results/m3_check/`.

**Use a GPU:** *Runtime -> Change runtime type -> T4 GPU*, then *Runtime -> Run all*. It takes about 10-20 minutes.
Run `colab_data_setup.ipynb` first (once). When it finishes, **save the printed output of steps 4-6**
(copy it into `docs/PROGRESS.md`) for the M3 checkpoint.
""")
cells.extend(dict(c) for c in SETUP_CELLS)

md("## 3. Install the pinned model library")
code(r"""
TRANSFORMERS_VERSION = "5.23.0"  # the version pinned in requirements.txt
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={TRANSFORMERS_VERSION}"],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
check = subprocess.run([sys.executable, "-c", "import torch, transformers; "
                        "print('torch', torch.__version__, '| transformers', transformers.__version__, "
                        "'| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
                       capture_output=True, text=True)
print(check.stdout, check.stderr[-2000:])
if "GPU: NONE" in check.stdout:
    print("WARNING: no GPU. Use Runtime -> Change runtime type -> T4 GPU, then run all again.")
""")

md("## 4. Unpack the prepared data")
code(r"""
for z in sorted(DATA_DRIVE.glob("*.zip")):
    marker = REPO / "data" / f".{z.stem}_unpacked"
    if not marker.exists():
        print("Unpacking", z.name)
        dp.unzip_to(z, REPO / "data")
        marker.touch()
print(subprocess.run([sys.executable, "-m", "lowlight.cli", "check-data"], cwd=REPO,
                     capture_output=True, text=True).stdout)
""")

md(r"""
## 5. Run every test, including the slow model tests
""")
code(r"""
r = subprocess.run([sys.executable, "-m", "pytest", "-m", "slow or not slow", "-q", "-rs", "-p", "no:cacheprovider"],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
""")

md(r"""
## 6. Score the three models on NightCity val (50 images) and save example pictures
""")
code(r"""
def run(args):
    r = subprocess.run([sys.executable, "scripts/model_check.py", *args], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-4000:])

if (REPO / "data" / "nightcity").is_dir():
    run(["--dataset", "nightcity", "--split", "val", "--limit", "50"])
else:
    print("NightCity is not unpacked: run colab_data_setup.ipynb first.")
""")

md(r"""
## 7. Sanity check on clean Cityscapes val (only if Cityscapes was downloaded)
""")
code(r"""
if (REPO / "data" / "cityscapes").is_dir():
    run(["--dataset", "cityscapes", "--split", "val", "--models", "segformer_b0",
         "--short-side", "512", "1024", "--grids", "2"])
else:
    print("Cityscapes not available yet - sanity check skipped (it runs once you have the account).")
""")

md("## 8. Copy the results to Drive")
code(r"""
src = REPO / "results" / "m3_check"
if src.is_dir():
    shutil.copytree(src, RESULTS_DRIVE / "m3_check", dirs_exist_ok=True)
    print("Copied to", RESULTS_DRIVE / "m3_check")
    print((src / "summary.csv").read_text())
""")
save("colab_model_check.ipynb")
