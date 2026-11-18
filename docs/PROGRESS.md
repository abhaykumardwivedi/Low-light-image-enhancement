# Progress

## Current status
- **Current milestone:** M1 (data layer). Code is DONE and tested; waiting for dataset access and a run of `notebooks/colab_data_setup.ipynb`, then the `check-data` table is copied here. M1 is tagged only after the real counts (400 / 106 / 500) are confirmed.
- **M2 (enhancement and darkening):** code DONE and tested (138 tests). Waiting for real data only for the calibration run + figure (`scripts/calibrate_darkness.py`, on Colab; needs a night train split and Cityscapes val).
- **M3 (segmentation and metrics):** code DONE and tested locally (171 tests). Waiting for a run of `notebooks/colab_model_check.ipynb` (GPU) for the slow model tests, the NightCity model scores and the example grids. The Cityscapes sanity check (B0 >= ~65% mIoU) waits for the Cityscapes account.
- **Next:** M4 (runner + experiments). The runner can be built and tested with fake data now.

## Done
### M0 (2026-11-19)
- Repo at `C:\Projects\lowlight-seg`, Git initialised (branch `main`).
- Python 3.12.10 installed (winget), `.venv` created.
- `.gitignore`, this file, `docs/PID.md` (copy of PID v1.1), `docs/DATA.md`, `docs/EXPERIMENTS.md` (stub).
- `pyproject.toml` (package + ruff + pytest config), `requirements.txt` (pinned).
- `configs/default.yaml`, `configs/darkness_levels.yaml` (starting values, not yet calibrated).
- `src/lowlight/config.py` (load, merge, validate YAML), `src/lowlight/labels.py` (classes, palette, labelId→trainId LUT, VOC mapping, id2label check).
- Tests: `tests/test_config.py`, `tests/test_labels.py`.
- Explanations: `docs/explanations/setup.md`, `config.md`, `labels.md`.
- `results/ENVIRONMENT.md`.
- Checks: 25 tests pass, `ruff check` and `ruff format --check` clean, `pip install --dry-run -r requirements.txt` resolves all pins together, and a fresh clone + new venv + README steps passed all tests.

### M1 (2026-11-19, code part)
- `src/lowlight/datasets.py`: RGB/BGR-safe image and label reading (also non-English paths), finders for ACDC night / Cityscapes val / Dark Zurich val / plain folders, `select_subset` (seeded dev subsets), `build_dataset`, `check_datasets` + report.
- `src/lowlight/data_prep.py`: extract only needed files from official zips, make/unpack Drive zips, Cityscapes download with our own login (Colab).
- `src/lowlight/cli.py` with `check-data`; `scripts/check_data.py`; `scripts/pack_code.ps1` (git archive -> zip for Colab).
- `notebooks/colab_data_setup.ipynb`: rehearsed locally end to end with a fake `google.colab`, fake Drive and a fake ACDC zip with real counts (400/106 kept; fog, test and colour previews dropped; check-data OK for ACDC). The real Cityscapes download could not be rehearsed (needs an approved account).
- Tests: 54 pass (`test_datasets.py`, `test_data_prep.py`, `test_cli.py`, shared fake data in `conftest.py`).
- Explanations: `datasets.md`, `data_prep.md`, `cli.md`.

### M2 (2026-11-19, code part)
- `src/lowlight/enhance.py`: registry + none, gamma, he, clahe, gamma_clahe, adaptive_gamma (L channel of LAB for HE/CLAHE; LUT for gamma; flat-image guard).
- `src/lowlight/darken.py`: Poisson-Gaussian darkening L0-L4, CRC32 per-image seed (D2), noise-free ablation.
- `src/lowlight/features.py`: 9 selector features (L on 0-1 scale, D3), Immerkær noise estimate (D31).
- `src/lowlight/calibrate.py` + `scripts/calibrate_darkness.py`: bisection fit of L3/L4 alpha so they bracket real night mean lightness; YAML rewrite; histogram figure.
- Tests: `test_enhance.py`, `test_darken.py`, `test_features.py`, `test_calibrate.py`. Explanations: `enhance.md`, `darken.md`, `features.md`.

### M3 (2026-11-19, code part)
- `metrics.py` (bincount confusion matrix, IoU/Dice/PA/class acc, NaN rule D5, save/load .npz), `stats.py` (paired bootstrap CI with weighted resampling, Wilcoxon, `Comparison.significant`).
- `models.py` (SegFormer/Mask2Former wrappers, short-side resize, model mean/std, id2label remap, Mask2Former padding + manual post-processing D36, cached loading), `evaluate.py` (darken -> enhance -> predict -> score loop).
- `viz.py` (colorize, overlay, error map, comparison grid, legend, histogram chart, colour-blind method colours), CLI `enhance` and `segment`.
- `scripts/model_check.py` (scores + sanity rule + example grids), `scripts/build_notebooks.py` (source of both Colab notebooks), `notebooks/colab_model_check.ipynb`.
- Tests: `test_metrics.py` (hand-made 4x4 example), `test_stats.py` (CI contains the true difference), `test_viz.py`, `test_models.py` (slow real-model tests for Colab), `test_evaluate.py`, `test_model_check_script.py`. Explanations: `metrics.md`, `stats.md`, `models.md`.

## Open problems / waiting
- Dataset access: ACDC and Cityscapes registration (personal email, D20). Blocks the M1 checkpoint.
- Confirm which Google account holds the project Drive and runs Colab (D27).
- First real run of `colab_data_setup.ipynb` (ACDC + Cityscapes downloads are untested against the real servers).
- Group member names and supervisor name: added at the END of the project (D17).
- First Colab session: check Colab's Python version and that the pinned requirements install there.

## Decision log
| Date | ID | Decision | Why |
|---|---|---|---|
| 2026-11-19 | D1 | Repo lives at `C:\Projects\lowlight-seg`, not inside OneDrive. | OneDrive would sync `.venv`, ~25 GB of data and `.git` (risk of corruption); path had spaces. |
| 2026-11-19 | D2 | Per-image random seed = `seed + zlib.crc32(image_id)` instead of `seed + hash(image_id)`. | Python's `hash()` of strings changes every run (hash randomisation), so results would not be reproducible. |
| 2026-11-19 | D3 | L channel is always scaled to 0-1 in features, selector and `adaptive_gamma`. | OpenCV 8-bit LAB gives L in 0-255; the PID thresholds (e.g. mean L ≥ 0.40) assume 0-1. |
| 2026-11-19 | D4 | `adaptive_gamma` clamps mean L to at least 1/255 before `log`. | All-black image would give `log(0) = -inf`. |
| 2026-11-19 | D5 | mIoU stored as 0-1. Selector "within 0.5 mIoU points of none" = difference < 0.005. A class is NaN only if it has no GT and no predicted pixels; a class predicted but absent in GT gets IoU 0 (standard Cityscapes). Classes averaged are recorded per experiment. | PID wording was ambiguous. |
| 2026-11-19 | D6 | Add `adaptive_gamma` `target_mean ∈ {0.35, 0.45, 0.55}` to the E3 sweep. | E1 uses "best params from E3" for adaptive_gamma, but E3 did not sweep it. |
| 2026-11-19 | D7 | Laptop has an NVIDIA RTX 3060 Laptop GPU (6 GB) and 7.4 GB RAM. Full runs may be done locally with a CUDA PyTorch build instead of (or as well as) Colab. To be confirmed at M3. | Hardware check in M0. |
| 2026-11-19 | D8 | PyTorch/torchvision are **not** in `requirements.txt`; they are installed first from pytorch.org (CPU or CUDA build), as PID 6 step 3 says. Their tested versions will be recorded in `results/ENVIRONMENT.md` at M3. | The right wheel depends on CPU vs CUDA. |
| 2026-11-19 | D9 | `requirements.txt` pins the latest versions on PyPI today. Only numpy, pyyaml, pytest and ruff are installed and tested in M0; the rest are installed and verified when first needed (M2/M3/M6) and re-pinned if anything clashes. | Keep the laptop install light in M0. |
| 2026-11-19 | D10 | `.gitignore` also ignores `results/**/cm/`, `*.npz`, `*.npy`, `run.log`, `*.zip`. | E2 alone produces ~18,000 confusion-matrix files; CSV + `meta.json` are enough in Git, and matrices can be regenerated. |
| 2026-11-19 | D11 | If SegFormer-B0 scores below 65% mIoU on Cityscapes val at short side 512, also test at 1024 before declaring a bug. | The `-1024-1024` checkpoint may lose more at 512 with whole-image inference than the paper's protocol. |
| 2026-11-19 | D12 | Package is installed in editable mode (`pip install -e . --no-deps`) and pytest also has `pythonpath = ["src"]`. | So `python -m lowlight.cli` works from anywhere and tests work even before installing. |
| 2026-11-19 | D13 | Windows long-path support is OFF on this laptop. Keep the repo at a short path; README warns about it. | A fresh-venv test in a long temp folder failed installing NumPy (path > 260 characters). |
| 2026-11-19 | D14 | `ruff format` excludes `docs/` (Markdown code blocks). | PID.md must stay word for word; explanation snippets are hand-aligned for teaching. |
| 2026-11-19 | D15 | **Cloud-only workflow.** Datasets, model weights and all model runs (dev and full) live on Google Colab + our private Google Drive. The laptop only holds the code and a light venv (no PyTorch, no datasets). Replaces D7; D8 now means PyTorch is only installed on Colab. | Keep data and heavy runs off the laptop. |
| 2026-11-19 | D16 | **No GitHub at first** (no repo, no push). Code reaches Colab another way (zip on Drive). Replaced by D41. | Keep the code private while it was being set up. |
| 2026-11-19 | D17 | README: no member/supervisor names until the end; other details filled in now. | Names are confirmed at the end. |
| 2026-11-19 | D18 | **Zero cost.** Only free tiers: Colab free (T4 GPU), Google Drive free 15 GB, Kaggle Notebooks as backup GPU, Hugging Face (no token needed). Supabase is not used: its free storage (about 1 GB, 50 MB per file, project pauses when idle) is far too small for the datasets and adds nothing Drive does not already do. No API keys are needed so far; if one ever is, it goes in Colab Secrets, never in code or Git. | Everything must be free and reliable. |
| 2026-11-19 | D19 | Cityscapes is downloaded **directly inside Colab** with the official `cityscapesScripts` downloader (`csDownload`, login via Colab Secrets); only the val split is kept and saved to Drive. ACDC: direct download in Colab if the site allows it, otherwise browser download then upload to Drive. | Keeps the 11 GB zip off the laptop and off Drive. |
| 2026-11-19 | D20 | Dataset accounts use a **personal** email (affiliation "GLA University"); GLA email only as a fallback if Cityscapes refuses. | University mail closes after graduation; data lives in personal Drive anyway. |
| 2026-11-19 | D21 | Code reaches Colab as `lowlight-code.zip` made by `scripts/pack_code.ps1` (`git archive`, commit id stored in the zip comment) and uploaded to Drive by hand. A private GitHub repo would be easier; switch later if needed. | GitHub not used yet (see D41). |
| 2026-11-19 | D22 | The check-data logic lives in `lowlight.datasets.check_datasets`; `scripts/check_data.py` and the CLI are thin wrappers (PID had the logic in the script). | One implementation, importable and unit-tested. |
| 2026-11-19 | D23 | Added `src/lowlight/data_prep.py` (not in the PID layout) for the cloud data workflow. | Needed for the cloud-only decision D15; keeps notebook cells short and the logic testable. |
| 2026-11-19 | D24 | Images are read with `np.fromfile` + `cv2.imdecode(IMREAD_UNCHANGED)`, then converted to RGB (gray -> RGB, BGRA -> RGB drop alpha, 16-bit -> top 8 bits). | `cv2.imread` fails on non-English Windows paths; one function handles every input type. |
| 2026-11-19 | D25 | opencv-python-headless 5.1.0.95 installed on the laptop. OpenCV 5 API checked: all functions used so far behave as in 4.x. | Needed for image tests (M1) and enhancement tests (M2). |
| 2026-11-19 | D26 | ACDC is downloaded automatically in Colab through the ACDC website's own JSON API (`https://acdc.vision.ee.ethz.ch/api`: login -> packages -> getPackageUri -> downloadPackage), with our own account from Colab Secrets. The code never accepts the licence automatically; it stops with a message if the licence is not yet accepted. Browser download + Drive upload stays as a fallback. Tested with a fake server and rehearsed; not yet tried against the real server (needs our account). | Less manual work. |
| 2026-11-19 | D27 | Only one Google account is used for the project Drive and Colab; which one is confirmed before the first real run. | Avoid putting project files in the wrong Drive. |
| 2026-11-19 | D28 | **NightCity added as a no-account real-night dataset** (Tan et al., IEEE TIP 2021; 4,297 night street images, public Google Drive links, non-commercial research terms), using the corrected **NightCity-fine** labels (Wei et al., ICCV 2023, DTP repo). Same rule as ACDC: tune on NightCity train, report on NightCity val. ACDC stays in the code; if we register later, results are reported on both (more evidence, not less). Folder layout and label format are not documented, so a flexible finder (`find_paired_tree`) detects them and the notebook saves `results/nightcity_layout.txt`; expected counts 2998/1299 from the paper, to be confirmed on the first run. | ACDC needs registration, which may take time. |
| 2026-11-19 | D29 | Cityscapes (controlled darkening, E2) has no equivalent no-account dataset with the same 19 classes. It stays planned; E2 waits until we register. Unofficial mirrors (e.g. on Hugging Face) are not used: the licences forbid redistribution. | Honest limit; does not block M2-M3. |
| 2026-11-19 | D30 | M2 started before the M1 checkpoint, (approved by the group). M1 is closed (tag `m1-done`) after the first real data check. | Data access is waiting on outside approval. |
| 2026-11-19 | D31 | Noise is estimated with Immerkær's method (numpy + OpenCV) instead of `skimage.restoration.estimate_sigma`. | No extra package on the laptop, fast enough for every image; tested to within 15% of the true sigma. |
| 2026-11-19 | D32 | Calibration compares night TRAIN brightness with Cityscapes val brightness (images only, no labels, no models). | Cityscapes val is the only daytime set we have; brightness statistics cannot leak accuracy into tuning. |
| 2026-11-19 | D33 | The per-method "L histogram before/after" plot is made in `viz.py` (M3); M2 provides the data (`features.l_histogram`) and the calibration histogram figure. | matplotlib is not installed on the laptop; plotting belongs with the other charts. |
| 2026-11-19 | D34 | `calibrate_darkness.py` defaults to `--night nightcity` (no account needed); `--night acdc_night` uses ACDC train as in the PID. | D28. |
| 2026-11-19 | D35 | `sliding_window=True` is not implemented yet (PID marks it optional); whole-image inference with short-side resize is used everywhere and recorded in metadata. | Keep one protocol; add only if the sanity check needs it. |
| 2026-11-19 | D36 | Mask2Former post-processing is done in our code (same formula as `post_process_semantic_segmentation`), with padding to a multiple of 32 and cropping the padding before upsampling. A slow test checks >98% agreement with the official function. | The official function would misalign non-padded/padded sizes; one consistent resize protocol for all models. |
| 2026-11-19 | D37 | Added `src/lowlight/evaluate.py` and `scripts/model_check.py` (not in the PID layout). Notebooks are generated from `scripts/build_notebooks.py`. | Shared inner loop for M3 checks and the M4 runner; notebooks reviewable as Python. |
| 2026-11-19 | D38 | On Colab only `transformers==5.23.0` is installed on top of Colab's preinstalled torch/numpy/OpenCV; the exact versions are printed and saved with results. The full pinned `requirements.txt` install is decided in M4. | Reinstalling numpy/OpenCV pins over Colab's packages risks breaking its torch build. |
| 2026-11-19 | D39 | Comparison grids and legends are drawn with OpenCV (no matplotlib); only charts use matplotlib (imported lazily). | Pictures can be made and tested on the laptop. |
| 2026-11-19 | D40 | GitHub repo `https://github.com/abhaykumardwivedi/Low-light-image-enhancement` added as remote `origin`. Datasets, weights and result images are never in Git (.gitignore), so a push contains only code, docs and small CSV/JSON. | Backup and easier Colab setup. |
| 2026-11-19 | D41 | The GitHub repo is **public** (replaces D16/D21). Colab can get the code with `git clone` instead of a zip. Git history was restarted from one clean commit. | Public repo is fine for the group; no token needed in Colab. |
