# `data_prep.py` and the Colab data notebook: getting data into the cloud

## What it does
The official dataset zips are huge and contain much more than we use (all ACDC weather types, all
Cityscapes splits, colour previews, polygon files). `data_prep.py` extracts **only** the files the
project needs, packs them into small zips stored in your private Google Drive, and unpacks them at
the start of each Colab session. `notebooks/colab_data_setup.ipynb` runs these steps on Colab.

## Key idea and analogy
It is like moving house: instead of carrying the whole warehouse (11 GB Cityscapes zip), we pick
the few boxes we need (500 val images, about 1 GB) and put them in a small storage unit (Google
Drive). Every work day (Colab session) we copy those boxes to the work desk (Colab's fast local
disk), because working directly from the storage unit (reading thousands of files from Drive) is
very slow.

## Walkthrough of the important lines

```python
re.compile(r"(?:^|/)(rgb_anon/night/(?:train|val)/[^/]+/[^/]+_rgb_anon\.png)$")
```
A **regular expression** that matches only ACDC night train/val images. `(?:train|val)` excludes
`test` and `fog/rain/snow`. The part in brackets `( ... )` is "captured": it is the path we keep.
Anything before it (for example an extra `ACDC/` folder inside the zip) is dropped, so the output
always has the layout the loaders expect.

```python
if target.is_file() and target.stat().st_size == info.file_size:
    continue
```
If a file was already extracted (same size), skip it. Free Colab can disconnect; re-running the
cell then continues instead of starting again.

```python
if p.is_absolute() or ".." in p.parts:
    raise ValueError(f"Unsafe path inside zip: {rel!r}")
```
A badly made or malicious zip could contain `../../something`, which would write outside the target
folder ("zip slip"). We refuse such paths. A test checks this.

```python
zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED)
```
PNG images are already compressed, so compressing them again wastes time and saves almost nothing.
`ZIP_STORED` just packs them.

```python
cs_user = userdata.get("CITYSCAPES_USERNAME")
```
In the notebook, the Cityscapes password comes from **Colab Secrets**, not from the code. It is
never printed, never saved in the notebook and never stored in Git.

**Why the code reaches Colab as a zip:** `scripts/pack_code.ps1` runs `git archive`, which packs only
committed files and writes the **commit id** into the zip. The notebook reads it back and saves it
with the results, so every number can be traced to the exact code version that made it.

## Viva questions
1. **Why not store the full datasets on Google Drive?**
   Free Drive has 15 GB, and the Cityscapes image zip alone is 11 GB. We keep only the val split and
   only ACDC night, which together take a few GB.
2. **Why copy data from Drive to Colab's local disk before using it?**
   Drive is a network file system; opening thousands of small files from it is very slow. One big
   zip copies fast, and unzipping on local disk is fast.
3. **How is the Cityscapes password kept safe?**
   It is stored in Colab Secrets and read with `userdata.get`. It never appears in the notebook,
   the code, the outputs or Git.
4. **What is a "zip slip" and how do you prevent it?**
   A zip entry with a path like `../../x` could overwrite files outside the target folder. We check
   every path and refuse absolute paths or paths containing `..`.
5. **How do you know which code version produced a result?**
   `git archive` writes the commit id into the code zip; the notebook reads it and saves it with the
   results (`colab_env_*.json`, and later `meta.json`).
