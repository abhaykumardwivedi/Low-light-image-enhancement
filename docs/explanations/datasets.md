# `datasets.py`: finding and loading images and labels

## What it does
It finds every image of a dataset (ACDC night, Cityscapes val, Dark Zurich val, or any folder of
photos) together with its matching label file, and loads them as NumPy arrays when asked. It also
picks fixed small subsets for fast development runs, and checks that a downloaded dataset is
complete (`check-data`).

## Key idea and analogy
Think of a library catalogue. First we write a card for every book (a `Sample`: image path + label
path) without opening any book; that is fast even for thousands of files. A book is only taken
off the shelf (pixels read from disk) when someone actually asks for it (`dataset[i]`). This is
called **lazy loading**, and it keeps memory use low: we never hold 500 full-size images in RAM.

## Walkthrough of the important lines

```python
data = np.fromfile(path, dtype=np.uint8)
img = cv2.imdecode(data, flags)
```
`cv2.imread` cannot open files whose path contains non-English characters on Windows. Reading the
raw bytes with NumPy and letting OpenCV decode them from memory works everywhere.

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
OpenCV stores colours as Blue-Green-Red, most other libraries (and the models) expect
Red-Green-Blue. We convert **immediately on load**, so every image inside the project is RGB. If
we forgot this, red cars would look blue to the model, and accuracy would drop for no real reason.
A test writes a pure red image and checks it comes back red.

```python
for img_path in sorted(image_dir.glob(f"*/*{image_suffix}")):
    image_id = img_path.name[: -len(image_suffix)]
    label_path = label_dir / img_path.parent.name / f"{image_id}{label_suffix}"
```
ACDC stores `rgb_anon/night/val/GOPR0351/GOPR0351_frame_000159_rgb_anon.png` and its label at
`gt/night/val/GOPR0351/GOPR0351_frame_000159_gt_labelTrainIds.png`. We cut off the suffix to get the
**image id**, then build the label path from the same folder name and id. `sorted` makes the order
identical on every computer, which is needed for reproducible results.

```python
if split not in ("train", "val"):
    raise ValueError("ACDC split must be 'train' or 'val' (test has no labels)")
```
The code itself refuses the ACDC test split, so nobody can accidentally use it.

```python
rng = np.random.default_rng(seed)
chosen = np.sort(rng.choice(len(samples), size=limit, replace=False))
```
For development we use, say, 20 images instead of 400. A seeded random generator always picks the
**same** 20 (same seed, same list), so results from different days can be compared. `np.sort`
keeps them in the original order.

```python
bad = np.setdiff1d(np.unique(label), [*range(NUM_CLASSES), IGNORE_INDEX])
```
`check-data --deep` opens every label and lists any value that is not a valid trainId (0–18 or
255). This would catch, for example, a colour preview file used by mistake instead of the id mask.

## Viva questions
1. **Why do you convert BGR to RGB right after loading?**
   OpenCV loads colours in BGR order, but the segmentation models were trained on RGB. Converting
   once, at the boundary, means every function in the project can assume RGB.
2. **What does `check-data` verify?**
   That each dataset folder exists, the number of images matches the official count (400/106/500),
   every image has a label, image and label have the same size, and labels only contain valid
   class ids.
3. **Why load images lazily instead of all at once?**
   500 Cityscapes images at 2048×1024 take about 3 GB of RAM. Loading one at a time keeps memory
   low, which matters on free Colab.
4. **How do you make sure a 20-image development subset is always the same?**
   We use a NumPy random generator with a fixed seed from the config. The same seed always picks
   the same images.
5. **Why are Cityscapes labels converted but ACDC labels are not?**
   ACDC provides `labelTrainIds` files that already use 0–18/255. Cityscapes `gtFine_labelIds`
   files use the original 34 ids, so we convert them with the lookup table from `labels.py`.
