# `labels.py`: the 19 classes and label conversion

## What it does
It defines the 19 Cityscapes classes (road, sidewalk, ..., bicycle) in a fixed order, numbered
0–18 (these numbers are called **trainIds**; 255 means "ignore this pixel"). It gives each class
its standard colour, converts Cityscapes' original label numbers (labelIds) into trainIds, and
checks that every model uses the same class order as we do.

## Key idea and analogy
Different datasets and models are like people speaking different dialects for the same things.
Cityscapes' raw files call a car "26", our metrics expect "13", and a model might list its
classes in another order. `labels.py` is the **dictionary** that translates everything into one
common language, so that "13" always means "car" everywhere in the project. If the translation
is wrong, every mIoU number is wrong, which is why this module is tested carefully.

## Walkthrough of the important lines

```python
LABELID_TO_TRAINID = {7: 0, 8: 1, 11: 2, ..., 26: 13, ..., 33: 18}
```
Cityscapes files store about 34 detailed labels (labelIds). Only 19 are used for evaluation. For
example labelId 7 is road (trainId 0) and 26 is car (trainId 13). Everything else (ground,
license plate, ...) becomes 255 and is ignored when scoring.

```python
lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)   # 256 slots, all "ignore"
for label_id, train_id in LABELID_TO_TRAINID.items():
    lut[label_id] = train_id                        # fill in the 19 real classes
...
return LABELID_LUT[label_mask]
```
This is a **lookup table** (LUT). A uint8 pixel can only hold values 0–255, so a 256-entry array
covers every possible value. `LABELID_LUT[label_mask]` uses NumPy "fancy indexing": every pixel
value is replaced by the table entry at that position, for the whole 2048×1024 image in one fast
step, with no Python loop over pixels.

```python
VOC_TO_TRAINID = {person: (person, rider), car: (car,), ...}
```
torchvision's DeepLabV3 (stretch goal) knows the 20 Pascal VOC classes, not Cityscapes. Only 6
are shared. VOC has no "rider", so its "person" covers both our "person" and "rider". That is
why the mapping goes to a tuple of trainIds, and why DeepLabV3 must be reported in a separate
"mapped-class mIoU" table.

```python
def check_id2label(id2label, model_name):
    items = sorted((int(k), normalise_name(v)) for k, v in id2label.items())
    ...
    remap[k] = ours[name]
```
Each Hugging Face model stores its class list as `config.id2label`, for example `{"0": "road"}`.
Keys are strings in the JSON file, so we convert them to `int`. Names are normalised
(`"Traffic_Light"` → `"traffic light"`). The function returns `remap`, where
`remap[model_output_index] = our_trainId`. If the model's order already matches, `remap` is just
`[0, 1, ..., 18]`. If not, the difference is logged and the remap fixes it. If the model has a class
we do not know, it raises an error instead of guessing.

## Viva questions
1. **What is a trainId and why is 255 used?**
   A trainId is the class number 0–18 used for training and evaluation on Cityscapes. 255 marks
   pixels that should not be scored (unlabelled areas, ego-vehicle, rare classes), and the metrics
   skip them.
2. **Why use a lookup table instead of a loop?**
   A full Cityscapes image has about 2 million pixels. A Python loop would take seconds; a NumPy
   LUT does it in milliseconds, because the replacement runs in compiled code.
3. **What would happen if a model's class order did not match ours?**
   The scores would be meaningless (for example "sky" pixels counted as "road"). We check
   `id2label` when each model is loaded, and remap or stop with an error.
4. **Why can't DeepLabV3 be compared directly with SegFormer?**
   Its pretrained weights know only the 20 Pascal VOC classes. It cannot predict road, building,
   sky and so on, so it can only be scored on the 6 shared classes, in a separate table.
5. **Where do the class colours come from?**
   They are the standard Cityscapes palette, so our coloured masks look the same as in published
   papers and readers can compare them.
