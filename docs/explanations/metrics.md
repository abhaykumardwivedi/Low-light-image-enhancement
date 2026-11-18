# `metrics.py`: how segmentation is scored

## What it does
It compares a predicted mask with the ground-truth mask and produces IoU, mIoU, Dice, pixel accuracy
and class accuracy. Everything is computed from a **confusion matrix**: a 19 × 19 table counting how
many pixels of each true class were predicted as each class. One matrix is saved per image, so any
score can be recomputed later without running the models again.

## Key idea and analogy
Think of a teacher marking an exam where every pixel is a question with 19 possible answers. The
confusion matrix is the mark sheet: row = correct answer, column = the student's answer. The
diagonal is "right"; everything else shows which classes get confused (for example "sidewalk"
answered as "road").

## Walkthrough of the important lines

```python
valid = gt != ignore_index
counts = np.bincount(g * num_classes + p, minlength=num_classes * num_classes)
return counts.reshape(num_classes, num_classes)
```
Ignored pixels (255) are removed first. Then each pixel's (true, predicted) pair is turned into
**one** number, `true * 19 + predicted`, which is unique for every pair (for example true 13 and
predicted 0 gives 247). `np.bincount` counts how often each number occurs, all in compiled code with
no Python loop, and reshaping the 361 counts to 19 × 19 gives the matrix.

```python
tp = np.diag(cm)                    # correct pixels of each class
fp = cm.sum(axis=0) - tp            # predicted as the class, but wrong
fn = cm.sum(axis=1) - tp            # really the class, but missed
iou = tp / (tp + fp + fn)
dice = 2 * tp / (2 * tp + fp + fn)
```
**IoU** (Intersection over Union) = overlap between predicted and true area divided by their combined
area. **Dice** is similar but counts the overlap twice, so it is always at least as high as IoU.
**mIoU** is the mean IoU over classes.

```python
np.divide(num, den, out=out, where=den > 0)   # NaN where a class never appears
```
If a class is neither in the ground truth nor predicted (no buses in the image), its IoU is
undefined (0/0). It becomes NaN and is left out of the mean. A class that is predicted but not
present gets IoU 0: a pure false alarm should lower the score (decision D5).

**Dataset scores use the summed matrices**, not the average of per-image scores. A test shows why:
a big image that is perfect and a 1-pixel image that is wrong give pixel accuracy 100/101, not 50%.

## Viva questions
1. **What is IoU and why is it better than pixel accuracy?**
   IoU = overlap / union for one class. Pixel accuracy is dominated by large classes such as road
   and sky: a model that never finds a single person can still get 90% pixel accuracy. mIoU gives
   every class equal weight.
2. **How do you compute the confusion matrix without loops?**
   Each valid pixel becomes the single number true × 19 + predicted; `np.bincount` counts them and
   the result is reshaped to 19 × 19.
3. **What happens to classes that do not appear in an image?**
   If a class is neither present nor predicted, its IoU is NaN and is excluded from the mean. If it
   is predicted but not present, its IoU is 0.
4. **Why sum confusion matrices instead of averaging per-image mIoU?**
   It is the standard Cityscapes protocol: every pixel counts equally, and tiny images or images
   where a class covers a few pixels cannot swing the result.
5. **Why save one confusion matrix per image?**
   Any metric, any subset of images and the bootstrap statistics can be recomputed later from these
   small files, without running the models again.
