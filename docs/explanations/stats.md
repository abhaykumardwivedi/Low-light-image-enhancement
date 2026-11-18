# `stats.py`: is a difference real or just luck?

## The two tests in one line each (PID Section 9.6)
- **Paired bootstrap 95% confidence interval:** we re-draw the test images with replacement 1,000
  times and recompute "mIoU with the method minus mIoU without it" each time; the middle 95% of these
  differences is the range the true difference very likely lies in.
- **Wilcoxon signed-rank test:** it checks whether the method raises or lowers the per-image mIoU more
  often and by more than chance would explain, without assuming the differences follow a bell curve.

We call a change **significant only when the 95% CI does not include zero.**

## Key idea and analogy
If you toss a coin 10 times and get 6 heads, the coin is probably still fair: small samples
wobble. Our 106 or 1,299 test images are a sample of all possible night scenes, so a +0.8 mIoU
point gain might be luck. The bootstrap answers "how much would this number wobble if we had
picked a different set of images of the same kind?"

## Walkthrough of the important lines

```python
weights = np.bincount(rng.integers(0, n, n), minlength=n)
diffs[b] = dataset_miou(cms_method, weights) - dataset_miou(cms_base, weights)
```
One bootstrap round draws N image numbers with replacement (some images twice, some not at all).
`weights[i]` says how often image i was drawn. The **same** draw is used for the method and for
`none`: that is what "paired" means. It matters because some images are simply harder than others,
and pairing cancels that out. Weighting the saved confusion matrices is much faster than copying
images.

```python
low, high = np.percentile(diffs, [2.5, 97.5])
```
The 2.5th and 97.5th percentiles of the 1,000 differences form the 95% interval. The random
generator has a fixed seed, so the interval is identical every time.

```python
return self.ci_low > 0 or self.ci_high < 0
```
Significant = the whole interval is above zero (a real gain) or below zero (a real loss).

**Why both tests?** They answer slightly different questions. The bootstrap is about the
**dataset-level mIoU** (the headline number, computed from summed matrices). Wilcoxon is about
**typical images**. Usually they agree; when they do not, both are reported.

A test builds a large "population" of 20,000 synthetic images with a known true difference, draws
150 of them as our "test set", and checks that the bootstrap interval contains the true difference.

## Viva questions
1. **What does a 95% confidence interval of [+0.4, +1.6] mIoU points mean?**
   If we repeated the study with other images of the same kind, the improvement would very likely
   (95%) lie between 0.4 and 1.6 points. Zero is outside, so the gain is significant.
2. **Why "paired"?**
   Both methods are evaluated on exactly the same images and the same resampled sets, so differences
   in image difficulty cancel out and small real effects become visible.
3. **Why not a t-test?**
   A t-test assumes the differences follow a normal distribution. Per-image mIoU differences are often
   skewed, with a few large outliers. The bootstrap and Wilcoxon make no such assumption.
4. **Can the result be significant but unimportant?**
   Yes. With 1,299 images even a 0.2-point change can be significant. That is why we always report
   the size of the difference and its interval, not only "significant: yes/no".
