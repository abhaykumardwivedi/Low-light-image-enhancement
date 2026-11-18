# `features.py`: measuring an image before enhancing it

## What it does
It computes nine cheap numbers that describe an input image: mean, median and spread of lightness,
the share of very dark and very bright pixels, contrast, entropy (how much detail), a noise estimate,
and the share of light-source pixels (street lamps, headlights). The adaptive selector (M5) uses
them to decide which enhancement to apply, and calibration uses mean lightness.

## Key idea and analogy
A doctor takes your temperature and blood pressure before prescribing anything. These features are
the image's "vital signs": quick to measure, and enough to decide whether it needs a strong boost,
a gentle one, or nothing.

## Walkthrough of the important lines

```python
lab[..., 0].astype(np.float32) / 255.0
```
All lightness features use a 0–1 scale. OpenCV stores L as 0–255 in 8-bit images (not the textbook
0–100), so dividing by 255 makes thresholds like "mean L ≥ 0.40" mean the same thing everywhere
(decision D3).

```python
p = p[p > 0]
return float(-(p * np.log2(p)).sum())
```
**Shannon entropy** of the brightness histogram, in bits. One flat value gives 0 bits; two equally
common values give 1 bit; all 256 values equally common give 8 bits. Night images often have low
entropy: few distinct levels, so little visible detail.

```python
kernel = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
sigma = sqrt(pi / 2) * sum(|I * kernel|) / (6 * (W - 2) * (H - 2))
```
**Immerkær's noise estimate** (1996). The kernel cancels smooth brightness changes, so what is left
is mostly noise; the formula converts the average leftover into a standard deviation. It is fast
and needs no extra library (decision D31). A test adds noise of known strength (0.02 and 0.05) and
checks the estimate is within 15%.

## Viva questions
1. **Why compute features on the input image and not the enhanced one?**
   The selector must decide which enhancement to apply before applying any, so it can only look at
   the original.
2. **What does entropy tell you about an image?**
   How many different brightness levels are used, roughly how much detail is visible. A very dark
   image has low entropy.
3. **Why estimate noise at all?**
   HE and CLAHE amplify noise. If an image is already noisy, the selector should prefer a gentler
   method such as adaptive gamma.
4. **Why measure light sources separately?**
   Street lamps and headlights are saturated bright spots in otherwise dark images. They raise the
   mean brightness without the rest of the scene being visible, which can mislead a rule based on
   the mean alone.
