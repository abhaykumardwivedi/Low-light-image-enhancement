# `darken.py` and `calibrate.py`: making controlled "night" images

## What it does
`darken.py` turns a well-lit Cityscapes image into a darker, noisier version at five known levels,
L0 (unchanged) to L4 (very dark). Because the scene and its labels stay the same, any drop in
segmentation accuracy is caused only by the lighting. `calibrate.py` adjusts the levels so that L3
and L4 match the brightness of **real** night photos.

## Key idea and analogy
It is like a controlled experiment in a lab: to test whether a medicine works, you change one thing
and keep everything else fixed. Here the "one thing" is the light level. Real night datasets
(ACDC, NightCity) are realistic but mix many changes (different scenes, headlights, blur), so the
controlled version and the real version answer different questions together.

## Walkthrough of the important lines

```python
d = alpha * x**g
```
`x` is the image scaled to 0–1. `alpha` (below 1) scales all brightness down. `g` (above 1) pushes
dark values down more than bright ones, so shadows lose contrast first, as in real low light.

```python
d = rng.poisson(d * peak) / peak
d = d + rng.normal(0.0, sigma, size=d.shape)
```
Real camera noise has two main parts. **Shot noise**: light arrives as individual photons, and their
count is random (Poisson distribution); with fewer photons (lower `peak`) the relative randomness is
larger, which is why dark photos look grainy. **Read noise**: the sensor electronics add a small
random value (Gaussian, standard deviation `sigma`). This "Poisson–Gaussian" model is standard in
low-light research.

```python
def image_seed(seed, image_id):
    return (seed + zlib.crc32(image_id.encode("utf-8"))) % 2**32
```
Each image gets its own fixed random seed, so the same image always gets exactly the same noise, on
any computer and any day. The PID suggested `hash(image_id)`, but Python changes string hashes every
time it starts (a security feature), which would make results impossible to repeat. CRC32 always
returns the same number (decision D2). A test fixes the exact value.

```python
if not bracketed:
    new["L3"]["alpha"] = fit_alpha(day_images, levels["L3"], night_mean * (1 + margin), seed)
    new["L4"]["alpha"] = fit_alpha(day_images, levels["L4"], night_mean * (1 - margin), seed)
```
Calibration measures the mean lightness of real night **train** images. If L3 is not brighter, and
L4 not darker, than real night, it searches for the `alpha` that puts L3 15% above and L4 15% below
the real night value. The search is a **bisection**: brightness grows with alpha, so we halve the
interval until we hit the target. Only brightness statistics are used, never labels or accuracy.

`noise=False` produces the "brightness only" ablation: comparing it with the noisy version shows how
much of the accuracy loss comes from darkness and how much from noise.

## Viva questions
1. **Why use synthetic darkening when real night datasets exist?**
   It gives a controlled experiment: same scene, same labels, only the lighting changes, at known
   levels. Real night images change many things at once, so you cannot tell what caused a drop.
2. **What are shot noise and read noise?**
   Shot noise comes from the random arrival of photons (Poisson); it is relatively stronger when
   there is less light. Read noise is a small random error added by the sensor electronics
   (Gaussian).
3. **How do you make sure the noise is the same every time?**
   Each image's random generator is seeded with the experiment seed plus a CRC32 checksum of its id,
   which is identical on every computer and every run.
4. **How did you choose the darkness parameters?**
   We started from values in the PID, then calibrated L3 and L4 so their mean lightness brackets
   that of real night training images.
5. **What is the noise-free ablation for?**
   It separates the two causes of accuracy loss, darkness and noise, which matters because HE and
   CLAHE amplify noise.
