# `enhance.py`: the enhancement methods

## What it does
It holds the classical methods we test before segmentation: **none** (baseline), **gamma**,
**HE** (histogram equalization), **CLAHE**, **gamma + CLAHE**, and **adaptive gamma**. Every method
takes an RGB uint8 image and returns a new one of the same size, and all of them are reached
through one function: `enhance(img, "clahe", clip_limit=2.0)`.

## Key idea and analogy
A night photo has most of its pixels crowded into the darkest few brightness values. Each method
spreads them out differently:
- **Gamma** is the "shadows" slider in a photo app: it lifts dark values a lot and bright values a
  little.
- **HE** re-spaces runners crowded at one end of a track so they use the whole track.
- **CLAHE** does HE separately for small tiles, with a limit so empty, noisy areas (like the night
  sky) are not blown up into grain.
- **Adaptive gamma** is phone auto-brightness: it measures how dark the image is and picks the
  gamma to match.

## Walkthrough of the important lines

```python
_REGISTRY: dict[str, EnhanceFn] = {}

def register(name):
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator

@register("clahe")
def clahe(img, clip_limit=2.0, tile_grid=8): ...
```
A **registry** (a dictionary from name to function) filled by a **decorator**. Experiments, the CLI
and the demo only need the method's name. Adding a new method is one decorated function; nothing
else changes.

```python
x = np.arange(256) / 255.0
lut = np.clip(np.rint(255.0 * x**gamma), 0, 255).astype(np.uint8)
return cv2.LUT(img, lut)
```
A uint8 pixel has only 256 possible values, so we compute `255 * (v/255) ** gamma` once for each
value (a **lookup table**). `cv2.LUT` then replaces every pixel by table lookup, about a millisecond
for a full image. With gamma = 0.5, value 25 becomes 80 but 200 only becomes 226.

```python
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
lab[..., 0] = fn(lab[..., 0])          # change only L (lightness)
return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```
HE and CLAHE are applied to the **L channel of LAB**, which holds brightness only, while A and B
hold colour. Equalizing R, G and B separately would stretch each colour differently and change
the hues (a red car could turn orange). A test checks that a gray ramp stays gray.

```python
if lightness.min() == lightness.max():
    return img.copy()
```
An all-black, all-white or single-colour image has no contrast to stretch, so it is returned
unchanged instead of producing an arbitrary result.

```python
m = min(max(m, 1.0 / 255.0), 1.0 - 1e-6)
gamma = log(target_mean) / log(m)
```
Adaptive gamma solves `m ** gamma = target_mean`, which moves the mean lightness to the target. `m`
is clamped so an all-black image does not cause `log(0)` (decision D4), and the result is clipped to
[0.25, 1.0] so it never over-brightens or darkens.

**Why gamma before CLAHE in `gamma_clahe`?** In a very dark image most pixels sit in a handful of
bins, and CLAHE can only re-spread the levels it is given. Gamma first lifts them into more distinct
levels, so CLAHE has real detail to work with.

## Viva questions
1. **Why do HE and CLAHE work on the L channel of LAB instead of on RGB?**
   L holds only brightness. Changing R, G and B separately changes their ratios, which shifts the
   colours; the model would then see unrealistic colours.
2. **What does the CLAHE clip limit do?**
   It caps how much any brightness level can be boosted within a tile. Without it, flat regions such
   as a dark sky would have their tiny noise differences stretched into strong grain.
3. **Why use a lookup table for gamma?**
   There are only 256 possible input values, so we compute each output once. Applying the table is
   one fast pass over the image, instead of a power function per pixel.
4. **What happens with gamma = 1?**
   Every value maps to itself, so the image is unchanged. A test checks this exactly.
5. **How is adaptive gamma different from fixed gamma?**
   Fixed gamma uses the same strength for every image. Adaptive gamma measures each image's mean
   lightness and picks the gamma that brings it to a target, so dark images get a stronger boost
   and bright images are left alone.
