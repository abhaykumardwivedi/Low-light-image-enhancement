# `models.py`, `evaluate.py` and `viz.py`: running the models and seeing the results

## What they do
- `models.py` loads pretrained segmentation models from Hugging Face (SegFormer-B0, SegFormer-B2,
  Mask2Former Swin-Tiny) and gives them one common method, `predict(image) -> mask`. They are used
  **as-is**: nothing is trained or fine-tuned.
- `evaluate.py` runs one model over a dataset: darken (optional), enhance, predict, score each image.
- `viz.py` draws coloured masks, overlays, error maps (green right, red wrong) and comparison grids.

## Key idea and analogy
The models are like three different expert translators (SegFormer-B0 is the small, fast one). We do
not retrain them. We only change the **input** (the enhanced photo) and watch how their answers
change. To keep the comparison fair, every photo is prepared exactly the same way for every
translator: same size, same colour scaling.

## Walkthrough of the important lines

```python
scale = short_side / min(h, w)
size = (round(w * scale), round(h * scale))
```
The image is resized so its **shorter side** is 512 (or 1024 on GPU), keeping the 2:1 shape of street
photos. Hugging Face's default would squash every image into a square, which distorts cars and
people and lowers accuracy.

```python
x = resized.astype(np.float32) * self.rescale        # 0-255 -> 0-1
x = (x - self.mean) / self.std                       # the model's own normalisation
```
Models were trained on images normalised with specific mean and standard deviation values. We read
them from each model's own config file instead of typing them in, so they are always right.

```python
logits = self.model(pixel_values=x).logits            # SegFormer: 19 scores per pixel, 1/4 size
scores = F.interpolate(scores, size=(h, w), mode="bilinear")
pred = scores.argmax(dim=1)
```
SegFormer outputs 19 class scores per pixel at a quarter of the input size. We enlarge the
**scores** smoothly (bilinear) to the original size and only then choose the best class per pixel
(argmax). Enlarging the final class map instead would give blocky edges.

```python
probs = class_logits.softmax(dim=-1)[..., :-1]
scores = torch.einsum("bqc,bqhw->bchw", probs, masks.sigmoid())
```
Mask2Former works differently: it proposes 100 candidate regions ("queries"), each with a mask and
a class probability. The score of class c at a pixel adds up, over all queries, "how likely this
query is class c × how strongly the pixel belongs to its mask". We pad images to a multiple of 32
for its backbone and crop the padding away afterwards. A slow test checks that this matches
Hugging Face's own post-processing.

```python
self.remap = check_id2label(self.model.config.id2label, name)
return self.remap[pred]
```
At load time the model's class list is checked against our 19 classes. If the order were different,
`remap` would translate it.

```python
@functools.lru_cache(maxsize=4)
def load_segmenter(name, short_side, device): ...
```
Loading a model takes seconds, so each one is loaded once and reused.

## Viva questions
1. **Do you train the segmentation models?**
   No. They are pretrained on Cityscapes daytime images and used only for inference. The project
   asks whether changing the input image alone helps.
2. **Why resize by the shorter side instead of to a fixed square?**
   Street images are about twice as wide as tall. Squashing them into a square distorts shapes and
   lowers accuracy, and it would be unfair across images of different shapes.
3. **Why upsample the scores and not the final mask?**
   Interpolating 19 continuous scores and then picking the best class gives smooth, accurate object
   edges; enlarging a mask of class numbers gives blocky edges.
4. **How do SegFormer and Mask2Former differ?**
   SegFormer classifies every pixel directly (a light transformer encoder with a simple decoder).
   Mask2Former predicts a set of masks with a class for each, then combines them. Using both shows
   whether our findings depend on one architecture.
5. **How do you make sure the model's class order matches yours?**
   `check_id2label` compares the model's class names with our 19 classes when it is loaded and stops
   with an error if they do not match.
