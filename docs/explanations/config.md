# `config.py`: loading and checking settings

## What it does
All settings (random seed, device, image size, folder paths, enhancement parameters) live in
YAML files in `configs/`, not inside the Python code. `config.py` reads `default.yaml`, merges an
experiment file and any command-line overrides on top, and checks that every value is sensible
before anything runs. It also loads the five darkness levels L0–L4.

## Key idea and analogy
Think of `default.yaml` as a restaurant's standard recipe and an experiment file as a customer's
special request: "same as usual, but extra spicy". The experiment file only lists what changes;
everything else comes from the default. Validation is the chef checking the order makes sense
before cooking, so we never waste an hour-long run because of a typo like `device: gpu`.

## Walkthrough of the important lines

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
```
`__file__` is `src/lowlight/config.py`. Going up two folders gives the repository root, so paths
like `data/acdc` work no matter which folder you run the command from. `pathlib.Path` builds
correct paths on both Windows (`\`) and Linux (`/`).

```python
data = yaml.safe_load(f)
```
`safe_load` reads only plain data (numbers, text, lists, dicts). The plain `yaml.load` can be
tricked into running code, so we never use it.

```python
def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)   # merge nested sections key by key
        else:
            result[key] = copy.deepcopy(value)             # anything else simply replaces
```
This is **recursive**: if both sides have a section (such as `enhance_defaults`), it merges inside
that section too. So an experiment can change only `gamma` and still keep the default CLAHE
settings. `deepcopy` makes sure the original dicts are never changed by accident.

```python
if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
```
In Python, `True` counts as the integer 1, so `seed: true` would wrongly pass a plain `int` check.
That is why `bool` is excluded explicitly.

```python
if (p.get("peak") is None) != (p.get("sigma") is None):
    raise ConfigError(...)
```
For a darkness level, noise needs both a photon count (`peak`) and a read-noise level (`sigma`).
This line says "either both are set or both are empty", like an XOR check.

## Viva questions
1. **Why keep settings in YAML files instead of in the code?**
   So experiments can be changed and repeated without editing code, and the exact settings used
   can be saved next to the results (`meta.json`) for reproducibility.
2. **What happens if an experiment file only sets `inference_short_side`?**
   `deep_merge` takes every other setting from `default.yaml`; only that one value changes.
3. **Why validate the config at the start?**
   A wrong value (for example `device: gpu` instead of `cuda`) would otherwise crash or, worse,
   silently do the wrong thing hours into a run. Checking first fails fast with a clear message.
4. **Why `yaml.safe_load` and not `yaml.load`?**
   `safe_load` only creates plain data types; `load` can create arbitrary Python objects, which is
   a security risk if a config file is ever untrusted.
5. **Why use `pathlib` instead of writing paths as strings?**
   Windows uses `\` and Linux/Colab use `/`. `pathlib` handles both, so the same code runs on the
   laptop and on Colab.
