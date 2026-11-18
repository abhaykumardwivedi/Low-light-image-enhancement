# Project setup (M0): virtual environment, Git, tests and linting

## What it is
Before any image code, M0 sets up the "workshop": a private Python environment, version control,
automatic tests and a style checker. None of this changes the results, but it makes the results
**reproducible** (anyone can get the same numbers) and **trustworthy** (bugs are caught early).

## Key idea and analogy
A virtual environment is like a separate toolbox for one project: the exact versions of NumPy,
PyTorch and so on are kept inside `.venv` and do not clash with other projects on the laptop.
Git is like a save-game system: every commit is a point you can go back to, and the tags
`m0-done`, `m1-done`, ... mark the approved checkpoints.

## The important files
| File | Why it exists |
|---|---|
| `.venv/` | The project's own Python 3.12 and packages. Not in Git. |
| `requirements.txt` | Exact package versions (`numpy==2.5.5`), so everyone installs the same thing. |
| `pyproject.toml` | Says "this is a package called `lowlight`, its code is in `src/`" and holds the pytest and ruff settings. |
| `.gitignore` | Lists what Git must never store: datasets (licence), model weights and result images (too big), `.venv`. |
| `docs/PROGRESS.md` | Where we are, what is next, and a dated log of every decision that differs from the PID. |
| `tests/` | Small programs that check the code gives known answers. `pytest` runs them all. |

Some lines worth knowing:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]          # tests can import lowlight even before it is installed
addopts = "-m 'not slow'"     # skip tests that download big models unless asked
```

```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "NPY"]   # style, real bugs, import order, NumPy pitfalls
```

`pip install -e . --no-deps` installs the project in "editable" mode: Python imports the code
straight from `src/lowlight`, so edits take effect without reinstalling.

**Why a `src/` layout?** The code lives in `src/lowlight/` rather than directly in the repository
root. This forces tests to import the installed package the same way users will, so a missing
file is caught by the tests instead of hidden.

**Why the project moved out of OneDrive:** OneDrive tries to upload every changed file. With a
virtual environment (thousands of files), about 25 GB of datasets and Git's internal files, that
is slow and can corrupt the Git repository.

## Viva questions
1. **Why do you pin exact package versions?**
   A new library version can change results or break code (for example, Transformers v5 changed
   some APIs). Pinning means anyone who installs from `requirements.txt` gets the same behaviour
   and the same numbers.
2. **Why are the datasets not in your Git repository?**
   They are very large, and their licences allow only non-commercial research use and forbid
   redistribution. `.gitignore` makes Git skip the `data/` folder.
3. **What does `pytest` do and why does it matter for a research project?**
   It runs small checks with known answers (for example, labelId 26 must become trainId 13, car).
   If a later change breaks something, a test fails at once, instead of us silently reporting
   wrong mIoU numbers.
4. **What is `ruff`?**
   A linter and formatter. It finds likely bugs (unused variables, risky patterns) and keeps the
   code style consistent, so the code is easier for everyone in the group to read.
5. **Why Python 3.12 and not the newest 3.14?**
   Deep learning libraries such as PyTorch and OpenCV publish ready-made packages for well-established
   Python versions first. 3.12 is fully supported by all of them, which avoids install problems.
