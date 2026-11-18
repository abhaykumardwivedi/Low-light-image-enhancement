# `cli.py`: the command-line interface

## What it does
It lets you run project tasks from a terminal with `python -m lowlight.cli <command>`. In M1 the
only command is `check-data`; `enhance`, `segment`, `run`, `analyse` and `report-assets` are added
in later milestones.

## Key idea and analogy
The CLI is the project's reception desk: you say what you want (`check-data --deep`), and it sends
the request to the right module. The real work stays in the modules (`datasets.py` here), so the
CLI, `scripts/check_data.py`, the notebooks and the tests all use **one** implementation.

## Walkthrough of the important lines

```python
sub = parser.add_subparsers(dest="command", required=True)
p = sub.add_parser("check-data", help="check dataset folders, counts and image/label pairs")
p.set_defaults(func=_cmd_check_data)
```
`argparse` (Python's standard library) builds one sub-command per task, with its own options and
automatic `--help` text. `set_defaults(func=...)` remembers which function handles the command.

```python
return 0 if all(r.passed for r in results) else 1
```
The **exit code** is 0 for success and 1 for failure. Scripts and notebooks can test it to stop a
long run early if the data is incomplete.

## Viva questions
1. **Why have a CLI at all if you have notebooks?**
   Commands are repeatable and can be copied exactly into the README and the report. Notebooks are
   for running in the cloud, but they call the same commands.
2. **What does the exit code mean?**
   0 means everything passed; any other number means something failed. It lets other programs react
   automatically.
3. **Where is the real logic of `check-data`?**
   In `lowlight/datasets.py` (`check_datasets`). The CLI only reads the options and prints the
   report, so the logic is tested once and reused everywhere.
