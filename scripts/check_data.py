"""Check dataset folders, image counts and image/label pairing (PID Section 8).

Same as `python -m lowlight.cli check-data`; the logic lives in lowlight.datasets so that the
CLI, this script and the tests all use one implementation.

Usage:
    python scripts/check_data.py            # quick check (reads 3 images per split)
    python scripts/check_data.py --deep     # read every image and label (use on Colab)
"""

import sys

from lowlight.cli import main

if __name__ == "__main__":
    sys.exit(main(["check-data", *sys.argv[1:]]))
