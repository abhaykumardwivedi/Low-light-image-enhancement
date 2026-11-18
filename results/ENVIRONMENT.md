# Environment

Updated whenever packages are added or changed. Each experiment's `meta.json` also records the
library versions it actually used.

## Machine (development laptop)
- OS: Windows 11 Home Single Language (build 10.0.26300)
- CPU: AMD Ryzen 7 5800H (8 cores, 16 threads)
- RAM: 7.4 GB usable
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (plus AMD Radeon integrated graphics)
- Python: 3.12.10 (python.org build, installed with winget)

## Installed and tested so far (M0)
| Package | Version |
|---|---|
| numpy | 2.5.5 |
| PyYAML | 6.0.3 |
| opencv-python-headless | 5.1.0.95 (installed in M1) |
| pytest | 9.2.0 |
| ruff | 0.17.5 |

## Pinned but not yet installed
See `requirements.txt`. PyTorch and torchvision: installed only on Google Colab (decision D15). The Colab versions are recorded here after the first Colab run.
