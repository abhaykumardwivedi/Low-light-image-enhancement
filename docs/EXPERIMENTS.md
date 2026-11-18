# Experiments

What each experiment does and how to run it. The full design is in [PID Section 9.7](PID.md).
This file is filled in during M4 and M5, when the runner and configs exist.

| ID | Name | Data | Status |
|---|---|---|---|
| E1 | Real-night baseline | ACDC night val | not yet run |
| E2 | Controlled darkness | Cityscapes val, levels L0–L4 | not yet run |
| E3 | Parameter sweep | ACDC night train | not yet run |
| E4 | Class-wise analysis | reuses E1 and E2 | not yet run |
| E5 | Speed | 50 ACDC val images | not yet run |
| E6 | Adaptive selector | train on ACDC train, test on ACDC val | not yet run |
| E7 | Cross-dataset (stretch) | Dark Zurich val | not yet run |

Order: E3 first (it picks the parameters), then E1, E2, E4, E5, then E6 in M5.
