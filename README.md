# Optimal Control of an SIR Model of Misinformation through Educational Campaigns

Code for the paper *"Optimal Control of an SIR Model of Misinformation through
Educational Campaigns"* 

**Author:** Nehal Joshi  ·  **Mentor / co-author:** Dr. Padmanabhan Seshaiyer, George Mason University

## Overview

We model the spread of misinformation with an SIR-type compartmental model — the
**SCM model** — that splits a population into **S**usceptible, **C**ritically-literate,
and **M**isinformed individuals, and add a time-dependent **educational-campaign
control** `E(t)` that moves people from S and M into C. The natural-spread
parameters are calibrated to **real rumour cascades** from the PHEME Twitter dataset
using a **physics-informed neural network (PINN)**, and the best campaign is found
with **optimal control theory** (Pontryagin's Maximum Principle) under both a
quadratic (continuous) and a linear (bang-bang) cost.

**Headline results:** the calibrated rumour is viral (basic reproduction number
`R0 ≈ 1.56`); the optimal continuous campaign reduces peak misinformation by
**98.0%** and total spread by **98.4%** while never exceeding ~13% of maximum
intensity, and the bang-bang campaign reaches **98.8% / 99.7%** via a single
front-loaded full-intensity interval. A cost-effectiveness comparison shows the
continuous campaign is the more efficient of the two.

## Pipeline

Run the scripts in this order (each stage's output feeds the next):

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `build_compartments.py` | Turn raw PHEME threads into per-rumour S/C/M time-series CSVs (real veracity + SDQC stance labels). |
| 2 | `pool_high_sdqc.py` | Average the well-labelled threads into one pooled curve for stable fitting. |
| 3 | `PINNs.py` | Fit the SCM spread parameters `(β, γ, d)` to the CSVs with a PINN. |
| 4 | `Misinfo_Sensitivity_Analysis.py` | Sobol global sensitivity analysis of `M(t)`. |
| 5 | `forward_sweep_baseline.py` | Simulate the uncontrolled outbreak (the baseline). |
| 6 | `optimal_control_figures.py` | Solve both optimal campaigns and produce the result figures + reduction / ACER / ICER numbers. |
| — | `linear_fbsm.py` | Standalone check that the linear-cost control is bang-bang with **no singular arc**. |

Earlier, exploratory scripts (superseded classical fits, the pre-SDQC ETL, and
small helpers) are kept in [`exploratory/`](exploratory/) for provenance; they are
not part of the main pipeline.

## Setup

```bash
pip install -r requirements.txt
```

The scripts locate the project via the `MISINFO_BASE` environment variable, which
defaults to the parent of this folder. Data-reading scripts expect a `Data/`
directory laid out as `Data/all-rnr-annotated-threads/`,
`Data/rumoureval-2019-training-data/`, etc.

## Data

The PHEME and RumourEval data are **not** included in this repository (they are
large and separately licensed). Download them and place them under `Data/`:

- PHEME "all-rnr-annotated-threads": https://doi.org/10.6084/m9.figshare.6392078
- RumourEval-2019 training data (SDQC stance labels): SemEval-2019 Task 7.

## Requirements

Python 3.9+. See `requirements.txt` (numpy, scipy, pandas, matplotlib, SALib,
DeepXDE + PyTorch for the PINN). `Misinfo_Sensitivity_Analysis.py` uses SALib ≥ 1.4.

## License

License **to be confirmed** with the mentor / ASPIRE program before public release.
Until then, treat as all-rights-reserved.

## Notes

- The project's private `memory/` notes and the raw `Data/` files are intentionally
  outside this code folder — do not publish them.
- Cost weights (`A`, `W1`, `W2`) in the optimal-control scripts are illustrative;
  only the ratio `W/A` affects the optimal campaign.
