# Exploratory / superseded scripts

These scripts are kept for provenance but are **not** part of the main pipeline
(see the top-level `README.md`). They were early experiments or helpers that later
stages replaced:

- `rumor_hebdo_dataset.py` — the original ETL, before real veracity + SDQC stance
  labels were used. Superseded by `build_compartments.py`.
- `pooled_fit.py`, `quick_param_check.py` — classical (scipy) parameter fits.
  Found numerically unstable on these short, noisy curves; the PINN (`PINNs.py`)
  is used instead. Kept as a fast sanity check.
- `get_largest_rumor_dataset.py`, `peak_misinformed.py`, `virality.py` — small
  data-inspection / helper utilities used during exploration.

They may reference old paths or data layouts and are not maintained.
