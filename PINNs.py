"""
PINNs.py -- Physics-Informed Neural Network (PINN) estimation of the natural-spread
parameters (beta, gamma, d) of the SCM misinformation model from real rumour cascades.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns" 
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

Pipeline position: run AFTER build_compartments.py (creates per-rumour S/C/M CSVs)
and pool_high_sdqc.py (creates the pooled curve). This script fits each CSV in
CSV_DIR and writes a parameter summary.

Method: for each rumour we train a network t -> (S, C, M) to (i) pass through the
observed C and M incidence and (ii) satisfy the SCM ODEs, learning beta, gamma, d
as trainable variables. Incidence (per-hour new counts), not cumulative totals, is
fit so the outflow rate d comes out physical (>= 0); see the comment at the fit.

Usage:
    python3 PINNs.py                 # full run over every CSV in CSV_DIR
    SMOKE=1 python3 PINNs.py         # quick sign check on the largest few datasets
    CSV_DIR=/path/to/CSVs python3 PINNs.py   # fit a different folder

Requires: deepxde (pytorch backend), torch, numpy, pandas.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"

import glob
import deepxde as dde
import numpy as np
import pandas as pd

# --- SMOKE MODE: quick check of parameter signs (esp. d) without the full run. ---
#     `SMOKE=1 python3 PINNs.py` trains only the largest few datasets for fewer
#     iterations. Leave unset for the full batch run.
SMOKE_MODE  = os.environ.get('SMOKE', '0') == '1'
SMOKE_N     = 2      # how many largest datasets to use in smoke mode
SMOKE_ITERS = 1500   # training iterations in smoke mode
FULL_ITERS  = 3000   # training iterations in a normal run

# --- FIXED PARAMETERS (historical PINN training; no campaign in the data) ---
Lambda  = 0.0    # closed-population assumption (no recruitment)
E       = 0.0    # no centralized educational campaign existed in the historical data
alpha_1 = 0.0    # campaign effect on S -- inert since E = 0
alpha_2 = 0.0    # campaign effect on M -- inert since E = 0
psi     = 0.04   # C -> S forgetting (24-hour attention fade)
theta   = 0.04   # M -> S forgetting (24-hour attention fade)
mu      = 0.0    # closed-population assumption: no natural exit (matches the paper's mu = 0)

# --- Data location (override with the CSV_DIR env var) ----------------------
# Default: the pooled curve from pool_high_sdqc.py. Point CSV_DIR at
# Data/CSVs_annotated to fit every individual thread instead.
BASE = os.environ.get(
    "MISINFO_BASE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
CSV_DIR = os.environ.get("CSV_DIR", f"{BASE}/Data/CSVs_pooled")

results = []

csv_files = glob.glob(f"{CSV_DIR}/*_rumor_*_data.csv")
if SMOKE_MODE:
    sizes = []
    for _f in csv_files:
        try:
            sizes.append((len(pd.read_csv(_f)), _f))
        except Exception:
            pass
    sizes.sort(reverse=True)
    csv_files = [f for _, f in sizes[:SMOKE_N]]
    print(f"[SMOKE] {SMOKE_ITERS} iters on the {len(csv_files)} largest dataset(s): "
          f"{[os.path.basename(f) for f in csv_files]}")
else:
    print(f"Found {len(csv_files)} datasets in {CSV_DIR}. Starting batch training...")

for file in csv_files:
    print("\n" + "=" * 45)
    print(f"Training PINN on: {os.path.basename(file)}")
    print("=" * 45)

    df = pd.read_csv(file)

    # Skip empty or improperly formatted datasets.
    if len(df) < 5:
        print("Dataset too small, skipping...")
        continue
    if 'M_cumulative' not in df.columns or 'C_cumulative' not in df.columns:
        print("Expected columns not found, skipping...")
        continue

    t_max = df['time_hour'].max()
    if t_max == 0:
        t_max = 1.0

    # --- Fit INCIDENCE (per-hour NEW counts), not cumulative totals. ---
    # Cumulative curves only ever rise, which forces the outflow rates (d, theta)
    # negative to keep the model's M(t) climbing. The hour-to-hour differences
    # rise AND fall like the model's prevalence M(t)/C(t), giving the outflow
    # terms real signal, so d can come out physical (>= 0).
    M_inc = df['M_cumulative'].diff().fillna(df['M_cumulative'].iloc[0]).clip(lower=0)
    C_inc = df['C_cumulative'].diff().fillna(df['C_cumulative'].iloc[0]).clip(lower=0)

    N_est = max(M_inc.max(), C_inc.max()) * 3
    if N_est == 0:
        N_est = 1.0

    # Normalize time to [0, 1] and counts by N_est; cast to float32 for PyTorch.
    t_data = (df['time_hour'].values / t_max).astype(np.float32).reshape(-1, 1)
    M_data = (M_inc.values / N_est).astype(np.float32).reshape(-1, 1)
    C_data = (C_inc.values / N_est).astype(np.float32).reshape(-1, 1)

    # 1. Fresh trainable parameters for this rumour.
    beta  = dde.Variable(0.1)
    gamma = dde.Variable(0.1)
    d     = dde.Variable(0.05)

    # 2. The SCM ODE residuals (time rescaled by t_max because t is normalized).
    def scm_ode(t, y):
        S, C, M = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        dS_dt = dde.grad.jacobian(y, t, i=0)
        dC_dt = dde.grad.jacobian(y, t, i=1)
        dM_dt = dde.grad.jacobian(y, t, i=2)

        res_S = dS_dt - t_max * (Lambda - (beta * S * M) - (alpha_1 * E * S) + (psi * C) + (theta * M) - (mu * S))
        res_C = dC_dt - t_max * ((alpha_1 * E * S) + (alpha_2 * E * M) + (gamma * C * M) - (psi * C) - (mu * C))
        res_M = dM_dt - t_max * ((beta * S * M) - (alpha_2 * E * M) - (gamma * C * M) - (theta * M) - ((d + mu) * M))

        return [res_S, res_C, res_M]

    # 3. Build and compile the model.
    geom = dde.geometry.TimeDomain(0, 1.0)
    observe_M = dde.icbc.PointSetBC(t_data, M_data, component=2)
    observe_C = dde.icbc.PointSetBC(t_data, C_data, component=1)

    data = dde.data.PDE(geom, scm_ode, [observe_M, observe_C],
                        num_domain=100, num_boundary=2, anchors=t_data)
    net = dde.nn.FNN([1] + [64] * 3 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, external_trainable_variables=[beta, gamma, d])

    # 4. Train.
    model.train(iterations=(SMOKE_ITERS if SMOKE_MODE else FULL_ITERS),
                display_every=1000)

    # 5. Read off the learned parameters.
    final_beta  = beta.detach().cpu().item()
    final_gamma = gamma.detach().cpu().item()
    final_d     = d.detach().cpu().item()

    print(f"Done. beta={final_beta:.4f}  gamma={final_gamma:.4f}  d={final_d:.4f}")

    results.append({'Dataset': os.path.basename(file),
                    'Beta': final_beta, 'Gamma': final_gamma, 'd': final_d})

# --- Export the parameter summary. ---
results_df = pd.DataFrame(results)
suffix = "_SMOKE" if SMOKE_MODE else ""
out_path = f"{CSV_DIR}/all_rumors_parameters_summary{suffix}.csv"
results_df.to_csv(out_path, index=False)
print(f"\nAll done. Parameter summary saved to {out_path}")
