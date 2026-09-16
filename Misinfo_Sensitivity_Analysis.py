"""
Misinfo_Sensitivity_Analysis.py -- Sobol global sensitivity analysis for the SCM
misinformation model.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns" 
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

Output of interest: M(t), the misinformed population, over 0-500 hours. Produces a
time-series of first-order (S1) and total-order (ST) Sobol indices for each
parameter, with 95% confidence bands, plus .npy arrays and two PNG figures.

Requires: SALib, numpy, scipy, matplotlib.
Usage: python3 Misinfo_Sensitivity_Analysis.py
"""
from SALib.analyze import sobol
try:                                    # SALib >= 1.4 (preferred)
    from SALib.sample.sobol import sample as sobol_sample
except Exception:                       # older SALib fallback
    from SALib.sample import saltelli
    sobol_sample = saltelli.sample

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Constants and the calibrated ("true") parameter values
# ----------------------------------------------------------------------------
INTERVAL = [0, 500]        # hours

# --- Campaign baseline E: READ THIS ----------------------------------------
# E is the educational-campaign CONTROL. In the model it appears ONLY as
# alpha_1*E*S and alpha_2*E*M, so:
#   * If E = 0, alpha_1 and alpha_2 have NO effect -> their sensitivity is 0.
#   * A small E > 0 keeps the campaign parameters "active" so they show up.
# The paper's remark says the UNCONTROLLED baseline uses E = 0. Pick the
# scenario this run represents and keep the paper consistent with it:
#   E_BASELINE = 0.0  -> uncontrolled baseline (alpha_1, alpha_2, E are inert)
#   E_BASELINE = 0.1  -> fixed modest campaign (all 8 parameters active)
E_BASELINE = 0.1
# ---------------------------------------------------------------------------

# Parameter order MUST match the scm_system() signature below:
#   alpha_1, E, psi, beta, theta, gamma, alpha_2, d
PARAMETER_NAMES   = ["alpha_1", "E", "psi", "beta", "theta", "gamma", "alpha_2", "d"]
PARAMETER_SYMBOLS = [r"$\alpha_1$", r"$E$", r"$\psi$", r"$\beta$",
                     r"$\theta$", r"$\gamma$", r"$\alpha_2$", r"$d$"]
TRUE_PARAMETER = [
    0.1,        # alpha_1  campaign effect S->C
    E_BASELINE, # E        campaign intensity (baseline)   <-- this slot is E
    0.04,       # psi      C->S forgetting
    0.1141,     # beta     transmission S->M      (pooled PINN)
    0.04,       # theta    M->S forgetting
    0.1056,     # gamma    debunking M->C         (pooled PINN, sign-corrected)
    0.1,        # alpha_2  campaign effect M->C
    0.0333,     # d        disengagement exit from M (pooled PINN)
]

INITIAL_CONDITION = [0.998, 0.001, 0.001]      # S0, C0, M0

# ----------------------------------------------------------------------------
# 2. The SCM model (mu = Lambda = 0; debunking sign convention)
# ----------------------------------------------------------------------------
def scm_system(y, t, alpha_1, E, psi, beta, theta, gamma, alpha_2, d):
    S, C, M = y
    dSdt = -(alpha_1*E*S) + (psi*C) - (beta*S*M) + (theta*M)
    dCdt =  (alpha_1*E*S) - (psi*C) + (gamma*C*M) + (alpha_2*E*M)
    dMdt =  (beta*S*M) - (gamma*C*M) - (theta*M) - (alpha_2*E*M) - (d*M)
    return [dSdt, dCdt, dMdt]

# ----------------------------------------------------------------------------
# 3. Sensitivity problem: vary each parameter +/- 25% around its true value
# ----------------------------------------------------------------------------
BOUND_FACTOR = 0.25
ZERO_UPPER   = 0.05        # bound for any parameter whose baseline is exactly 0
N_BASE       = 2**10       # Saltelli base size. Total model runs = N_BASE*(2k+2).
                           # 2**6 is too small and gives noisy, sometimes-negative
                           # indices; 2**10 is a good default (lower to 2**8 if slow).

bounds = []
for value in TRUE_PARAMETER:
    if value == 0:
        bounds.append([0.0, ZERO_UPPER])
    else:
        bounds.append([value*(1 - BOUND_FACTOR), value*(1 + BOUND_FACTOR)])

problem = {"num_vars": len(PARAMETER_NAMES),
           "names": PARAMETER_NAMES,          # plain names for SALib
           "bounds": bounds}

param_values = sobol_sample(problem, N_BASE, calc_second_order=True)

# ----------------------------------------------------------------------------
# 4. Run the model ONCE per sample over [0, 500]; read M at each hour.
#    (One adaptive integration replaces an hour-by-hour loop -- same result,
#     much faster.)
# ----------------------------------------------------------------------------
t_eval = np.arange(0, INTERVAL[1] + 1)          # 0, 1, ..., 500
hours  = t_eval[1:]                              # 1, ..., 500
M_output = np.zeros((len(hours), param_values.shape[0]))

for s, params in enumerate(param_values):
    sol = scipy.integrate.odeint(scm_system, INITIAL_CONDITION, t_eval,
                                 args=tuple(params), rtol=1e-8, atol=1e-10)
    M_output[:, s] = np.clip(sol[1:, 2], 0.0, None)   # M at hours 1..500

# ----------------------------------------------------------------------------
# 5. Sobol analysis at each hour -> time series of S1 and ST (+ 95% conf.)
# ----------------------------------------------------------------------------
k = len(PARAMETER_NAMES)
S1  = np.full((len(hours), k), np.nan); S1c = np.zeros((len(hours), k))
ST  = np.full((len(hours), k), np.nan); STc = np.zeros((len(hours), k))

for h in range(len(hours)):
    if np.ptp(M_output[h]) < 1e-15:      # output ~constant -> indices undefined
        continue
    res = sobol.analyze(problem, M_output[h], calc_second_order=True,
                        print_to_console=False)
    S1[h], S1c[h] = res["S1"], res["S1_conf"]
    ST[h], STc[h] = res["ST"], res["ST_conf"]

# clip ONLY for display; keep raw arrays (they carry the real noise)
S1_disp = np.clip(np.nan_to_num(S1), 0, 1)
ST_disp = np.clip(np.nan_to_num(ST), 0, 1)

np.save("t_scm.npy", hours)
np.save("Si_M_F.npy", S1);  np.save("Si_M_F_conf.npy", S1c)
np.save("Si_M_T.npy", ST);  np.save("Si_M_T_conf.npy", STc)
np.save("PARAMETER_SYMBOLS_scm.npy", np.array(PARAMETER_SYMBOLS, dtype=object))

# ----------------------------------------------------------------------------
# 6. Plot (with 95% confidence bands so the noise is honest)
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 22, "axes.labelsize": 28,
                     "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "legend.fontsize": 17})
COLORS = ["red", "gold", "green", "blue", "purple", "orange", "brown", "pink",
          "gray", "cyan", "olive", "black"]
STYLES = ["-", "--", "-.", ":"]
def style(j):
    return COLORS[j % len(COLORS)], STYLES[(j // len(COLORS)) % len(STYLES)]

def plot_indices(values, conf, ylabel, fname):
    fig, ax = plt.subplots(figsize=(15, 9))
    for j in range(k):
        c, ls = style(j)
        ax.plot(hours, values[:, j], color=c, linestyle=ls, linewidth=3,
                label=PARAMETER_SYMBOLS[j])
        ax.fill_between(hours, values[:, j]-conf[:, j], values[:, j]+conf[:, j],
                        color=c, alpha=0.15)      # 95% confidence band
    ax.set_ylim([-0.1, 1.1]); ax.set_xlim(INTERVAL)
    ax.set_ylabel(ylabel, fontsize=28); ax.set_xlabel("Time (Hours)", fontsize=28)
    ax.legend(loc="best", fontsize=17, ncol=2)
    fig.tight_layout()
    plt.savefig(fname, dpi=600, bbox_inches="tight")
    plt.show(); plt.close()

plot_indices(S1_disp, S1c, "First-order Sensitivity Indices",
             "SCM_Sensitivity_First_order_M_t0_500.png")
plot_indices(ST_disp, STc, "Total-order Sensitivity Indices",
             "SCM_Sensitivity_Total_order_M_t0_500.png")

# quick text summary at the final hour
last = -1
order = np.argsort(-ST_disp[last])
print(f"Parameter influence on M at t = {hours[last]} h "
      f"(E_BASELINE = {E_BASELINE}, N_BASE = {N_BASE}):")
for j in order:
    inter = max(ST_disp[last, j] - S1_disp[last, j], 0.0)
    print(f"  {PARAMETER_NAMES[j]:8s}  S1={S1_disp[last,j]:.3f}  "
          f"ST={ST_disp[last,j]:.3f}  interaction(ST-S1)={inter:.3f}")
