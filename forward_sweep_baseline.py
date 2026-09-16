"""
forward_sweep_baseline.py -- Forward simulation of the UNCONTROLLED SCM
misinformation model (no educational campaign, no cost functional).

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns" 
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

Integrates the state equations forward in time with the calibrated parameters to
show the natural (no-intervention) spread of a rumour, and reports the peak and
total misinformation used as the baseline for the optimal-control results.
Writes SCM_forward_sweep_baseline.png.

Usage: python3 forward_sweep_baseline.py
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# --- Calibrated parameters (pooled PINN estimate; mu = Lambda = 0) ----------
beta   = 0.1141     # transmission rate  (S -> M)
gamma  = 0.1056     # debunking rate     (M -> C)
psi    = 0.04       # forgetting rate    (C -> S)
theta  = 0.04       # forgetting rate    (M -> S)
d      = 0.0333     # disengagement / drop-out from M

# No-campaign baseline: the control and its effectiveness are all zero.
E       = 0.0       # educational-campaign intensity (OFF)
alpha_1 = 0.0       # campaign effect on S  (irrelevant since E = 0)
alpha_2 = 0.0       # campaign effect on M  (irrelevant since E = 0)
Lambda  = 0.0       # recruitment  (closed population)
mu      = 0.0       # natural exit (closed population)

# --- Simulation settings ----------------------------------------------------
T_FINAL  = 500                              # hours
N_POINTS = 1201                             # output resolution
INITIAL_CONDITION = [0.998, 0.001, 0.001]   # S0, C0, M0

# --- The SCM model (state order: S, C, M; corrected debunking sign) ---------
def scm_system(y, t):
    S, C, M = y
    dSdt = Lambda - (alpha_1*E*S) + (psi*C) - (beta*S*M) + (theta*M) - (mu*S)
    dCdt = (alpha_1*E*S) + (alpha_2*E*M) + (gamma*C*M) - (psi*C) - (mu*C)
    dMdt = (beta*S*M) - (alpha_2*E*M) - (gamma*C*M) - (theta*M) - ((d+mu)*M)
    return [dSdt, dCdt, dMdt]

# --- Integrate forward ------------------------------------------------------
t = np.linspace(0, T_FINAL, N_POINTS)
sol = scipy.integrate.odeint(scm_system, INITIAL_CONDITION, t)
S, C, M = sol[:, 0], sol[:, 1], sol[:, 2]

# --- Report -----------------------------------------------------------------
print(f"Peak misinformed  M : {M.max():.4f} at t = {t[M.argmax()]:.1f} h")
print(f"Peak crit.-literate C: {C.max():.4f} at t = {t[C.argmax()]:.1f} h")
print(f"Final (t={T_FINAL}h)  : S={S[-1]:.4f}  C={C[-1]:.4f}  M={M[-1]:.4f}")
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 renamed trapz
print(f"Total misinformation (area under M): {_trapz(M, t):.3f}")

# --- Plot -------------------------------------------------------------------
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18})
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='S (susceptible)',         color='tab:blue',  lw=2.5)
plt.plot(t, C, label='C (critically literate)', color='tab:green', lw=2.5)
plt.plot(t, M, label='M (misinformed)',         color='tab:red',   lw=2.5)
plt.xlabel('Time (hours)')
plt.ylabel('Population proportion')
plt.title(r'Uncontrolled SCM dynamics (no campaign, $E=0$)')
plt.xlim(0, T_FINAL); plt.ylim(0, 1.05)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('SCM_forward_sweep_baseline.png', dpi=300, bbox_inches='tight')
plt.show()
