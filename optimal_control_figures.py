"""
optimal_control_figures.py -- Regenerate the optimal-control result figures for the
paper: the misinformation-reduction comparison and the full state trajectories,
for both the linear (bang-bang) and quadratic (continuous) campaigns.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

The quadratic control is solved by a damped forward-backward sweep; the linear
(bang-bang) control by a one-dimensional switch-time search (there is no singular
arc -- see linear_fbsm.py). Prints the peak/total reductions, campaign effort,
misinformation averted, ACER and ICER, then writes:
    oc_control_comparison.png   (M with/without campaign + E(t), both cost types)
    oc_state_trajectories.png   (S, C, M under each optimal campaign)

Cost weights (A, W1, W2) are illustrative, not estimated from data; only the ratio
W/A affects the optimal control, and the qualitative structure is robust to it.

Usage: python3 optimal_control_figures.py
Requires: numpy, matplotlib.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Calibrated SCM parameters (pooled PINN; mu = Lambda = 0) ---
beta, gamma, psi, theta, d = 0.1141, 0.1056, 0.04, 0.04, 0.0333
a1, a2 = 0.1, 0.1                 # campaign effectiveness (alpha_1, alpha_2)

# --- Control problem settings (illustrative cost weights) ---
A = 1.0                           # weight on misinformation M(t)
W1 = 0.01                         # linear cost weight (bang-bang case)
W2 = 0.9                          # quadratic cost weight (continuous case)
EMAX = 1.0                        # normalized campaign bound, E in [0, 1]
T, N = 500.0, 500                 # horizon (hours) and time steps
dt = T / N
tg = np.arange(N + 1) * dt
S0, C0, M0 = 0.998, 0.001, 0.001  # initial condition


def trapz(y):
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * dt))


def forward(E):
    """Integrate the state forward (explicit Euler) under campaign schedule E."""
    S = np.zeros(N + 1); C = np.zeros(N + 1); M = np.zeros(N + 1)
    S[0], C[0], M[0] = S0, C0, M0
    for k in range(N):
        s, c, m, e = S[k], C[k], M[k], E[k]
        S[k + 1] = s + dt * (-beta*s*m - a1*e*s + psi*c + theta*m)
        C[k + 1] = c + dt * (a1*e*s + a2*e*m + gamma*c*m - psi*c)
        M[k + 1] = m + dt * (beta*s*m - a2*e*m - gamma*c*m - theta*m - d*m)
    return S, C, M


def backward(S, C, M, E):
    """Integrate the adjoints backward (from lambda(T) = 0)."""
    l1 = np.zeros(N + 1); l2 = np.zeros(N + 1); l3 = np.zeros(N + 1)
    for k in range(N, 0, -1):
        s, c, m, e = S[k], C[k], M[k], E[k]
        l1[k - 1] = l1[k] - dt * ((a1*e + beta*m)*l1[k] - a1*e*l2[k] - beta*m*l3[k])
        l2[k - 1] = l2[k] - dt * (-psi*l1[k] + (psi - gamma*m)*l2[k] + gamma*m*l3[k])
        l3[k - 1] = l3[k] - dt * (-A + (beta*s - theta)*l1[k] - (gamma*c + a2*e)*l2[k]
                                  - (beta*s - gamma*c - theta - a2*e - d)*l3[k])
    return l1, l2, l3


def Phi(S, C, M, l1, l2, l3):
    return a1 * S * (l1 - l2) + a2 * M * (l3 - l2)


# --- Uncontrolled baseline ---
Sb, Cb, Mb = forward(np.zeros(N + 1))
peakb = Mb.max(); totb = trapz(Mb)

# --- Quadratic (continuous) optimal control: damped forward-backward sweep ---
Eq = np.zeros(N + 1)
for _ in range(80000):
    S, C, M = forward(Eq); l1, l2, l3 = backward(S, C, M, Eq)
    En = np.clip(Phi(S, C, M, l1, l2, l3) / W2, 0, EMAX)
    Enew = 0.06 * En + 0.94 * Eq
    if np.max(np.abs(Enew - Eq)) < 1e-10:
        Eq = Enew; break
    Eq = Enew
Sq, Cq, Mq = forward(Eq)

# --- Linear (bang-bang) optimal control: single front-loaded switch search ---
best = (1e18, 0)
for ts in np.arange(0, T + dt, 1):
    E = np.where(tg < ts, EMAX, 0.0); S, C, M = forward(E)
    J = trapz(A*M + W1*E)
    if J < best[0]:
        best = (J, ts)
tsw = best[1]
El = np.where(tg < tsw, EMAX, 0.0); Sl, Cl, Ml = forward(El)


def stats(name, E, M):
    eff = trapz(E); av = totb - trapz(M)
    pr = 100 * (1 - M.max() / peakb); tr = 100 * (1 - trapz(M) / totb)
    acer = eff / av if av > 0 else float('nan')
    print(f"{name:10}: peakRed={pr:5.1f}%  totalRed={tr:5.1f}%  "
          f"effort={eff:6.2f}  averted={av:6.2f}  ACER={acer:.4f}")
    return dict(name=name, eff=eff, av=av, pr=pr, tr=tr, acer=acer)


print(f"baseline peakM={peakb:.4f} area={totb:.3f}  (W1={W1}, W2={W2}, Emax={EMAX})")
sB = stats("Bang-bang", El, Ml); sQ = stats("Quadratic", Eq, Mq)
lo, hi = (sB, sQ) if sB['av'] <= sQ['av'] else (sQ, sB)
icer = (hi['eff'] - lo['eff']) / (hi['av'] - lo['av'])
print(f"ICER ({lo['name']}->{hi['name']}) = {icer:.4f} effort per extra averted unit")
print(f"bang-bang switch time = {tsw:.0f} h ; quadratic peak E = {Eq.max():.3f}")

# ---------------- FIGURE 1: M reduction + control, 2 panels ----------------
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
cases = [("Linear cost (bang-bang control)", El, Ml, tsw),
         ("Quadratic cost (continuous control)", Eq, Mq, None)]
for ax, (title, E, M, ts) in zip(axes, cases):
    ax.plot(tg, Mb, color='#C0392B', ls='--', lw=2.2, label='$M(t)$ no campaign')
    ax.plot(tg, M, color='#1F4E79', lw=2.4, label='$M(t)$ with campaign')
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel('Time (hours)'); ax.set_xlim(0, T); ax.set_ylim(0, 0.14)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(tg, E, color='#548235', lw=2.0, label='campaign $E(t)$')
    ax2.set_ylim(0, 1.08); ax2.set_ylabel('campaign intensity $E(t)$', color='#3d5c25')
    ax2.tick_params(axis='y', colors='#3d5c25')
    h1, la1 = ax.get_legend_handles_labels(); h2, la2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, la1 + la2, loc='upper right', fontsize=9.5, framealpha=0.95)
axes[0].set_ylabel('misinformed fraction $M(t)$')
fig.suptitle(f'Optimal educational campaign vs. the uncontrolled outbreak '
             f'(illustrative weights $A{{=}}1,\\ W_1{{=}}{W1:g},\\ W_2{{=}}{W2:g}$)',
             fontsize=12.5, y=1.005)
fig.tight_layout()
fig.savefig('oc_control_comparison.png', dpi=150, bbox_inches='tight')
print("saved oc_control_comparison.png")

# ---------------- FIGURE 2: full S, C, M state trajectories ----------------
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, (title, S, C, M) in zip(
        axes2, [("Linear (bang-bang) control", Sl, Cl, Ml),
                ("Quadratic (continuous) control", Sq, Cq, Mq)]):
    ax.plot(tg, S, color='#1F4E79', lw=2.2, label='$S$ susceptible')
    ax.plot(tg, C, color='#548235', lw=2.2, label='$C$ critically-literate')
    ax.plot(tg, M, color='#C0392B', lw=2.2, label='$M$ misinformed')
    ax.set_title(title, fontsize=12.5); ax.set_xlabel('Time (hours)')
    ax.set_xlim(0, T); ax.set_ylim(0, 1.05); ax.grid(alpha=0.25)
    ax.legend(loc='center right', fontsize=9.5)
axes2[0].set_ylabel('population fraction')
fig2.suptitle('State trajectories under the optimal campaign (illustrative weights)',
              fontsize=12.5, y=1.005)
fig2.tight_layout()
fig2.savefig('oc_state_trajectories.png', dpi=150, bbox_inches='tight')
print("saved oc_state_trajectories.png")
