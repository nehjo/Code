"""
linear_fbsm.py  --  Linear-cost (bang-bang) optimal control for the SCM model,
and a numerical test for whether a SINGULAR ARC occurs at the calibrated parameters.

Problem:  minimize  J = integral_0^T ( A*M(t) + W1*E(t) ) dt
subject to the SCM dynamics, with 0 <= E(t) <= Emax.

Pontryagin gives a bang-bang law with switching function  sigma(t) = W1 - Phi(t),
where  Phi(t) = a1*S*(l1-l2) + a2*M*(l3-l2)   (l1,l2,l3 = adjoints for S,C,M).
  E* = Emax  where Phi > W1  (sigma < 0)
  E* = 0     where Phi < W1  (sigma > 0)
  singular   where Phi = W1 on an interval.

This script (a) solves the control two independent ways -- a smoothed forward-backward
sweep and a direct switch-time optimization -- and (b) checks the singular case by
looking at the zeros of sigma along the optimal trajectory.

Conclusion for the pooled calibration: NO singular arc. sigma has only simple
(transversal) zeros, so the control is pure bang-bang.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns" 
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.
Cost weights A, W1, Emax are illustrative.

Usage: python3 linear_fbsm.py
"""
import numpy as np

# ---- pooled calibration (Table: final params) ----
beta, gamma, psi, theta, d = 0.1141, 0.1056, 0.04, 0.04, 0.0333
a1, a2 = 0.1, 0.1
# ---- cost / horizon (illustrative) ----
A      = 1.0        # weight on misinformation M(t)
W1     = 0.2        # linear weight on the campaign E(t)
Emax   = 0.05       # max campaign intensity when "on"
T, N   = 500.0, 100
dt     = T / N
S0, C0, M0 = 0.998, 0.001, 0.001
tg     = np.arange(N + 1) * dt

def trapz(y):
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * dt))

def forward(E):
    S = np.zeros(N+1); C = np.zeros(N+1); M = np.zeros(N+1)
    S[0], C[0], M[0] = S0, C0, M0
    for k in range(N):
        s, c, m, e = S[k], C[k], M[k], E[k]
        S[k+1] = s + dt*(-beta*s*m - a1*e*s + psi*c + theta*m)
        C[k+1] = c + dt*( a1*e*s + a2*e*m + gamma*c*m - psi*c)
        M[k+1] = m + dt*( beta*s*m - a2*e*m - gamma*c*m - theta*m - d*m)
    return S, C, M

def backward(S, C, M, E):
    """Adjoints, integrated backward from lambda(T)=0."""
    l1 = np.zeros(N+1); l2 = np.zeros(N+1); l3 = np.zeros(N+1)
    for k in range(N, 0, -1):
        s, c, m, e = S[k], C[k], M[k], E[k]
        dl1 = (a1*e + beta*m)*l1[k] - a1*e*l2[k] - beta*m*l3[k]
        dl2 = -psi*l1[k] + (psi - gamma*m)*l2[k] + gamma*m*l3[k]
        dl3 = -A + (beta*s - theta)*l1[k] - (gamma*c + a2*e)*l2[k] \
              - (beta*s - gamma*c - theta - a2*e - d)*l3[k]
        l1[k-1] = l1[k] - dt*dl1; l2[k-1] = l2[k] - dt*dl2; l3[k-1] = l3[k] - dt*dl3
    return l1, l2, l3

def switching(S, C, M, l1, l2, l3):
    Phi = a1*S*(l1 - l2) + a2*M*(l3 - l2)
    return W1 - Phi, Phi

def Jcost(E):
    S, C, M = forward(E)
    return trapz(A*M + W1*E), S, C, M

def bang(*intervals):
    """Control equal to Emax on the given time intervals, else 0."""
    E = np.zeros(N+1)
    for a, b in intervals:
        E[(tg >= a) & (tg < b)] = Emax
    return E

def sweep_smoothed(taus=(0.5, 0.2, 0.08, 0.03, 0.012, 0.005, 0.002), c=0.4, iters=400):
    """Forward-backward sweep with a sigmoid-smoothed bang-bang update, annealed tau."""
    E = np.zeros(N+1)
    for tau in taus:
        for _ in range(iters):
            S, C, M = forward(E)
            l1, l2, l3 = backward(S, C, M, E)
            _, Phi = switching(S, C, M, l1, l2, l3)
            Enew = Emax / (1.0 + np.exp(-(Phi - W1)/tau))
            Eu = (1-c)*E + c*Enew
            if np.max(np.abs(Eu - E)) < 1e-9:
                E = Eu; break
            E = Eu
    return E

def optimal_switch_control():
    """Direct search over bang-bang structures: single switch on->off, or a pulse."""
    b1 = min(((Jcost(bang((0, ts)))[0], ('single', ts, None)) for ts in np.arange(0, T+dt, 2.5)))
    b2 = (1e18, None)
    for t1 in np.arange(0, T+dt, 10):
        for t2 in np.arange(t1, T+dt, 10):
            J = Jcost(bang((t1, t2)))[0]
            if J < b2[0]:
                b2 = (J, ('pulse', t1, t2))
    J, tag = min([b1, b2], key=lambda x: x[0])
    kind = tag[0]
    E = bang((0, tag[1])) if kind == 'single' else bang((tag[1], tag[2]))
    return E, J, tag

def singular_arc_report(E):
    """Along control E, report zeros of the switching function and their slopes."""
    S, C, M = forward(E)
    l1, l2, l3 = backward(S, C, M, E)
    sig, Phi = switching(S, C, M, l1, l2, l3)
    cr = np.where(np.diff(np.sign(sig)) != 0)[0]
    print(f"  switching function sigma = W1 - Phi:")
    print(f"    zero crossings at t ~ {[round(tg[i]) for i in cr]} h")
    for i in cr:
        slope = (sig[i+1] - sig[i]) / dt      # sigma-dot is control-free
        print(f"    t={tg[i]:.0f}h: d(sigma)/dt = {slope:+.3e}  "
              f"({'transversal -> bang-bang' if abs(slope) > 1e-6 else 'FLAT -> singular?'})")
    # longest interval where sigma stays ~0
    flat = np.abs(sig) < 0.01*W1
    longest = cur = 0
    for f in flat:
        cur = cur+1 if f else 0
        longest = max(longest, cur)
    verdict = "NO singular arc" if longest <= 1 else "possible singular arc -- investigate"
    print(f"    longest run with |sigma|<1% W1: {longest} step(s) -> {verdict}")

if __name__ == "__main__":
    print(f"Parameters: A={A}, W1={W1}, Emax={Emax}, T={T}, N={N}")
    E_sw = sweep_smoothed()
    E_opt, J_opt, tag = optimal_switch_control()
    print(f"\nSmoothed sweep cost   J = {Jcost(E_sw)[0]:.5f}  (E in [{E_sw.min():.3f},{E_sw.max():.3f}])")
    print(f"Direct switch optimum J = {J_opt:.5f}  structure = {tag}")
    print("\nSingular-arc test along the optimal bang-bang control:")
    singular_arc_report(E_opt)
