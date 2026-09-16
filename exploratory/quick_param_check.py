"""
quick_param_check.py
--------------------
FAST sanity check for the sign/plausibility of (beta, gamma, d) WITHOUT the PINN.

It does a classical multi-start ODE least-squares fit (scipy) of the SAME SCM
model to the SAME incidence data, in ~a second per dataset. Use it as a quick
smoke test before a full PINN run.

IMPORTANT: this is a PROXY. It has no physics regularization, so on tiny/noisy
threads it is deliberately unforgiving and can blow up -- that instability is
itself the signal that a thread is not identifiable. For trustworthy numbers,
run a (short) PINN on the informative threads. Parameters are fit UNCONSTRAINED
so their natural sign shows.

Model (E=0, Lambda=0, mu=0, psi=theta=0.04, corrected gamma sign):
    dS = -beta*S*M + psi*C + theta*M
    dC =  gamma*C*M - psi*C
    dM =  beta*S*M - gamma*C*M - theta*M - d*M
"""
import glob, os
import numpy as np, pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

CSV_DIR = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/CSVs"
PSI = THETA = 0.04
STARTS = ([0.1,0.1,0.05],[0.3,-0.1,0.1],[0.05,0.05,0.2],[0.5,-0.5,0.5],[0.1,-0.1,0.02])

def scm(y, t, b, g, d):
    S, C, M = y
    return [-b*S*M+PSI*C+THETA*M, g*C*M-PSI*C, b*S*M-g*C*M-THETA*M-d*M]

def fit(path):
    df = pd.read_csv(path)
    if len(df) < 5:
        return None, None
    t  = df['time_hour'].values.astype(float)
    Mi = df['M_cumulative'].diff().fillna(df['M_cumulative'].iloc[0]).clip(lower=0).values
    Ci = df['C_cumulative'].diff().fillna(df['C_cumulative'].iloc[0]).clip(lower=0).values
    N  = max(Mi.max(), Ci.max())*3 or 1.0
    Md, Cd = Mi/N, Ci/N
    y0 = [max(1-Md[0]-Cd[0], 0.0), Cd[0], Md[0]]
    def resid(p):
        try: sol = odeint(scm, y0, t, args=tuple(p), mxstep=10000)
        except Exception: return np.full(2*len(t), 1e3)
        return np.concatenate([sol[:,2]-Md, sol[:,1]-Cd])
    best = None
    for x0 in STARTS:
        try:
            r = least_squares(resid, x0, method='trf', loss='soft_l1',
                              f_scale=0.05, x_scale=[0.3,0.3,0.1], max_nfev=6000)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return (best.x if best else None), dict(rows=len(df), peakC=int(Ci.max()))

def main():
    print(f"{'thread':44s} {'beta':>8} {'gamma':>8} {'d':>8}  notes")
    pos = neg = 0
    for f in sorted(glob.glob(os.path.join(CSV_DIR, "*_rumor_*_data.csv"))):
        p, info = fit(f)
        n = os.path.basename(f).replace('_data.csv','')
        if p is None:
            print(f"{n:44s}  skipped (rows<5)"); continue
        note = f"rows={info['rows']}, peakC={info['peakC']}"
        if info['peakC'] <= 2: note += "  <-- C~0: gamma & d weakly identifiable"
        print(f"{n:44s} {p[0]:>8.3f} {p[1]:>8.3f} {p[2]:>8.3f}  {note}")
        pos += p[2] >= 0; neg += p[2] < 0
    print(f"\nUNCONSTRAINED d:  d>=0 in {pos} threads,  d<0 in {neg} threads")
    print("Wild |beta|,|d| values = that thread is not identifiable (too few points / C~0).")

if __name__ == '__main__':
    main()
