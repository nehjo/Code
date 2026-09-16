"""
pooled_fit.py
-------------
Pooled (shared-parameter) fit of the SCM model.

Motivation: the accurately-labelled (high-SDQC) threads are individually small,
so any one of them can't identify (beta, gamma, d). Pooling fits ONE shared
(beta, gamma, d) across all selected threads at once -- each thread keeps its own
initial condition, but they jointly constrain the shared kinetics. This buys back
identifiability from accurate-but-small data.

It also fits the single big keyword-labelled thread alone, as a robustness check.

Incidence target, mu=0, psi=theta=0.04, corrected gamma sign, unconstrained.
This is a fast classical (scipy) estimate; for a PINN version see notes at end.
"""
import os, csv
import numpy as np, pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

BASE    = os.environ.get("MISINFO_BASE",
    "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research")
CSV_DIR = f"{BASE}/Data/CSVs_annotated"
COV     = f"{CSV_DIR}/_coverage_summary.csv"
PSI = THETA = 0.04
MIN_ROWS, MIN_PEAKC, MIN_SDQC = 15, 3, 0.5
BIG_THREAD = "ferguson_rumor_500280809652514816_data.csv"
STARTS = ([0.1,0.1,0.05],[0.3,0.1,0.1],[0.1,-0.1,0.02],[0.5,0.2,0.1],[0.2,0.05,0.08])

def scm(y,t,b,g,d):
    S,C,M=y
    return [-b*S*M+PSI*C+THETA*M, g*C*M-PSI*C, b*S*M-g*C*M-THETA*M-d*M]

def load_thread(path):
    df=pd.read_csv(path)
    t=df['time_hour'].values.astype(float)
    Mi=df['M_cumulative'].diff().fillna(df['M_cumulative'].iloc[0]).clip(lower=0).values
    Ci=df['C_cumulative'].diff().fillna(df['C_cumulative'].iloc[0]).clip(lower=0).values
    N=max(Mi.max(),Ci.max())*3 or 1.0
    Md,Cd=Mi/N,Ci/N
    y0=[max(1-Md[0]-Cd[0],0.0),Cd[0],Md[0]]
    return t,Md,Cd,y0

def resid(threads,p):
    b,g,d=p; out=[]
    for (t,Md,Cd,y0) in threads:
        try: sol=odeint(scm,y0,t,args=(b,g,d),mxstep=10000)
        except Exception: return np.full(sum(2*len(x[0]) for x in threads),1e3)
        out.append(sol[:,2]-Md); out.append(sol[:,1]-Cd)
    return np.concatenate(out)

def fit(threads):
    best=None
    for x0 in STARTS:
        try:
            r=least_squares(lambda p:resid(threads,p),x0,method='trf',loss='soft_l1',
                            f_scale=0.05,x_scale=[0.3,0.3,0.1],max_nfev=8000)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best.x if best else (float('nan'),)*3

def main():
    cov={r['event']+'_rumor_'+r['thread']+'_data.csv':r for r in csv.DictReader(open(COV))}
    sel=[fn for fn,r in cov.items()
         if int(r['rows'])>=MIN_ROWS and int(r['peakC'])>=MIN_PEAKC and float(r['sdqc_frac'])>=MIN_SDQC]
    print(f"Pooling {len(sel)} high-SDQC threads (rows>={MIN_ROWS}, peakC>={MIN_PEAKC}, sdqc_frac>={MIN_SDQC}):")
    for fn in sel: print("   ", fn)
    threads=[load_thread(os.path.join(CSV_DIR,fn)) for fn in sel]
    b,g,d=fit(threads)
    print(f"\n== POOLED estimate (shared across {len(sel)} accurately-labelled threads) ==")
    print(f"   beta={b:.4f}   gamma={g:.4f}   d={d:.4f}")
    bp=os.path.join(CSV_DIR,BIG_THREAD)
    if os.path.exists(bp):
        bb,gg,dd=fit([load_thread(bp)])
        print(f"\n== ROBUSTNESS: big keyword-labelled thread {BIG_THREAD} (sdqc~0) ==")
        print(f"   beta={bb:.4f}   gamma={gg:.4f}   d={dd:.4f}")

if __name__=='__main__': main()
