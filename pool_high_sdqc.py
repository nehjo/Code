"""
pool_high_sdqc.py -- Aggregate the accurately-labelled (high-SDQC) false-rumour
threads into ONE population-average curve, to be fit with the (stable) PINN.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns"
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

Pipeline position: run AFTER build_compartments.py, BEFORE PINNs.py.

Why: the high-SDQC threads are individually too small to identify (beta, gamma, d),
and a classical joint fit is numerically unstable. Averaging them into a single
representative incidence curve (equal weight per thread, preserving each thread's
S:C:M ratios) yields a longer, smoother curve with REAL stance labels that the
PINN can fit stably -> a population-level parameter estimate.

Selection: rows>=15, peakC>=3, sdqc_frac>=0.5 (from _coverage_summary.csv).
Output: Data/CSVs_pooled/pooled_highSDQC_rumor_POOLED_data.csv  (fit with PINNs.py)

Usage:
    python3 pool_high_sdqc.py
    MISINFO_BASE=/path/to/project python3 pool_high_sdqc.py
"""
import os, csv
import numpy as np, pandas as pd
from collections import defaultdict

# Repo root by default (parent of Code/); override with MISINFO_BASE.
BASE = os.environ.get(
    "MISINFO_BASE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
CSV_DIR = f"{BASE}/Data/CSVs_annotated"
OUT_DIR = f"{BASE}/Data/CSVs_pooled"
COV     = f"{CSV_DIR}/_coverage_summary.csv"
MIN_ROWS, MIN_PEAKC, MIN_SDQC = 15, 3, 0.5


def main():
    cov = {r['event'] + '_rumor_' + r['thread'] + '_data.csv': r
           for r in csv.DictReader(open(COV))}
    sel = [fn for fn, r in cov.items()
           if int(r['rows']) >= MIN_ROWS and int(r['peakC']) >= MIN_PEAKC
           and float(r['sdqc_frac']) >= MIN_SDQC
           and os.path.exists(os.path.join(CSV_DIR, fn))]
    acc = defaultdict(lambda: {'S': [], 'C': [], 'M': []})
    for fn in sel:
        df = pd.read_csv(os.path.join(CSV_DIR, fn))
        Si = df['S_cumulative'].diff().fillna(df['S_cumulative'].iloc[0]).clip(lower=0).values
        Ci = df['C_cumulative'].diff().fillna(df['C_cumulative'].iloc[0]).clip(lower=0).values
        Mi = df['M_cumulative'].diff().fillna(df['M_cumulative'].iloc[0]).clip(lower=0).values
        tot = (Si + Ci + Mi).sum() or 1.0        # equal weight per thread; keep S:C:M ratio
        for h in range(len(df)):
            acc[h]['S'].append(Si[h] / tot)
            acc[h]['C'].append(Ci[h] / tot)
            acc[h]['M'].append(Mi[h] / tot)
    hmax = max(acc); Sc = Cc = Mc = 0.0; rows = []
    for h in range(hmax + 1):
        Sc += np.mean(acc[h]['S']) if acc[h]['S'] else 0
        Cc += np.mean(acc[h]['C']) if acc[h]['C'] else 0
        Mc += np.mean(acc[h]['M']) if acc[h]['M'] else 0
        rows.append((h, round(Sc, 6), round(Cc, 6), round(Mc, 6)))
    os.makedirs(OUT_DIR, exist_ok=True)
    outp = os.path.join(OUT_DIR, "pooled_highSDQC_rumor_POOLED_data.csv")
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_hour', 'S_cumulative', 'C_cumulative', 'M_cumulative'])
        w.writerows(rows)
    print(f"Pooled {len(sel)} high-SDQC threads -> {len(rows)} hourly points")
    print(f"Wrote {outp}")


if __name__ == '__main__':
    main()
