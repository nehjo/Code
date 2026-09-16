"""
build_compartments.py -- Build per-rumour S/C/M time-series CSVs from the PHEME
veracity dataset, using REAL veracity tags and REAL SDQC stance where available.

Project: "Optimal Control of an SIR Model of Misinformation through Educational
Campaigns" 
Author: Nehal Joshi.  Mentor / co-author: Dr. Padmanabhan Seshaiyer, George Mason University.

Pipeline position: first stage. Output CSVs feed pool_high_sdqc.py and PINNs.py.

What it does:
  * Reads the ANNOTATED dataset (all-rnr-annotated-threads) -> real veracity.
  * Keeps only FALSE rumours (misinformation) -- "misinformed" is only
    meaningful when the rumour is actually false. (Set INCLUDE_UNVERIFIED=True
    to also keep unverified.)
  * Stance per tweet comes from RumourEval-2019 SDQC labels when available,
    and falls back to a keyword heuristic otherwise. Coverage is reported.
  * Handles the 'source-tweets' folder name (this dataset) and 'source-tweet'.

Compartment mapping (for a FALSE rumour):
    SDQC support -> M (spreading the false claim)
    SDQC deny    -> C (critically literate / pushing back)
    SDQC query   -> S (engaged, no belief formed yet)
    SDQC comment -> S
    source tweet -> M (the false claim itself)
    keyword fallback (no SDQC): deny-keyword -> C, else -> M
      (NOTE: keyword can't detect query/comment, so it slightly over-counts M
       on threads outside the SDQC subset; those are flagged by sdqc_frac.)

Data is NOT bundled with this repo. Download the PHEME "all-rnr-annotated-threads"
corpus and the RumourEval-2019 training data, place them under Data/, and set the
MISINFO_BASE environment variable if your layout differs from the repo default.

Usage:
    python3 build_compartments.py
    MISINFO_BASE=/path/to/project python3 build_compartments.py
"""
import os, json, glob, csv
from datetime import datetime
from collections import defaultdict

# Repo root by default (parent of Code/); override with MISINFO_BASE.
BASE      = os.environ.get(
    "MISINFO_BASE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
ANNOTATED = f"{BASE}/Data/all-rnr-annotated-threads"
RE_DIR    = f"{BASE}/Data/rumoureval-2019-training-data"
OUT_DIR   = f"{BASE}/Data/CSVs_annotated"          # new dir; old CSVs/ untouched

INCLUDE_UNVERIFIED = False
LIMIT_PER_EVENT    = int(os.environ.get("LIMIT", "0"))  # 0 = no limit (full run)
DENY_KEYWORDS = ['fake', 'false', 'hoax', 'lie', 'bullshit', 'not true', 'debunk']
SDQC_TO_COMP  = {'support': 'M', 'deny': 'C', 'query': 'S', 'comment': 'S'}


def load_sdqc():
    m = {}
    for f in ('train-key.json', 'dev-key.json', 'test-key.json',
              'final-eval-key.json'):
        p = os.path.join(RE_DIR, f)
        if os.path.exists(p):
            try:
                m.update(json.load(open(p)).get('subtaskaenglish', {}))
            except Exception:
                pass
    return m


def veracity(a):
    t = str(a.get('true', '')).lower()
    if t == '1': return 'true'
    if str(a.get('misinformation', 0)) in ('1', 'true'): return 'false'
    if t == '0': return 'false'
    return 'unverified'


def parse_dt(s):
    return datetime.strptime(s, '%a %b %d %H:%M:%S %z %Y')


def tweets_in(folder):
    for sub in ('source-tweets', 'source-tweet', 'reactions'):
        d = os.path.join(folder, sub)
        if not os.path.isdir(d):
            continue
        is_source = sub.startswith('source')
        for fn in os.listdir(d):
            if fn.endswith('.json'):
                yield os.path.join(d, fn), is_source


def compartment(tw_id, text, is_source, sdqc):
    if is_source:
        return 'M', (tw_id in sdqc)           # source of a false rumour = M
    if tw_id in sdqc:
        return SDQC_TO_COMP.get(sdqc[tw_id], 'S'), True
    t = (text or '').lower()                  # keyword fallback
    return ('C' if any(k in t for k in DENY_KEYWORDS) else 'M'), False


def process_thread(folder, sdqc):
    rows, n_sdqc, n_tot = [], 0, 0
    for path, is_source in tweets_in(folder):
        try:
            tw = json.load(open(path))
        except Exception:
            continue
        ca = tw.get('created_at', '')
        if not ca:
            continue
        tid = tw.get('id_str', '') or str(tw.get('id', ''))
        comp, used = compartment(tid, tw.get('text', ''), is_source, sdqc)
        rows.append((parse_dt(ca), comp)); n_tot += 1; n_sdqc += used
    if len(rows) < 2:
        return None
    rows.sort()
    t0 = rows[0][0]
    hourly = defaultdict(lambda: {'S': 0, 'C': 0, 'M': 0})
    for dt, comp in rows:
        h = int((dt - t0).total_seconds() // 3600)
        hourly[h][comp] += 1
    hmax = max(hourly)
    Sc = Cc = Mc = 0; out = []
    for h in range(hmax + 1):
        Sc += hourly[h]['S']; Cc += hourly[h]['C']; Mc += hourly[h]['M']
        out.append((h, Sc, Cc, Mc))
    return out, n_tot, n_sdqc


def main():
    sdqc = load_sdqc()
    print(f"Loaded {len(sdqc)} SDQC tweet labels.")
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for ev_dir in sorted(os.listdir(ANNOTATED)):
        rdir = os.path.join(ANNOTATED, ev_dir, 'rumours')
        if not os.path.isdir(rdir):
            continue
        event = ev_dir.replace('-all-rnr-threads', '')
        kept = 0
        for tid in sorted(os.listdir(rdir)):
            folder = os.path.join(rdir, tid)
            ap = os.path.join(folder, 'annotation.json')
            if not os.path.isdir(folder) or not os.path.exists(ap):
                continue
            try:
                ver = veracity(json.load(open(ap)))
            except Exception:
                continue
            if ver == 'true' or (ver == 'unverified' and not INCLUDE_UNVERIFIED):
                continue
            if LIMIT_PER_EVENT and kept >= LIMIT_PER_EVENT:
                break
            res = process_thread(folder, sdqc)
            if not res:
                continue
            out, n_tot, n_sdqc = res
            fn = f"{event}_rumor_{tid}_data.csv"
            with open(os.path.join(OUT_DIR, fn), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time_hour', 'S_cumulative', 'C_cumulative', 'M_cumulative'])
                w.writerows(out)
            peakM = out[-1][3]; peakC = out[-1][2]
            summary.append(dict(event=event, thread=tid, veracity=ver,
                                rows=len(out), peakM=peakM, peakC=peakC,
                                tweets=n_tot, sdqc=n_sdqc,
                                sdqc_frac=round(n_sdqc / n_tot, 3) if n_tot else 0))
            kept += 1
        print(f"  {event:18s}: {kept} false rumours -> CSVs")
    # coverage summary
    with open(os.path.join(OUT_DIR, '_coverage_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['event', 'thread', 'veracity', 'rows',
                                          'peakM', 'peakC', 'tweets', 'sdqc', 'sdqc_frac'])
        w.writeheader(); w.writerows(summary)
    tot_tw = sum(s['tweets'] for s in summary)
    tot_sd = sum(s['sdqc'] for s in summary)
    print(f"\nDONE. {len(summary)} false-rumour CSVs in {OUT_DIR}")
    print(f"Total tweets: {tot_tw}   with real SDQC: {tot_sd} "
          f"({100*tot_sd/tot_tw:.1f}%)" if tot_tw else "no tweets")
    rich = [s for s in summary if s['rows'] >= 15 and s['peakC'] >= 3]
    print(f"Threads suitable for fitting (rows>=15 and peakC>=3): {len(rich)}")


if __name__ == '__main__':
    main()
