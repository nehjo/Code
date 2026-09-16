"""
peak_misinformed.py
-------------------
Find the rumour with the PEAK number of MISINFORMED users, using veracity tags.

A user is counted as "misinformed" in a veracity-aware way:
  * SUPPORTS a rumour that is FALSE, or
  * DENIES  a rumour that is TRUE.
(Supporting a true rumour, or denying a false one, is NOT misinformation.)

This needs two annotations per thread:
  1. Rumour-level veracity -- annotation.json with a 'true' / 'misinformation'
     field (the PHEME "all-rnr-annotated-threads" / veracity release).
  2. Tweet-level stance (SDQC: support / deny / query / comment) -- from an
     annotation 'responses' map, when present.

-------------------------------------------------------------------------------
INVESTIGATION FINDING (2026-08-23):
The dataset currently at PHEME_ROOT_DIR ('Mis-pheme-rnr-dataset') contains NO
veracity tags. Every rumour folder holds only 'source-tweet/' and 'reactions/'
-- there is no annotation.json anywhere and the tweets carry no veracity/stance
fields. So a TRUE veracity-based "most misinformed" cannot be computed from this
dataset. To get the real answer, point PHEME_ROOT_DIR at the ANNOTATED PHEME
release. Until then, this script falls back to a keyword stance proxy (counts
supporters) and prints a clear WARNING that the number is heuristic.
-------------------------------------------------------------------------------
"""
import os, json, csv

PHEME_ROOT_DIR = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/Mis-pheme-rnr-dataset"
OUTPUT_CSV     = "/Users/nehal/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/Misinfo-research/Data/CSVs/peak_misinformed_summary.csv"

DENY_KEYWORDS = ['fake', 'false', 'hoax', 'lie', 'bullshit', 'not true', 'debunk']


def rumour_veracity(folder):
    """Return 'true' / 'false' / 'unverified' / None from annotation.json."""
    ap = os.path.join(folder, 'annotation.json')
    if not os.path.exists(ap):
        return None
    try:
        a = json.load(open(ap))
    except Exception:
        return None
    if 'true' in a:
        v = str(a['true']).lower()
        if v in ('1', 'true'):  return 'true'
        if v in ('0', 'false'): return 'false'
        return 'unverified'
    if 'misinformation' in a:
        return 'false' if str(a['misinformation']).lower() in ('1', 'true') else 'true'
    return None


def stance_of(text, tweet_id, responses):
    """support / deny for a tweet: SDQC response if available, else keyword."""
    if tweet_id in responses:
        lab = str(responses[tweet_id]).lower()
        if lab in ('support', 'deny', 'query', 'comment'):
            return lab
    t = (text or '').lower()
    return 'deny' if any(k in t for k in DENY_KEYWORDS) else 'support'


def misinformed_count(folder):
    """(# misinformed users, veracity) for one rumour folder."""
    veracity = rumour_veracity(folder)
    responses = {}
    ap = os.path.join(folder, 'annotation.json')
    if os.path.exists(ap):
        try:
            a = json.load(open(ap))
            if isinstance(a.get('responses'), dict):
                responses = a['responses']
        except Exception:
            pass

    count = 0
    # accept both 'source-tweet' (rnr dataset) and 'source-tweets' spellings
    for sub in ('source-tweet', 'source-tweets', 'reactions'):
        d = os.path.join(folder, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith('.json'):
                continue
            try:
                tw = json.load(open(os.path.join(d, fn)))
            except Exception:
                continue
            tid = tw.get('id_str', '') or str(tw.get('id', ''))
            st = stance_of(tw.get('text', ''), tid, responses)
            if   veracity == 'false' and st == 'support': count += 1
            elif veracity == 'true'  and st == 'deny':    count += 1
            elif veracity is None     and st == 'support': count += 1  # heuristic fallback
    return count, veracity


def main():
    if not os.path.isdir(PHEME_ROOT_DIR):
        print("Cannot find PHEME_ROOT_DIR:", PHEME_ROOT_DIR)
        return

    rows, any_veracity = [], False
    for ev in sorted(os.listdir(PHEME_ROOT_DIR)):
        rdir = os.path.join(PHEME_ROOT_DIR, ev, 'rumours')
        if not os.path.isdir(rdir):
            continue
        for rid in os.listdir(rdir):
            folder = os.path.join(rdir, rid)
            if not os.path.isdir(folder):
                continue
            c, ver = misinformed_count(folder)
            any_veracity = any_veracity or (ver is not None)
            rows.append({'event': ev, 'rumour_id': rid,
                         'misinformed_users': c, 'veracity': ver})

    rows.sort(key=lambda r: r['misinformed_users'], reverse=True)

    if not any_veracity:
        print("=" * 72)
        print("WARNING: no veracity tags (annotation.json) found in this dataset.")
        print("Numbers below are a KEYWORD-BASED PROXY (supporters per rumour),")
        print("NOT a veracity-based misinformed count. Point PHEME_ROOT_DIR at the")
        print("annotated PHEME release for the real answer.")
        print("=" * 72)

    print("\nTop 10 rumours by misinformed users:")
    for r in rows[:10]:
        print(f"  {r['event']:18s} {r['rumour_id']:22s} "
              f"misinformed={r['misinformed_users']:5d} veracity={r['veracity']}")
    if rows:
        t = rows[0]
        print(f"\nPEAK: {t['event']} / {t['rumour_id']} -> "
              f"{t['misinformed_users']} misinformed users (veracity={t['veracity']}).")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['event', 'rumour_id',
                                          'misinformed_users', 'veracity'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nFull ranking written to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
