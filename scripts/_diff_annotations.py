"""Compare gold vs double annotation to find actual differences."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
GOLD_DIR = BASE / 'Klenke14_revised' / 'json'
DOUBLE_DIR = BASE / 'Klenke14_revised' / 'double__annotation'
print(f"GOLD_DIR: {GOLD_DIR} exists={GOLD_DIR.exists()}")
print(f"DOUBLE_DIR: {DOUBLE_DIR} exists={DOUBLE_DIR.exists()}")

total_diffs = 0
total_units = 0

for ch in range(1, 27):
    gf = GOLD_DIR / f'Chapter{ch}.json'
    df = DOUBLE_DIR / f'Chapter{ch}_double_annotation_v2.json'
    if not gf.exists() or not df.exists():
        continue

    a = json.load(gf.open('r', encoding='utf-8'))
    b = json.load(df.open('r', encoding='utf-8'))
    total_units += len(a)

    ch_diffs = 0
    # Check count difference
    if len(a) != len(b):
        print(f"Ch{ch:2d}: COUNT DIFF  A={len(a)} B={len(b)}")
        ch_diffs += abs(len(a) - len(b))

    # Check unit-by-unit
    for i in range(min(len(a), len(b))):
        diffs = []
        if a[i].get('label') != b[i].get('label'):
            diffs.append(f"label: '{a[i].get('label')}' vs '{b[i].get('label')}'")
        if a[i].get('env') != b[i].get('env'):
            diffs.append(f"env: '{a[i].get('env')}' vs '{b[i].get('env')}'")
        if a[i].get('content') != b[i].get('content'):
            # Compute rough diff size
            ac = a[i].get('content', '')
            bc = b[i].get('content', '')
            diff_chars = abs(len(ac) - len(bc))
            # Find first difference position
            for j in range(min(len(ac), len(bc))):
                if ac[j] != bc[j]:
                    snippet_a = ac[max(0,j-20):j+30]
                    snippet_b = bc[max(0,j-20):j+30]
                    diffs.append(f"content differs at pos {j} (len {len(ac)} vs {len(bc)})")
                    break
            else:
                diffs.append(f"content length differs ({len(ac)} vs {len(bc)})")
        if a[i].get('proof') != b[i].get('proof'):
            ap = a[i].get('proof', '')
            bp = b[i].get('proof', '')
            diffs.append(f"proof differs (len {len(ap)} vs {len(bp)})")
        if a[i].get('dependencies') != b[i].get('dependencies'):
            diffs.append(f"dependencies differ")

        if diffs:
            ch_diffs += 1
            if ch_diffs <= 3:  # Show first 3 per chapter
                print(f"  Ch{ch:2d} unit {i} ({a[i].get('label','?')}): {'; '.join(diffs)}")

    if ch_diffs > 3:
        print(f"  Ch{ch:2d}: ... and {ch_diffs-3} more diffs")
    total_diffs += ch_diffs

    if ch_diffs == 0:
        print(f"Ch{ch:2d}: IDENTICAL (A={len(a)} B={len(b)})")

print(f"\n{'='*60}")
print(f"Total units: {total_units}")
print(f"Total units with differences: {total_diffs}")
print(f"Diff rate: {total_diffs/total_units*100:.1f}%")
