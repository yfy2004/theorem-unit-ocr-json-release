"""COMPLETE DATA INTEGRITY CHECK — verify every piece of data we're using."""
import json, sys, os
from pathlib import Path
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path('scripts')))
from eval_v2 import match_units, calc_boundary, calc_type, calc_completeness, calc_edit

GOLD_DIR = Path('Klenke14_revised/json')
METHODS = {
    'rule': Path('predictions/rule'),
    'llm': Path('predictions/llm'),
    'hybrid': Path('predictions/hybrid'),
    'ours': Path('predictions/ours'),
    'ours_no_llm': Path('predictions/ours_no_llm'),
}

print("=" * 80)
print("COMPLETE DATA INTEGRITY CHECK")
print("=" * 80)

# ===== 1. Check all files exist =====
print("\n[1] FILE EXISTENCE CHECK")
for ch in range(1, 27):
    gf = GOLD_DIR / f'Chapter{ch}.json'
    if not gf.exists():
        print(f"  ❌ MISSING GOLD: {gf}")
    for method, pd in METHODS.items():
        pf = pd / f'Chapter{ch}_pred.json'
        if not pf.exists():
            print(f"  ❌ MISSING {method}: {pf}")

print("  File existence check complete.")

# ===== 2. Check file sizes and timestamps =====
print("\n[2] FILE SIZES & TIMESTAMPS")
for method, pd in METHODS.items():
    sizes = []
    times = []
    for ch in range(1, 27):
        pf = pd / f'Chapter{ch}_pred.json'
        if pf.exists():
            st = pf.stat()
            sizes.append(st.st_size)
            times.append(datetime.fromtimestamp(st.st_mtime))
    if times:
        print(f"  {method:>12}: {len(sizes)} files, "
              f"size {min(sizes):,}-{max(sizes):,} bytes, "
              f"modified {min(times).strftime('%m/%d %H:%M')}-{max(times).strftime('%m/%d %H:%M')}")

# ===== 3. Check gold vs pred unit counts =====
print("\n[3] UNIT COUNTS per chapter")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    issues = []
    for ch in range(1, 27):
        gf = GOLD_DIR / f'Chapter{ch}.json'
        pf = METHODS[method] / f'Chapter{ch}_pred.json'
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        diff = len(pred) - len(gold)
        if diff != 0:
            issues.append(f"Ch{ch}:g={len(gold)},p={len(pred)},Δ={diff:+d}")
    if issues:
        print(f"  {method:>8}: {len(issues)} chapters with count mismatch: {', '.join(issues[:5])}{'...' if len(issues)>5 else ''}")
    else:
        print(f"  {method:>8}: ALL chapters match gold unit count ✓")

# ===== 4. Check label consistency =====
print("\n[4] LABEL CONSISTENCY — do matched pairs have same labels?")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    label_mismatches = 0
    total_matched = 0
    for ch in range(1, 27):
        gold = json.load((GOLD_DIR / f'Chapter{ch}.json').open('r', encoding='utf-8'))
        pred = json.load((METHODS[method] / f'Chapter{ch}_pred.json').open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        total_matched += len(matched)
        for g, p in matched:
            if g.get('label') != p.get('label'):
                label_mismatches += 1
    print(f"  {method:>8}: {total_matched} matched, {label_mismatches} label mismatches")

# ===== 5. Check env consistency =====
print("\n[5] ENV CONSISTENCY — matched pairs with different env")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    env_mismatches = []
    for ch in range(1, 27):
        gold = json.load((GOLD_DIR / f'Chapter{ch}.json').open('r', encoding='utf-8'))
        pred = json.load((METHODS[method] / f'Chapter{ch}_pred.json').open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        for g, p in matched:
            if g.get('env') != p.get('env'):
                env_mismatches.append(f"{g['label']}: {g['env']}→{p.get('env')}")
    if env_mismatches:
        print(f"  {method:>8}: {len(env_mismatches)} env mismatches: {env_mismatches[:3]}")
    else:
        print(f"  {method:>8}: ALL env match ✓ (= Type Acc 100%)")

# ===== 6. Verify key metrics match what we reported =====
print("\n[6] METRIC VERIFICATION — recalculate and compare")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    all_matched = []
    total_gold = 0
    total_pred = 0
    for ch in range(1, 27):
        gold = json.load((GOLD_DIR / f'Chapter{ch}.json').open('r', encoding='utf-8'))
        pred = json.load((METHODS[method] / f'Chapter{ch}_pred.json').open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        all_matched.extend(matched)
        total_gold += len(gold)
        total_pred += len(pred)
    b = calc_boundary(all_matched, total_gold, total_pred)
    t = calc_type(all_matched)
    comp = calc_completeness(all_matched)
    edit = calc_edit(all_matched)
    print(f"  {method:>8}: B-F1={b['B-F1']} Type={t} Compl={comp} Edit={edit}")

# ===== 7. Check content sanity =====
print("\n[7] CONTENT SANITY — empty content, extremely short units")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    empty = 0
    short = 0
    total = 0
    for ch in range(1, 27):
        pred = json.load((METHODS[method] / f'Chapter{ch}_pred.json').open('r', encoding='utf-8'))
        for u in pred:
            total += 1
            c = u.get('content', '')
            if not c.strip():
                empty += 1
            elif len(c.strip()) < 10:
                short += 1
    print(f"  {method:>8}: {total} units, {empty} empty content, {short} very short (<10 chars)")

# ===== 8. Check _source_blocks sanity =====
print("\n[8] _SOURCE_BLOCKS CHECK — do predictions have block references?")
for method in ['rule', 'hybrid', 'ours']:
    has_blocks = 0
    total = 0
    for ch in range(1, 27):
        pred = json.load((METHODS[method] / f'Chapter{ch}_pred.json').open('r', encoding='utf-8'))
        for u in pred:
            total += 1
            if u.get('_source_blocks'):
                has_blocks += 1
    print(f"  {method:>8}: {has_blocks}/{total} have _source_blocks")

# ===== 9. Gold data has LaTeX? =====
print("\n[9] GOLD CONTENT FORMAT — check if gold has LaTeX")
latex_cmds = 0
total_gold_units = 0
has_dollar = 0
for ch in range(1, 27):
    gold = json.load((GOLD_DIR / f'Chapter{ch}.json').open('r', encoding='utf-8'))
    for u in gold:
        total_gold_units += 1
        c = u.get('content', '')
        if '\\begin{' in c or '\\end{' in c:
            latex_cmds += 1
        if '$' in c:
            has_dollar += 1
print(f"  Gold units with \\begin/\\end: {latex_cmds}/{total_gold_units}")
print(f"  Gold units with $ (formulas): {has_dollar}/{total_gold_units}")

# ===== 10. F-Attach = 100.0 — why? =====
print("\n[10] F-ATTACH DEEP CHECK — why is it 100.0 for all methods?")
import re
FORMULA_RE = re.compile(r'\$\$.*?\$\$|\$[^$]+\$', re.DOTALL)
gold_with_formulas = 0
gold_without_formulas = 0
for ch in range(1, 27):
    gold = json.load((GOLD_DIR / f'Chapter{ch}.json').open('r', encoding='utf-8'))
    for u in gold:
        c = u.get('content', '')
        formulas = FORMULA_RE.findall(c)
        if formulas:
            gold_with_formulas += 1
        else:
            gold_without_formulas += 1
print(f"  Gold units WITH formula markup: {gold_with_formulas}")
print(f"  Gold units WITHOUT formula markup: {gold_without_formulas}")
print(f"  → If most gold units have no $...$ formulas, F-Attach defaults to 1.0 for those")

print("\n" + "=" * 80)
print("INTEGRITY CHECK COMPLETE")
print("=" * 80)
