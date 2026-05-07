"""
ALL TABLES — one script computes everything.
Run AFTER: _run_llm_all.py, _run_hybrid_all.py, _run_ours_all.py, _run_ablation.py
"""
import json, sys, re
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path('scripts')))
from eval_v2 import (match_units, calc_boundary, calc_type,
                      calc_completeness, calc_edit, calc_formula_attach,
                      calc_downstream, score_parse, score_symbol)

METHODS = {
    'rule': 'predictions/rule',
    'llm': 'predictions/llm',
    'hybrid': 'predictions/hybrid',
    'ours': 'predictions/ours',
    'ours_no_llm': 'predictions/ours_no_llm',
}
GOLD_DIR = 'Klenke14_revised/json'
OCR_DIR = 'real_data/Klenke14/OCR'
CHAPTERS = list(range(1, 27))
FORMULA_RE = re.compile(r'\$\$.*?\$\$|\$[^$]+\$', re.DOTALL)

# ================================================================
# Collect all data
# ================================================================
results = {}
for method, pred_dir in METHODS.items():
    all_matched = []
    all_matched_preds = []
    total_gold = 0
    total_pred = 0
    all_preds_raw = []

    for ch in CHAPTERS:
        gf = Path(GOLD_DIR) / f'Chapter{ch}.json'
        pf = Path(pred_dir) / f'Chapter{ch}_pred.json'
        if not gf.exists() or not pf.exists():
            continue
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        all_matched.extend(matched)
        total_gold += len(gold)
        total_pred += len(pred)
        all_matched_preds.extend([p for _, p in matched])
        all_preds_raw.extend(pred)

    results[method] = {
        'matched': all_matched,
        'matched_preds': all_matched_preds,
        'total_gold': total_gold,
        'total_pred': total_pred,
        'all_preds': all_preds_raw,
    }


# ================================================================
# TABLE 1: Dataset Statistics (computed from GOLD + OCR + Rule preds)
# ================================================================
print("=" * 70)
print("TABLE 1: Dataset Statistics")
print("=" * 70)

total_pages = 0
total_blocks = 0
total_gold_units = 0
env_cnt = Counter()
gold_with_formula = 0
gold_formula_count = 0

for ch in CHAPTERS:
    gf = Path(GOLD_DIR) / f'Chapter{ch}.json'
    gold = json.load(gf.open('r', encoding='utf-8'))
    total_gold_units += len(gold)
    
    ch_ocr = Path(OCR_DIR) / f'Chapter{ch}'
    pages = len(list(ch_ocr.glob('p*.json')))
    total_pages += pages
    
    blocks = 0
    for pf in sorted(ch_ocr.glob('p*.json')):
        pd = json.load(pf.open('r', encoding='utf-8'))
        blocks += len(pd.get('blocks', []))
    total_blocks += blocks
    
    for u in gold:
        env_cnt[u.get('env', '?')] += 1
        content = u.get('content', '')
        formulas = FORMULA_RE.findall(content)
        gold_formula_count += len(formulas)
        if formulas:
            gold_with_formula += 1

# Blocks per unit from Rule predictions' _source_blocks
blocks_per_unit = []
for ch in CHAPTERS:
    pf = Path('predictions/rule') / f'Chapter{ch}_pred.json'
    pred = json.load(pf.open('r', encoding='utf-8'))
    for u in pred:
        src = u.get('_source_blocks', [])
        proof_b = u.get('_proof_blocks', [])
        blocks_per_unit.append(len(list(src) + list(proof_b)))

avg_blocks = sum(blocks_per_unit) / len(blocks_per_unit)
multi_block = sum(1 for b in blocks_per_unit if b > 1)
multi_block_ratio = multi_block / len(blocks_per_unit) * 100
avg_formulas = gold_formula_count / total_gold_units

print(f"  Pages:              {total_pages}")
print(f"  OCR Blocks:         {total_blocks}")
print(f"  Theorem Units:      {total_gold_units}")
print(f"  Avg Blocks/Unit:    {avg_blocks:.1f}  (from _source_blocks)")
print(f"  Multi-block Ratio:  {multi_block_ratio:.1f}%")
print(f"  Avg Formulas/Unit:  {avg_formulas:.1f}  (from gold $...$ count)")
print(f"  Units w/ formulas:  {gold_with_formula}/{total_gold_units}")
print(f"  Type distribution:")
for env in ['thm', 'exr', 'def', 'ex', 'rem', 'lem', 'cor']:
    c = env_cnt.get(env, 0)
    print(f"    {env:>4}: {c:>4} ({c/total_gold_units*100:.1f}%)")


# ================================================================
# TABLE 2: Main Results
# ================================================================
print("\n" + "=" * 70)
print("TABLE 2: Main Results")
print("=" * 70)
print(f"{'Method':<10} {'B-P':>5} {'B-R':>5} {'B-F1':>5} {'Type':>5} {'F-Att':>5} "
      f"{'Compl':>5} {'Edit↓':>6} {'Ready':>5}")
print("-" * 55)

for method in ['rule', 'llm', 'hybrid', 'ours']:
    r = results[method]
    b = calc_boundary(r['matched'], r['total_gold'], r['total_pred'])
    t = calc_type(r['matched'])
    comp = calc_completeness(r['matched'])
    edit = calc_edit(r['matched'])
    fa = calc_formula_attach(r['matched'])
    ds = calc_downstream(r['matched_preds'])
    print(f"{method:<10} {b['B-P']:>5} {b['B-R']:>5} {b['B-F1']:>5} {t:>5} {fa:>5} "
          f"{comp:>5} {edit:>6} {ds['Ready']:>5}")


# ================================================================
# TABLE 3: Per-Type Breakdown
# ================================================================
print("\n" + "=" * 70)
print("TABLE 3: Per-Type Breakdown")
print("=" * 70)

ENV_NAMES = {'thm': 'Theorem', 'exr': 'Exercise', 'def': 'Definition',
             'ex': 'Example', 'rem': 'Remark', 'lem': 'Lemma', 'cor': 'Corollary'}

for method in ['rule', 'llm', 'hybrid', 'ours']:
    print(f"\n--- {method.upper()} ---")
    print(f"{'Type':<12} {'#':>4} {'Parse':>6} {'Symbol':>7} {'Retrieval':>9} {'Ready':>6}")
    r = results[method]
    by_type = defaultdict(list)
    for g, p in r['matched']:
        by_type[g.get('env', '?')].append((g, p))
    for env in ['thm', 'exr', 'def', 'ex', 'rem', 'lem', 'cor']:
        pairs = by_type.get(env, [])
        if not pairs:
            continue
        preds = [p for _, p in pairs]
        ds = calc_downstream(preds)
        print(f"{ENV_NAMES.get(env, env):<12} {len(pairs):>4} {ds['Parse']:>6} "
              f"{ds['Symbol']:>7} {ds['Retrieval']:>9} {ds['Ready']:>6}")


# ================================================================
# TABLE 4: Downstream Usability
# ================================================================
print("\n" + "=" * 70)
print("TABLE 4: Downstream Usability")
print("=" * 70)
print(f"{'Method':<10} {'Parse':>6} {'Symbol':>7} {'Retrieval':>9} {'Ready':>6}")
print("-" * 42)

for method in ['rule', 'llm', 'hybrid', 'ours']:
    r = results[method]
    ds = calc_downstream(r['matched_preds'])
    print(f"{method:<10} {ds['Parse']:>6} {ds['Symbol']:>7} {ds['Retrieval']:>9} {ds['Ready']:>6}")


# ================================================================
# TABLE 9: Ablation Study
# ================================================================
print("\n" + "=" * 70)
print("TABLE 9: Ablation Study")
print("=" * 70)
print(f"{'Configuration':<35} {'B-F1':>5} {'Parse':>6} {'Symbol':>7} {'Retrieval':>10} {'Ready':>6}")
print("-" * 75)

for method, label in [
    ('rule', 'Rule (no norm, no LLM)'),
    ('hybrid', 'Rule + LLM repair (Hybrid)'),
    ('ours_no_llm', 'Norm + Rule (no LLM)'),
    ('ours', 'Norm + Rule + LLM (Ours)'),
]:
    r = results[method]
    b = calc_boundary(r['matched'], r['total_gold'], r['total_pred'])
    ds = calc_downstream(r['matched_preds'])
    print(f"{label:<35} {b['B-F1']:>5} {ds['Parse']:>6} {ds['Symbol']:>7} "
          f"{ds['Retrieval']:>10} {ds['Ready']:>6}")


# ================================================================
# TABLE 10: Error Analysis
# ================================================================
print("\n" + "=" * 70)
print("TABLE 10: Error Analysis")
print("=" * 70)
print(f"{'Metric':<30} {'Rule':>6} {'LLM':>6} {'Hybrid':>6} {'Ours':>6}")
print("-" * 60)

rows = {'fn': [], 'fp': [], 'fffd': [], 'parse_fail': [], 'sym_fail': []}
for method in ['rule', 'llm', 'hybrid', 'ours']:
    r = results[method]
    rows['fn'].append(r['total_gold'] - len(r['matched']))
    rows['fp'].append(r['total_pred'] - len(r['matched']))
    
    total_fffd = 0
    parse_fail = 0
    sym_fail = 0
    for u in r['all_preds']:
        content = u.get('content', '') + u.get('proof', '')
        total_fffd += content.count('\ufffd')
        if score_parse(u) < 1.0:
            parse_fail += 1
        if score_symbol(u) < 1.0:
            sym_fail += 1
    rows['fffd'].append(total_fffd)
    rows['parse_fail'].append(parse_fail)
    rows['sym_fail'].append(sym_fail)

labels = {
    'fn': 'Boundary FN (missed)',
    'fp': 'Boundary FP (extra)',
    'fffd': 'Remaining U+FFFD chars',
    'parse_fail': 'Units w/ parse errors',
    'sym_fail': 'Units w/ symbol errors',
}
for key in ['fn', 'fp', 'fffd', 'parse_fail', 'sym_fail']:
    v = rows[key]
    print(f"{labels[key]:<30} {v[0]:>6} {v[1]:>6} {v[2]:>6} {v[3]:>6}")


# ================================================================
# TABLE 11: Computational Cost (from _stats.json if available)
# ================================================================
print("\n" + "=" * 70)
print("TABLE 11: Computational Cost")
print("=" * 70)
print(f"{'Method':<10} {'LLM Calls':>10} {'Prompt Tok':>12} {'Compl Tok':>12} "
      f"{'Total Tok':>12} {'Wall Time':>10}")
print("-" * 70)

for method in ['rule', 'llm', 'hybrid', 'ours']:
    if method == 'rule':
        print(f"{'rule':<10} {'0':>10} {'0':>12} {'0':>12} {'0':>12} {'<10s':>10}")
        continue

    pred_dir = Path(METHODS[method])
    stats_files = sorted(pred_dir.glob("*_stats.json"))
    
    if not stats_files:
        print(f"{method:<10} {'NO STATS':>10} — re-run {method} to collect stats")
        continue

    agg = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "total_tokens": 0, "wall_time_seconds": 0}
    for sf in stats_files:
        s = json.load(sf.open('r', encoding='utf-8'))
        agg["llm_calls"] += s.get("llm_calls", 0)
        agg["prompt_tokens"] += s.get("prompt_tokens", 0)
        agg["completion_tokens"] += s.get("completion_tokens", 0)
        agg["total_tokens"] += s.get("total_tokens", 0)
        agg["wall_time_seconds"] += s.get("wall_time_seconds", 0)

    wt = agg["wall_time_seconds"]
    wt_str = f"{int(wt//60)}m{int(wt%60)}s" if wt > 0 else "N/A"
    
    print(f"{method:<10} {agg['llm_calls']:>10,} {agg['prompt_tokens']:>12,} "
          f"{agg['completion_tokens']:>12,} {agg['total_tokens']:>12,} {wt_str:>10}")


print("\n" + "=" * 70)
print("ALL TABLES COMPUTED")
print("=" * 70)
