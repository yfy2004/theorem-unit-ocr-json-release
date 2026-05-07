"""Chapter-level macro average for Ready (and sub-metrics) across all methods."""
import json, sys, statistics
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
from eval_v2 import match_units, score_parse, score_symbol, score_retrieval

GOLD_DIR = BASE / 'Klenke14_revised' / 'json'
METHODS = {
    'Rule':   BASE / 'predictions' / 'rule',
    'LLM':    BASE / 'predictions' / 'llm',
    'Hybrid': BASE / 'predictions' / 'hybrid',
    'Ours':   BASE / 'predictions' / 'ours',
}

results = {}  # method -> list of per-chapter dicts

for method, pred_dir in METHODS.items():
    ch_results = []
    for ch in range(1, 27):
        gf = GOLD_DIR / f'Chapter{ch}.json'
        pf = pred_dir / f'Chapter{ch}_pred.json'
        if not gf.exists() or not pf.exists():
            continue
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        matched_preds = [p for _, p in matched]

        if not matched_preds:
            ch_results.append({'ch': ch, 'n': 0, 'Parse': 0, 'Symbol': 0, 'Retrieval': 0, 'Ready': 0})
            continue

        ps = [score_parse(u) for u in matched_preds]
        ss = [score_symbol(u) for u in matched_preds]
        rs = [score_retrieval(u) for u in matched_preds]
        ready = [(a+b+c)/3 for a,b,c in zip(ps, ss, rs)]

        ch_results.append({
            'ch': ch,
            'n': len(matched_preds),
            'Parse': sum(ps)/len(ps)*100,
            'Symbol': sum(ss)/len(ss)*100,
            'Retrieval': sum(rs)/len(rs)*100,
            'Ready': sum(ready)/len(ready)*100,
        })
    results[method] = ch_results

# Print per-chapter table for each method
for method in METHODS:
    print(f"\n{'='*70}")
    print(f"  {method} — Per-Chapter Breakdown")
    print(f"{'='*70}")
    print(f"  {'Ch':>4s}  {'N':>4s}  {'Parse':>6s}  {'Symbol':>7s}  {'Retrieval':>9s}  {'Ready':>6s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*6}")
    for r in results[method]:
        print(f"  {r['ch']:4d}  {r['n']:4d}  {r['Parse']:6.1f}  {r['Symbol']:7.1f}  {r['Retrieval']:9.1f}  {r['Ready']:6.1f}")

# Print summary table
print(f"\n\n{'='*70}")
print(f"  CHAPTER-LEVEL MACRO AVERAGE (for Appendix table)")
print(f"{'='*70}")
print(f"  {'Method':<8s}  {'Ch-Avg Ready':>12s}  {'Std':>6s}  {'Min':>6s}  {'Max':>6s}  {'Ch-Avg Parse':>12s}  {'Ch-Avg Symbol':>13s}  {'Ch-Avg Retr':>11s}")
print(f"  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*13}  {'-'*11}")

for method in METHODS:
    readys = [r['Ready'] for r in results[method] if r['n'] > 0]
    parses = [r['Parse'] for r in results[method] if r['n'] > 0]
    symbols = [r['Symbol'] for r in results[method] if r['n'] > 0]
    retrs = [r['Retrieval'] for r in results[method] if r['n'] > 0]

    avg_r = statistics.mean(readys)
    std_r = statistics.stdev(readys) if len(readys) > 1 else 0
    min_r = min(readys)
    max_r = max(readys)
    avg_p = statistics.mean(parses)
    avg_s = statistics.mean(symbols)
    avg_ret = statistics.mean(retrs)

    print(f"  {method:<8s}  {avg_r:12.1f}  {std_r:6.1f}  {min_r:6.1f}  {max_r:6.1f}  {avg_p:12.1f}  {avg_s:13.1f}  {avg_ret:11.1f}")
