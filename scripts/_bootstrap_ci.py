"""Bootstrap 95% CI for Parse, Symbol, Retrieval, Ready across all methods."""
import json, sys, random
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from eval_v2 import match_units, score_parse, score_symbol, score_retrieval

random.seed(42)
N_BOOT = 10000
GOLD_DIR = Path('Klenke14_revised/json')
METHODS = {'Rule': 'predictions/rule', 'LLM': 'predictions/llm',
           'Hybrid': 'predictions/hybrid', 'Ours': 'predictions/ours'}

def bootstrap_ci(scores, n=N_BOOT, alpha=0.05):
    means = []
    k = len(scores)
    for _ in range(n):
        sample = [scores[random.randint(0, k-1)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(n * alpha / 2)]
    hi = means[int(n * (1 - alpha / 2))]
    return lo, hi

print(f"Bootstrap 95% CI (n_boot={N_BOOT}, seed=42)")
print(f"{'='*80}")

for method, pred_dir in METHODS.items():
    # Collect all matched preds
    all_matched_preds = []
    for ch in range(1, 27):
        gf = GOLD_DIR / f'Chapter{ch}.json'
        pf = Path(pred_dir) / f'Chapter{ch}_pred.json'
        if not gf.exists() or not pf.exists():
            continue
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        all_matched_preds.extend([p for _, p in matched])

    n = len(all_matched_preds)

    # Compute per-unit scores
    parse_scores = [score_parse(u) for u in all_matched_preds]
    symbol_scores = [score_symbol(u) for u in all_matched_preds]
    retrieval_scores = [score_retrieval(u) for u in all_matched_preds]
    ready_scores = [(p + s + r) / 3 for p, s, r in
                    zip(parse_scores, symbol_scores, retrieval_scores)]

    # Point estimates
    parse_mean = sum(parse_scores) / n * 100
    symbol_mean = sum(symbol_scores) / n * 100
    retrieval_mean = sum(retrieval_scores) / n * 100
    ready_mean = sum(ready_scores) / n * 100

    # Bootstrap CIs
    p_lo, p_hi = bootstrap_ci(parse_scores)
    s_lo, s_hi = bootstrap_ci(symbol_scores)
    r_lo, r_hi = bootstrap_ci(retrieval_scores)
    rd_lo, rd_hi = bootstrap_ci(ready_scores)

    print(f"\n{method} (n={n} matched units)")
    print(f"  Parse:     {parse_mean:5.1f}  [{p_lo*100:5.1f}, {p_hi*100:5.1f}]")
    print(f"  Symbol:    {symbol_mean:5.1f}  [{s_lo*100:5.1f}, {s_hi*100:5.1f}]")
    print(f"  Retrieval: {retrieval_mean:5.1f}  [{r_lo*100:5.1f}, {r_hi*100:5.1f}]")
    print(f"  Ready:     {ready_mean:5.1f}  [{rd_lo*100:5.1f}, {rd_hi*100:5.1f}]")
