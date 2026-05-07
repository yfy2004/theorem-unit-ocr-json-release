"""Micro-averaged evaluation: Rule vs LLM vs Hybrid vs Ours."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path('scripts')))
from eval_v2 import (match_units, calc_boundary, calc_type,
                      calc_completeness, calc_edit, calc_formula_attach,
                      calc_downstream)

METHODS = ['rule', 'llm', 'hybrid', 'ours']
PRED_DIRS = {
    'rule':   'predictions/rule',
    'llm':    'predictions/llm',
    'hybrid': 'predictions/hybrid',
    'ours':   'predictions/ours',
}
GOLD_DIR = 'Klenke14_revised/json'
CHAPTERS = list(range(1, 27))

for method in METHODS:
    all_matched = []
    total_gold = 0
    total_pred = 0
    all_matched_preds = []
    for ch in CHAPTERS:
        gf = Path(GOLD_DIR) / f'Chapter{ch}.json'
        pf = Path(PRED_DIRS[method]) / f'Chapter{ch}_pred.json'
        if not gf.exists() or not pf.exists():
            continue
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        matched, unmatched_g, unmatched_p = match_units(gold, pred)
        all_matched.extend(matched)
        total_gold += len(gold)
        total_pred += len(pred)
        all_matched_preds.extend([p for _, p in matched])

    # Boundary
    b = calc_boundary(all_matched, total_gold, total_pred)
    # Type
    t = calc_type(all_matched)
    # Completeness
    comp = calc_completeness(all_matched)
    # Edit
    edit = calc_edit(all_matched)
    # Formula attach
    fa = calc_formula_attach(all_matched)
    # Downstream
    ds = calc_downstream(all_matched_preds)

    print(f"\n{'='*40}")
    print(f"Method: {method.upper()}")
    print(f"{'='*40}")
    print(f"  Matched: {len(all_matched)} / Gold={total_gold} Pred={total_pred}")
    print(f"  B-P:       {b['B-P']}")
    print(f"  B-R:       {b['B-R']}")
    print(f"  B-F1:      {b['B-F1']}")
    print(f"  Type:      {t}")
    print(f"  Compl:     {comp}")
    print(f"  Edit↓:     {edit}")
    print(f"  F-Att:     {fa}")
    print(f"  Parse:     {ds['Parse']}")
    print(f"  Symbol:    {ds['Symbol']}")
    print(f"  Retrieval: {ds['Retrieval']}")
    print(f"  Ready:     {ds['Ready']}")
