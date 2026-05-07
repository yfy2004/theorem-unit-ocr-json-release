"""Direct micro calculation from per-chapter results (no subprocess)."""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

# Add scripts to path
sys.path.insert(0, str(Path('scripts')))
from eval_v2 import match_units, calc_boundary, calc_type, calc_completeness, calc_edit, calc_formula_attach, calc_downstream

methods = {
    'rule': list(range(1,6)),
    'llm': [1, 2],
    'hybrid': [1, 2],
}

for method, chapters in methods.items():
    all_gold, all_pred = [], []
    for ch in chapters:
        gold_path = Path(f'Klenke14_revised/json/Chapter{ch}.json')
        pred_path = Path(f'predictions/{method}/Chapter{ch}_pred.json')
        if not gold_path.exists() or not pred_path.exists():
            continue
        with gold_path.open('r', encoding='utf-8') as f:
            all_gold.extend(json.load(f))
        with pred_path.open('r', encoding='utf-8') as f:
            all_pred.extend(json.load(f))
    
    matched, unmatched, extra = match_units(all_gold, all_pred)
    bd = calc_boundary(matched, len(all_gold), len(all_pred))
    tp = calc_type(matched)
    comp = calc_completeness(matched)
    ed = calc_edit(matched)
    fa = calc_formula_attach(matched)
    ds = calc_downstream([p for _, p in matched])
    
    ch_str = ','.join(str(c) for c in chapters)
    print(f'{method.upper()} (Ch{ch_str}): G={len(all_gold)} P={len(all_pred)} M={len(matched)}')
    print(f'  B-F1={bd["B-F1"]} Type={tp} Compl={comp} Edit={ed} F-Att={fa}')
    print(f'  Parse={ds["Parse"]} Symbol={ds["Symbol"]} Retrieval={ds["Retrieval"]} Ready={ds["Ready"]}')
    print()
