#!/usr/bin/env python3
"""Analyze Rule-based prediction weaknesses against Gold."""
import json, re, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('Klenke14_revised/json/Chapter1.json', encoding='utf-8') as f:
    gold = json.load(f)
with open('predictions/rule/Chapter1_pred.json', encoding='utf-8') as f:
    pred_list = json.load(f)

gold_map = {u['label']: u for u in gold}
pred_map = {u['label']: u for u in pred_list}

def strip_latex(t):
    t = re.sub(r'\\(begin|end)\{[^}]*\}', '', t)
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    t = re.sub(r'[\$\{\}\\^_]', '', t)
    return t.lower().split()

def normalize_ocr(t):
    return t.lower().split()

# 1. Compl: worst token overlaps
print('=== COMPL: Units with worst token overlap (Rule loses content) ===')
scores = []
for label in gold_map:
    if label in pred_map:
        g_tokens = set(strip_latex(gold_map[label].get('content', '')))
        p_tokens = set(normalize_ocr(pred_map[label].get('content', '')))
        if not g_tokens:
            continue
        recall = len(g_tokens & p_tokens) / len(g_tokens)
        scores.append((recall, label))

scores.sort()
for score, label in scores[:8]:
    gc = gold_map[label]['content'][:80].replace('\n', ' ')
    pc = pred_map[label]['content'][:80].replace('\n', ' ')
    print(f'  {label}: compl={score:.2f}')
    print(f'    Gold: {gc}')
    print(f'    Pred: {pc}')
    print()

# 2. Symbol: replacement chars and bracket imbalance
print('=== SYMBOL: Units with OCR corruption ===')
sym_issues = 0
for label in pred_map:
    content = pred_map[label].get('content', '')
    issues = []
    count_repl = content.count('\ufffd')
    if count_repl > 0:
        issues.append(f'{count_repl}x replacement char')
    opens = content.count('(') + content.count('[') + content.count('{')
    closes = content.count(')') + content.count(']') + content.count('}')
    if abs(opens - closes) > 2:
        issues.append(f'bracket imbalance ({opens} vs {closes})')
    if issues:
        sym_issues += 1
        print(f'  {label}: {", ".join(issues)}')

print(f'\n  Total units with symbol issues: {sym_issues}/{len(pred_map)}')

# 3. Type: check env mismatches
print('\n=== TYPE: env mismatches ===')
type_wrong = 0
for label in gold_map:
    if label in pred_map:
        ge = gold_map[label].get('env', '')
        pe = pred_map[label].get('env', '')
        if ge != pe:
            type_wrong += 1
            print(f'  {label}: gold={ge}, pred={pe}')

if type_wrong == 0:
    print('  (none)')
print(f'  Total mismatches: {type_wrong}')

# 4. Edit distance: worst cases
print('\n=== EDIT: Units with highest edit distance ===')
def edit_distance_ratio(s1, s2):
    if not s1 and not s2:
        return 0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0
    # Simple char-level Levenshtein approximation using difflib
    from difflib import SequenceMatcher
    return 1 - SequenceMatcher(None, s1, s2).ratio()

edit_scores = []
for label in gold_map:
    if label in pred_map:
        gc = re.sub(r'\\[a-zA-Z]+', '', gold_map[label].get('content', ''))
        gc = re.sub(r'[\$\{\}\\^_]', '', gc).strip()
        pc = pred_map[label].get('content', '').strip()
        dist = edit_distance_ratio(gc, pc)
        edit_scores.append((dist, label))

edit_scores.sort(reverse=True)
for dist, label in edit_scores[:8]:
    print(f'  {label}: edit_dist={dist:.3f}')
