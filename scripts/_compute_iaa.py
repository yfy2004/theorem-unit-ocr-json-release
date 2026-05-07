"""Compute Inter-Annotator Agreement (IAA) between gold and double annotation."""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
from eval_v2 import match_units, extract_formula_tokens

GOLD_DIR = BASE / 'Klenke14_revised' / 'json'
DOUBLE_DIR = BASE / 'Klenke14_revised' / 'double__annotation'
OCR_DIR = BASE / 'real_data' / 'Klenke14' / 'OCR'
CHAPTERS = list(range(1, 27))

total_A, total_B, total_matched = 0, 0, 0
type_agree, type_total = 0, 0
formula_scores = []
content_similarities = []

for ch in CHAPTERS:
    gf = GOLD_DIR / f'Chapter{ch}.json'
    df = DOUBLE_DIR / f'Chapter{ch}_double_annotation.json'
    if not gf.exists() or not df.exists():
        print(f"  Ch{ch}: MISSING (gf={gf.exists()} df={df.exists()})")
        continue

    ann_a = json.load(gf.open('r', encoding='utf-8'))
    ann_b = json.load(df.open('r', encoding='utf-8'))

    matched_ab, a_only, b_only = match_units(ann_a, ann_b)
    n_a, n_b = len(ann_a), len(ann_b)
    n_matched = len(matched_ab)

    total_A += n_a
    total_B += n_b
    total_matched += n_matched

    # Type agreement
    for g, p in matched_ab:
        type_total += 1
        if g.get('env', '') == p.get('env', ''):
            type_agree += 1

    # Formula attachment agreement
    for g, p in matched_ab:
        g_formulas = extract_formula_tokens(g.get('content', ''))
        p_formulas = extract_formula_tokens(p.get('content', ''))

        if not g_formulas and not p_formulas:
            formula_scores.append(1.0)
            continue

        g_tokens = set()
        for ft in g_formulas:
            g_tokens |= ft
        p_tokens = set()
        for ft in p_formulas:
            p_tokens |= ft

        if not g_tokens and not p_tokens:
            formula_scores.append(1.0)
        elif not g_tokens or not p_tokens:
            formula_scores.append(0.0)
        else:
            overlap = len(g_tokens & p_tokens)
            prec = overlap / len(p_tokens) if p_tokens else 0
            rec = overlap / len(g_tokens) if g_tokens else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            formula_scores.append(f1)

    # Content similarity (token-level F1)
    for g, p in matched_ab:
        g_tok = set(re.findall(r'[a-zA-Z0-9]+', g.get('content', '').lower()))
        p_tok = set(re.findall(r'[a-zA-Z0-9]+', p.get('content', '').lower()))
        if not g_tok and not p_tok:
            content_similarities.append(1.0)
        elif not g_tok or not p_tok:
            content_similarities.append(0.0)
        else:
            overlap = len(g_tok & p_tok)
            prec = overlap / len(p_tok)
            rec = overlap / len(g_tok)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            content_similarities.append(f1)

    print(f"  Ch{ch:2d}: A={n_a:3d} B={n_b:3d} matched={n_matched:3d} unmatched_A={len(a_only):2d} unmatched_B={len(b_only):2d}")

# Overall metrics
print(f"\n{'='*60}")
print(f"INTER-ANNOTATOR AGREEMENT (IAA)")
print(f"{'='*60}")

prec = total_matched / total_B if total_B else 0
rec = total_matched / total_A if total_A else 0
bf1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"\nAnnotator A (gold):       {total_A} units")
print(f"Annotator B (pdf-only):   {total_B} units")
print(f"Matched pairs:            {total_matched}")
print(f"Unmatched A (FN):         {total_A - total_matched}")
print(f"Unmatched B (FP):         {total_B - total_matched}")

print(f"\nBoundary Precision: {prec*100:.1f}%")
print(f"Boundary Recall:    {rec*100:.1f}%")
print(f"Boundary F1:        {bf1*100:.1f}%")

type_pct = type_agree / type_total * 100 if type_total else 0
print(f"\nType agreement: {type_agree}/{type_total} = {type_pct:.1f}%")

avg_formula = sum(formula_scores) / len(formula_scores) * 100 if formula_scores else 0
print(f"Formula-attachment F1: {avg_formula:.1f}%")

avg_content = sum(content_similarities) / len(content_similarities) * 100 if content_similarities else 0
print(f"Content token-F1: {avg_content:.1f}%")

# Page count
total_pages = sum(len(list((OCR_DIR/f'Chapter{ch}').glob('p*.json')))
                  for ch in CHAPTERS if (OCR_DIR/f'Chapter{ch}').exists())

print(f"\n{'='*60}")
print(f"FOR PAPER:")
print(f"{'='*60}")
print(f"  N_pages = {total_pages}")
print(f"  N_units_A = {total_A}")
print(f"  N_units_B = {total_B}")
print(f"  Boundary F1 = {bf1*100:.1f}")
print(f"  Type agreement = {type_pct:.1f}%")
print(f"  Formula-attachment F1 = {avg_formula:.1f}%")
print(f"  Content token-F1 = {avg_content:.1f}%")
