"""Compute dataset stats for paper tables."""
import json, sys
from pathlib import Path
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')

gold_dir = Path('Klenke14_revised/json')
ocr_dir = Path('real_data/Klenke14/OCR')

total_pages = 0
total_blocks = 0
total_units = 0
env_cnt = Counter()

for ch in range(1, 27):
    gf = gold_dir / f'Chapter{ch}.json'
    gold = json.load(gf.open('r', encoding='utf-8'))
    units = len(gold)
    total_units += units

    ch_ocr = ocr_dir / f'Chapter{ch}'
    pages = len(list(ch_ocr.glob('p*.json')))
    total_pages += pages

    blocks = 0
    for pf in sorted(ch_ocr.glob('p*.json')):
        pd = json.load(pf.open('r', encoding='utf-8'))
        blocks += len(pd.get('blocks', []))
    total_blocks += blocks

    for u in gold:
        env_cnt[u.get('env', '?')] += 1

# Classify
defs = env_cnt.get('def', 0)
thm_like = env_cnt.get('thm', 0) + env_cnt.get('lem', 0) + env_cnt.get('cor', 0) + env_cnt.get('prop', 0)
examples = env_cnt.get('ex', 0)
exercises = env_cnt.get('exr', 0)
remarks = env_cnt.get('rem', 0)

print(f"Total pages: {total_pages}")
print(f"Total OCR blocks: {total_blocks}")
print(f"Total theorem units: {total_units}")
print(f"Definitions: {defs}")
print(f"Theorem-like (thm+lem+cor+prop): {thm_like}")
print(f"  - Theorems: {env_cnt.get('thm', 0)}")
print(f"  - Lemmas: {env_cnt.get('lem', 0)}")
print(f"  - Corollaries: {env_cnt.get('cor', 0)}")
print(f"  - Propositions: {env_cnt.get('prop', 0)}")
print(f"Examples: {examples}")
print(f"Exercises: {exercises}")
print(f"Remarks: {remarks}")
print(f"\nAll env types: {dict(env_cnt)}")
print(f"Avg blocks/unit: {total_blocks / total_units:.1f}")
print(f"Avg units/page: {total_units / total_pages:.1f}")
