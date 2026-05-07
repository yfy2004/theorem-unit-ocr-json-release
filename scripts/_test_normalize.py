"""Run Rule + Normalization on a chapter and save output."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'scripts')
from pathlib import Path
from normalize_blocks import normalize_page_blocks
from rule_based import extract_units_from_page
import re

chapter = int(sys.argv[1]) if len(sys.argv) > 1 else 2

ocr_dir = Path('real_data/Klenke14/OCR/Chapter%d' % chapter)
page_files = sorted(ocr_dir.glob('p*.json'))

all_units = []
section_title = ''
section_number = 0

for pf in page_files:
    with pf.open('r', encoding='utf-8') as f:
        page_data = json.load(f)
    
    page_data = normalize_page_blocks(page_data)
    
    units, section_title, section_number = extract_units_from_page(
        page_data, chapter, section_title, section_number
    )
    all_units.extend(units)

# Deduplicate
seen = {}
for u in all_units:
    label = u.get('label', '')
    if label and (label not in seen or len(u.get('content','')) > len(seen[label].get('content',''))):
        seen[label] = u

merged = sorted(seen.values(), key=lambda u: u.get('number_components', [0]))
for i, u in enumerate(merged, 1):
    u['index'] = i

out_path = 'predictions/rule/Chapter%d_norm_pred.json' % chapter
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print('Rule+Norm Chapter %d: %d units -> %s' % (chapter, len(merged), out_path))
