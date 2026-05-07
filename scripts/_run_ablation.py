"""Run Ours WITHOUT LLM symbol repair (ablation: Norm+Rule only)."""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path('scripts')))
from normalize_blocks import normalize_page_blocks
from rule_based import extract_units_from_page

for ch in range(1, 27):
    ocr_dir = Path(f'real_data/Klenke14/OCR/Chapter{ch}')
    page_files = sorted(ocr_dir.glob('p*.json'))
    chapter_number = ch

    # Stage 1: Normalization
    pages = []
    for pf in page_files:
        with pf.open('r', encoding='utf-8') as f:
            page_data = json.load(f)
        normalize_page_blocks(page_data)
        pages.append(page_data)

    # Stage 3: Rule extraction (NO LLM)
    all_units = []
    section_title = ""
    section_number = 0
    for page_data in pages:
        units, section_title, section_number = extract_units_from_page(
            page_data, chapter_number, section_title, section_number
        )
        all_units.extend(units)

    # Deduplicate
    seen = set()
    deduped = []
    for u in all_units:
        key = u["label"]
        if key not in seen:
            seen.add(key)
            deduped.append(u)

    # Write output
    out_path = Path(f'predictions/ours_no_llm/Chapter{ch}_pred.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f'Ch{ch}: {len(deduped)} units')

print('\nDone! Ablation predictions saved to predictions/ours_no_llm/')
