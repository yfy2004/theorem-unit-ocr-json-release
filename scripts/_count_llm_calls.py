"""Count actual LLM calls and tokens for all methods."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

FFFD = '\ufffd'

print("=" * 70)
print("ACCURATE LLM CALL & TOKEN COUNTS")
print("=" * 70)

# ===== Ours: count U+FFFD in ours_no_llm (BEFORE LLM repair) =====
print("\n--- Ours: U+FFFD before LLM repair (from ours_no_llm) ---")
total_units_with_fffd = 0
total_fffd_chars = 0
total_units = 0
for ch in range(1, 27):
    pred = json.load(open(f'predictions/ours_no_llm/Chapter{ch}_pred.json', 'r', encoding='utf-8'))
    total_units += len(pred)
    ch_fffd_units = 0
    ch_fffd_chars = 0
    for u in pred:
        content = u.get('content', '') + u.get('proof', '')
        n = content.count(FFFD)
        if n > 0:
            ch_fffd_units += 1
            ch_fffd_chars += n
    total_units_with_fffd += ch_fffd_units
    total_fffd_chars += ch_fffd_chars
    if ch_fffd_units > 0:
        print(f"  Ch{ch}: {ch_fffd_units} units w/ U+FFFD ({ch_fffd_chars} chars)")

print(f"\n  TOTAL: {total_units_with_fffd} / {total_units} units need LLM repair")
print(f"  TOTAL U+FFFD chars to fix: {total_fffd_chars}")

# ===== After repair: how many remain? =====
print("\n--- Ours: U+FFFD AFTER LLM repair ---")
remaining = 0
remaining_units = 0
for ch in range(1, 27):
    pred = json.load(open(f'predictions/ours/Chapter{ch}_pred.json', 'r', encoding='utf-8'))
    for u in pred:
        content = u.get('content', '') + u.get('proof', '')
        n = content.count(FFFD)
        if n > 0:
            remaining_units += 1
            remaining += n
print(f"  Remaining: {remaining_units} units, {remaining} U+FFFD chars")

# ===== All methods: U+FFFD in predictions =====
print("\n--- All methods: U+FFFD in final predictions ---")
for method in ['rule', 'llm', 'hybrid', 'ours']:
    total_fffd = 0
    units_with = 0
    total = 0
    for ch in range(1, 27):
        pred = json.load(open(f'predictions/{method}/Chapter{ch}_pred.json', 'r', encoding='utf-8'))
        total += len(pred)
        for u in pred:
            content = u.get('content', '') + u.get('proof', '')
            n = content.count(FFFD)
            if n > 0:
                units_with += 1
                total_fffd += n
    print(f"  {method:>8}: {units_with:>4} units w/ U+FFFD, {total_fffd:>5} chars total (out of {total} units)")

# ===== Hybrid: actual batch count from run =====
print("\n--- Hybrid: LLM call count ---")
hybrid_batches = 0
for ch in range(1, 27):
    pred = json.load(open(f'predictions/hybrid/Chapter{ch}_pred.json', 'r', encoding='utf-8'))
    n = len(pred)
    batches = (n + 7) // 8  # batch_size=8
    hybrid_batches += batches
print(f"  Total batches (batch_size=8): {hybrid_batches}")

# ===== LLM-only: page count =====
print("\n--- LLM-only: LLM call count ---")
total_pages = 0
for ch in range(1, 27):
    ocr_dir = Path(f'real_data/Klenke14/OCR/Chapter{ch}')
    pages = len(list(ocr_dir.glob('p*.json')))
    total_pages += pages
print(f"  Total pages (1 call per page): {total_pages}")

# ===== Token estimation explanation =====
print("\n--- Token estimation ---")
print("  Note: We don't have actual API token logs.")
print("  The 'Est. Input Tokens' in Table 11 was chars/4, which is a ROUGH estimate.")
print("  For accurate numbers, we'd need to log tiktoken counts during API calls.")
