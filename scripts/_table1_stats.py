"""Correctly compute Table 1 metrics from predictions' _source_blocks."""
import json, sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

FFFD = '\ufffd'
FORMULA_RE = re.compile(r'\$[^$]+\$|\$\$.*?\$\$', re.DOTALL)

# Use rule predictions since they have _source_blocks
blocks_per_unit = []
formulas_per_unit = []
total_units = 0

for ch in range(1, 27):
    pf = Path(f'predictions/rule/Chapter{ch}_pred.json')
    pred = json.load(pf.open('r', encoding='utf-8'))
    for u in pred:
        total_units += 1
        # Blocks per unit
        src = u.get('_source_blocks', [])
        proof_b = u.get('_proof_blocks', [])
        all_blocks = list(src) + list(proof_b)
        blocks_per_unit.append(len(all_blocks))

        # Formulas per unit
        content = u.get('content', '') + ' ' + u.get('proof', '')
        formulas = FORMULA_RE.findall(content)
        formulas_per_unit.append(len(formulas))

avg_blocks = sum(blocks_per_unit) / len(blocks_per_unit)
multi_block = sum(1 for b in blocks_per_unit if b > 1)
multi_block_ratio = multi_block / len(blocks_per_unit) * 100
avg_formulas = sum(formulas_per_unit) / len(formulas_per_unit)

print(f"Total units: {total_units}")
print(f"Avg blocks/unit: {avg_blocks:.1f}")
print(f"Multi-block units: {multi_block} / {total_units} = {multi_block_ratio:.1f}%")
print(f"Avg formulas/unit: {avg_formulas:.1f}")

# Distribution
from collections import Counter
bc = Counter(blocks_per_unit)
print(f"\nBlocks/unit distribution:")
for k in sorted(bc.keys()):
    print(f"  {k} blocks: {bc[k]} units ({bc[k]/total_units*100:.1f}%)")
