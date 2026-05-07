#!/usr/bin/env python3
"""Compare Rule's block merging vs what Normalization would fix."""
import json, sys, os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Analyze multiple chapters for block-level issues
for ch in range(1, 6):
    ocr_dir = Path(f'real_data/Klenke14/OCR/Chapter{ch}')
    if not ocr_dir.exists():
        continue
    
    total_blocks = 0
    fragmented = 0      # Blocks that look like they were split wrongly
    ordering_issues = 0  # Blocks whose bbox suggests wrong reading order
    tiny_blocks = 0      # Suspiciously small blocks (< 10 chars)
    
    for pf in sorted(ocr_dir.glob('p*.json')):
        with pf.open('r', encoding='utf-8') as f:
            page = json.load(f)
        
        blocks = page.get('blocks', [])
        total_blocks += len(blocks)
        
        for k, b in enumerate(blocks):
            text = b.get('text', '')
            bbox = b.get('bbox', [0,0,0,0])
            
            # Check tiny blocks (fragmentation indicator)
            if 0 < len(text.strip()) < 10 and b.get('type') != 'image':
                tiny_blocks += 1
            
            # Check reading order: block k+1 should be below or right of block k
            if k + 1 < len(blocks):
                next_b = blocks[k + 1]
                next_bbox = next_b.get('bbox', [0,0,0,0])
                # If next block's top is ABOVE current block's top by > 20px
                # that's a reading order issue
                if next_bbox[1] < bbox[1] - 20 and next_bbox[1] > 50:
                    ordering_issues += 1
        
        # Check for consecutive text blocks with similar x-position 
        # that could be merged
        for k in range(len(blocks) - 1):
            b1 = blocks[k]
            b2 = blocks[k + 1]
            if b1.get('type') in ('text', 'formula') and b2.get('type') in ('text', 'formula'):
                bbox1 = b1.get('bbox', [0,0,0,0])
                bbox2 = b2.get('bbox', [0,0,0,0])
                # Same x-alignment, adjacent vertically (gap < 5px)
                if abs(bbox1[0] - bbox2[0]) < 5 and abs(bbox2[1] - bbox1[3]) < 5:
                    t1 = b1.get('text', '').strip()
                    t2 = b2.get('text', '').strip()
                    # Check if b1 ends without sentence ending
                    if t1 and not t1[-1] in '.!?:;' and t2 and t2[0].islower():
                        fragmented += 1
    
    print(f'Chapter {ch}: {total_blocks} blocks')
    print(f'  Tiny blocks (<10 chars): {tiny_blocks}')
    print(f'  Reading order issues:     {ordering_issues}')
    print(f'  Fragmented (mergeable):   {fragmented}')
    print()

# Show specific examples from Chapter 2+
print('=== Example fragmented blocks (Chapter 2) ===')
ocr_dir = Path('real_data/Klenke14/OCR/Chapter2')
for pf in sorted(ocr_dir.glob('p*.json'))[:5]:
    with pf.open('r', encoding='utf-8') as f:
        page = json.load(f)
    blocks = page.get('blocks', [])
    for k, b in enumerate(blocks):
        text = b.get('text', '').strip()
        if 0 < len(text) < 10 and b.get('type') != 'image':
            print(f'  {page["page_id"]} block {b["block_id"]}: type={b["type"]}, text="{text}"')
