"""Run NEW Hybrid (Rule + LLM repair/diff) on all 26 chapters."""
import subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')

for ch in range(1, 27):
    ocr_dir = f'real_data/Klenke14/OCR/Chapter{ch}'
    output = f'predictions/hybrid/Chapter{ch}_pred.json'
    print(f'\n{"="*50}')
    print(f'Chapter {ch}')
    print(f'{"="*50}')
    result = subprocess.run(
        ['python', 'scripts/hybrid_based.py', '--ocr_dir', ocr_dir, '--output', output,
         '--batch_size', '8', '--delay', '0.15'],
        text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        print(f'Ch{ch} FAILED (exit code {result.returncode})')
