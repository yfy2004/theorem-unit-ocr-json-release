"""Run Ours method on all 26 chapters sequentially."""
import subprocess, sys
sys.stdout.reconfigure(encoding='utf-8')

for ch in range(1, 27):
    ocr_dir = f'real_data/Klenke14/OCR/Chapter{ch}'
    output = f'predictions/ours/Chapter{ch}_pred.json'
    print(f'\n{"="*50}')
    print(f'Chapter {ch}')
    print(f'{"="*50}')
    result = subprocess.run(
        ['python', 'scripts/ours_based.py', '--ocr_dir', ocr_dir, '--output', output],
        text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        print(f'Ch{ch} FAILED (exit code {result.returncode})')
