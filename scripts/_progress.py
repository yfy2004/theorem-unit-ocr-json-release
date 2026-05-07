"""Check pipeline progress — which chapters have _stats.json files."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 50)
print("  PIPELINE PROGRESS")
print("=" * 50)

for method in ['llm', 'hybrid', 'ours']:
    stats = sorted(Path(f'predictions/{method}').glob('*_stats.json'))
    done = [f.stem.replace('_pred_stats', '') for f in stats]
    print(f"\n  {method.upper():>8}: {len(done)}/26 chapters")
    if done:
        print(f"           Done: {', '.join(done)}")
    else:
        print(f"           Not started")

print()
