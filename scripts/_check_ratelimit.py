"""Check which chapters have valid stats vs rate-limited (0 tokens)."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

for method in ['llm', 'hybrid', 'ours']:
    print(f"\n{'='*50}")
    print(f"  {method.upper()}")
    print(f"{'='*50}")
    stats_dir = Path(f'predictions/{method}')
    total_calls = 0
    total_tokens = 0
    for ch in range(1, 27):
        sf = stats_dir / f'Chapter{ch}_pred_stats.json'
        if not sf.exists():
            print(f"  Ch{ch:>2}: NO STATS FILE")
            continue
        s = json.load(sf.open('r', encoding='utf-8'))
        calls = s.get('llm_calls', 0)
        tokens = s.get('total_tokens', 0)
        total_calls += calls
        total_tokens += tokens
        if tokens == 0 and method != 'rule':
            print(f"  Ch{ch:>2}: ⚠️  calls={calls}, tokens={tokens} — RATE LIMITED")
        else:
            print(f"  Ch{ch:>2}: ✅ calls={calls}, tokens={tokens:,}")
    print(f"  TOTAL: calls={total_calls}, tokens={total_tokens:,}")
