"""Aggregate per-chapter _stats.json files into a single summary for Table 11.
Run AFTER re-running all methods.

Usage: python scripts/_aggregate_stats.py
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

METHODS = {
    'rule': 'predictions/rule',
    'llm': 'predictions/llm',
    'hybrid': 'predictions/hybrid',
    'ours': 'predictions/ours',
}

print("=" * 80)
print("TABLE 11: Computational Cost (from actual _stats.json files)")
print("=" * 80)
print(f"{'Method':<10} {'LLM Calls':>10} {'Prompt Tok':>12} {'Compl Tok':>12} "
      f"{'Total Tok':>12} {'Wall Time':>10}")
print("-" * 70)

for method, pred_dir in METHODS.items():
    agg = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
           "total_tokens": 0, "wall_time_seconds": 0}

    stats_files = sorted(Path(pred_dir).glob("*_stats.json"))

    if not stats_files:
        # Rule has no stats files
        if method == 'rule':
            print(f"{method:<10} {'0':>10} {'0':>12} {'0':>12} {'0':>12} {'<10s':>10}")
        else:
            print(f"{method:<10} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
        continue

    for sf in stats_files:
        with sf.open('r', encoding='utf-8') as f:
            s = json.load(f)
        agg["llm_calls"] += s.get("llm_calls", 0)
        agg["prompt_tokens"] += s.get("prompt_tokens", 0)
        agg["completion_tokens"] += s.get("completion_tokens", 0)
        agg["total_tokens"] += s.get("total_tokens", 0)
        agg["wall_time_seconds"] += s.get("wall_time_seconds", 0)

    wt = agg["wall_time_seconds"]
    wt_str = f"{int(wt//60)}m{int(wt%60)}s"

    print(f"{method:<10} {agg['llm_calls']:>10,} {agg['prompt_tokens']:>12,} "
          f"{agg['completion_tokens']:>12,} {agg['total_tokens']:>12,} {wt_str:>10}")

    # Also show per-chapter detail
    print(f"  ({len(stats_files)} chapter stats files found)")

print("\nNote: These are REAL numbers from API responses, not estimates.")
