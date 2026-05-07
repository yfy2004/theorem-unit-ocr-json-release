"""
Resume from breakpoint — only re-run chapters with 0 tokens (rate-limited).
Adds delay between chapters to avoid hitting rate limits again.

Usage:
    python scripts/_resume_failed.py                    # run all failed
    python scripts/_resume_failed.py --methods llm      # only LLM
    python scripts/_resume_failed.py --methods hybrid ours  # only Hybrid+Ours
"""
import subprocess, sys, json, time, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='*', default=['llm', 'hybrid', 'ours'])
parser.add_argument('--chapter-delay', type=int, default=5,
                    help='Seconds to wait between chapters (default 5)')
args = parser.parse_args()

COMMANDS = {
    'llm': lambda ch: ['python', 'scripts/llm_based.py',
                        '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                        '--output', f'predictions/llm/Chapter{ch}_pred.json',
                        '--delay', '0.5'],
    'hybrid': lambda ch: ['python', 'scripts/hybrid_based.py',
                           '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                           '--output', f'predictions/hybrid/Chapter{ch}_pred.json',
                           '--batch_size', '8', '--delay', '0.3'],
    'ours': lambda ch: ['python', 'scripts/ours_based.py',
                         '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                         '--output', f'predictions/ours/Chapter{ch}_pred.json'],
}

# Stats file naming differs by method
def stats_path(method, ch):
    d = Path(f'predictions/{method}')
    # Try both naming conventions
    for name in [f'Chapter{ch}_pred_stats.json', f'Chapter{ch}_stats.json']:
        p = d / name
        if p.exists():
            return p
    return d / f'Chapter{ch}_pred_stats.json'

def needs_rerun(method, ch):
    sf = stats_path(method, ch)
    if not sf.exists():
        return True
    try:
        s = json.load(sf.open('r', encoding='utf-8'))
        return s.get('total_tokens', 0) == 0
    except:
        return True

for method in args.methods:
    failed = [ch for ch in range(1, 27) if needs_rerun(method, ch)]
    if not failed:
        print(f"\n✅ {method.upper()}: all 26 chapters have valid data, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"  {method.upper()}: Re-running {len(failed)} failed chapters")
    print(f"  Chapters: {failed}")
    print(f"{'='*60}")

    for i, ch in enumerate(failed):
        print(f"\n--- {method.upper()} Chapter {ch} ({i+1}/{len(failed)}) ---")
        cmd = COMMANDS[method](ch)
        t0 = time.time()
        result = subprocess.run(cmd, text=True, encoding='utf-8')
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  ✗ FAILED (exit {result.returncode}) — {elapsed:.0f}s")
        else:
            # Verify it actually got data
            sf = stats_path(method, ch)
            if sf.exists():
                s = json.load(sf.open('r', encoding='utf-8'))
                tokens = s.get('total_tokens', 0)
                if tokens > 0:
                    print(f"  ✓ OK — {tokens:,} tokens — {elapsed:.0f}s")
                else:
                    print(f"  ⚠️ Completed but 0 tokens (still rate limited?) — {elapsed:.0f}s")
            else:
                print(f"  ⚠️ No stats file generated — {elapsed:.0f}s")

        # Wait between chapters to avoid rate limit
        if i < len(failed) - 1:
            print(f"  Waiting {args.chapter_delay}s before next chapter...")
            time.sleep(args.chapter_delay)

# Re-run ablation + tables
print(f"\n{'='*60}")
print("  Re-running Ablation + All Tables")
print(f"{'='*60}")
subprocess.run(['python', 'scripts/_run_ablation.py'], text=True, encoding='utf-8')
subprocess.run(['python', 'scripts/_compute_all_tables.py'], text=True, encoding='utf-8')

print("\n✅ Resume complete! Run 'python scripts/_check_ratelimit.py' to verify.")
