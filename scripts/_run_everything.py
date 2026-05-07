"""
ONE-SHOT: Run ALL methods + evaluate + compute all tables.

Usage:
    python scripts/_run_everything.py              # run all 4 methods + eval
    python scripts/_run_everything.py --eval-only   # only re-evaluate (don't re-run methods)
    python scripts/_run_everything.py --methods llm hybrid ours  # only re-run specific methods
"""
import subprocess, sys, time, argparse
sys.stdout.reconfigure(encoding='utf-8')

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True, encoding='utf-8')
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.0f}s")
    return result.returncode == 0

parser = argparse.ArgumentParser()
parser.add_argument('--eval-only', action='store_true', help='Skip re-running methods, only evaluate')
parser.add_argument('--methods', nargs='*', default=None, help='Specific methods to re-run: rule llm hybrid ours')
args = parser.parse_args()

methods_to_run = args.methods or ['rule', 'llm', 'hybrid', 'ours']
all_ok = True

if not args.eval_only:
    # Step 1: Run methods
    if 'rule' in methods_to_run:
        for ch in range(1, 27):
            ok = run(['python', 'scripts/rule_based.py',
                      '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                      '--output', f'predictions/rule/Chapter{ch}_pred.json'],
                     f'Rule — Chapter {ch}')
            if not ok:
                all_ok = False

    if 'llm' in methods_to_run:
        for ch in range(1, 27):
            ok = run(['python', 'scripts/llm_based.py',
                      '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                      '--output', f'predictions/llm/Chapter{ch}_pred.json',
                      '--delay', '0.3'],
                     f'LLM — Chapter {ch}')
            if not ok:
                all_ok = False

    if 'hybrid' in methods_to_run:
        for ch in range(1, 27):
            ok = run(['python', 'scripts/hybrid_based.py',
                      '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                      '--output', f'predictions/hybrid/Chapter{ch}_pred.json',
                      '--batch_size', '8', '--delay', '0.15'],
                     f'Hybrid — Chapter {ch}')
            if not ok:
                all_ok = False

    if 'ours' in methods_to_run:
        for ch in range(1, 27):
            ok = run(['python', 'scripts/ours_based.py',
                      '--ocr_dir', f'real_data/Klenke14/OCR/Chapter{ch}',
                      '--output', f'predictions/ours/Chapter{ch}_pred.json'],
                     f'Ours — Chapter {ch}')
            if not ok:
                all_ok = False

# Step 2: Ablation (Norm + Rule, no LLM) — always re-run, it's fast
run(['python', 'scripts/_run_ablation.py'], 'Ablation (Norm+Rule, no LLM)')

# Step 3: Compute all tables
run(['python', 'scripts/_compute_all_tables.py'], 'Compute All Tables')

if all_ok:
    print("\n✅ ALL DONE — all methods ran successfully + tables computed.")
else:
    print("\n⚠️ DONE — some methods failed. Check output above.")
