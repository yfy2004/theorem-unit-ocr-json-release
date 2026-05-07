#!/usr/bin/env python3
"""
eval_micro.py
=============
Micro-averaged evaluation across multiple chapters.
Pools all units and computes one set of scores for the paper's Table 2.

Usage:
  python scripts/eval_micro.py --method rule --chapters 1,2,3,4,5
  python scripts/eval_micro.py --method hybrid --chapters 1,2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_v2 import (
    match_units,
    calc_boundary,
    calc_type,
    calc_completeness,
    calc_edit,
    calc_formula_attach,
    calc_downstream,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="rule / llm / hybrid / ours")
    parser.add_argument("--chapters", default="1,2,3,4,5", help="comma-separated chapter numbers")
    parser.add_argument("--gold_dir", default="Klenke14_revised/json")
    parser.add_argument("--pred_dir", default=None, help="override predictions dir")
    args = parser.parse_args()

    chapters = [int(c.strip()) for c in args.chapters.split(",")]
    pred_dir = args.pred_dir or f"predictions/{args.method}"

    all_gold = []
    all_pred = []

    for ch in chapters:
        gold_path = Path(args.gold_dir) / f"Chapter{ch}.json"
        pred_path = Path(pred_dir) / f"Chapter{ch}_pred.json"

        if not gold_path.exists():
            print(f"  Warning: {gold_path} not found, skipping Chapter {ch}")
            continue
        if not pred_path.exists():
            print(f"  Warning: {pred_path} not found, skipping Chapter {ch}")
            continue

        with gold_path.open("r", encoding="utf-8") as f:
            gold = json.load(f)
        with pred_path.open("r", encoding="utf-8") as f:
            pred = json.load(f)

        all_gold.extend(gold)
        all_pred.extend(pred)

    if not all_gold:
        print("No data loaded.")
        return

    # Match
    matched, unmatched_gold, extra_pred = match_units(all_gold, all_pred)

    # Structure
    boundary = calc_boundary(matched, len(all_gold), len(all_pred))
    type_acc = calc_type(matched)

    # Content
    compl = calc_completeness(matched)
    edit = calc_edit(matched)

    # Formula
    f_attach = calc_formula_attach(matched)

    # Downstream
    downstream = calc_downstream([p for _, p in matched])

    # Print
    method_name = args.method.upper()
    ch_str = ",".join(str(c) for c in chapters)
    print(f"=== {method_name} Micro-averaged (Ch {ch_str}) ===")
    print(f"  Units: Gold={len(all_gold)}, Pred={len(all_pred)}, Matched={len(matched)}")
    print(f"  Unmatched Gold: {len(unmatched_gold)}, Extra Pred: {len(extra_pred)}")
    print(f"--- Structure ---")
    print(f"  B-P:  {boundary['B-P']}")
    print(f"  B-R:  {boundary['B-R']}")
    print(f"  B-F1: {boundary['B-F1']}")
    print(f"  Type: {type_acc}")
    print(f"--- Content ---")
    print(f"  Compl: {compl}")
    print(f"  Edit:  {edit}")
    print(f"--- Formula ---")
    print(f"  F-Att: {f_attach}")
    print(f"--- Downstream ---")
    print(f"  Parse:     {downstream['Parse']}")
    print(f"  Symbol:    {downstream['Symbol']}")
    print(f"  Retrieval: {downstream['Retrieval']}")
    print(f"  Ready:     {downstream['Ready']}")

    # One-line summary for table
    print(f"\n  TABLE ROW: {method_name} | {boundary['B-F1']} | {type_acc} | "
          f"{compl} | {edit} | {f_attach} | "
          f"{downstream['Parse']} | {downstream['Symbol']} | {downstream['Retrieval']} | "
          f"{downstream['Ready']}")


if __name__ == "__main__":
    main()
