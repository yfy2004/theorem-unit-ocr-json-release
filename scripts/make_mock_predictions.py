import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_gold(unit, idx):
    out = dict(unit)
    out["unit_id"] = f"mock_{idx:05d}"
    out["confidence_score"] = 0.99
    return out


def drop_formulas(unit, idx):
    out = copy_gold(unit, idx)
    out["formula_spans"] = []
    out["confidence_score"] = 0.85
    return out


def truncate_blocks(unit, idx):
    out = copy_gold(unit, idx)
    blocks = list(out.get("source_blocks", []))
    if len(blocks) >= 2:
        out["source_blocks"] = blocks[:-1]
        out["reading_order"] = list(out["source_blocks"])
    out["confidence_score"] = 0.72
    return out


def minimal_noise(unit, idx, rng):
    out = copy_gold(unit, idx)
    text = out.get("statement_text", "")
    if len(text) > 20 and rng.random() < 0.5:
        out["statement_text"] = text.replace(",", "", 1)
    formulas = list(out.get("formula_spans", []))
    if formulas and rng.random() < 0.4:
        out["formula_spans"] = formulas[:-1]
    blocks = list(out.get("source_blocks", []))
    if len(blocks) >= 3 and rng.random() < 0.5:
        out["source_blocks"] = blocks[:-1]
        out["reading_order"] = list(out["source_blocks"])
    out["confidence_score"] = round(rng.uniform(0.55, 0.9), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["copy_gold", "drop_formulas", "truncate_blocks", "minimal_noise"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    gold = load_jsonl(Path(args.gold))
    rng = random.Random(args.seed)
    preds = []
    for idx, unit in enumerate(gold):
        if args.mode == "copy_gold":
            pred = copy_gold(unit, idx)
        elif args.mode == "drop_formulas":
            pred = drop_formulas(unit, idx)
        elif args.mode == "truncate_blocks":
            pred = truncate_blocks(unit, idx)
        else:
            pred = minimal_noise(unit, idx, rng)
        preds.append(pred)
    write_jsonl(Path(args.output), preds)
    print(f"Wrote {len(preds)} mock predictions to {args.output}")


if __name__ == "__main__":
    main()
