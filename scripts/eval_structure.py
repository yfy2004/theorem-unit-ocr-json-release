import argparse
import json
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


def boundary_key(unit: dict[str, Any]) -> tuple[str, int, int]:
    blocks = unit.get("source_blocks", [])
    if not blocks:
        return (unit["page_id"], -1, -1)
    return (unit["page_id"], min(blocks), max(blocks))


def block_set(unit: dict[str, Any]) -> set[int]:
    return set(unit.get("source_blocks", []))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def evaluate_structure(gold, pred):
    gold_by_boundary = {boundary_key(u): u for u in gold}
    pred_by_boundary = {boundary_key(u): u for u in pred}
    gold_keys = set(gold_by_boundary.keys())
    pred_keys = set(pred_by_boundary.keys())
    matched = gold_keys & pred_keys
    precision = len(matched) / len(pred_keys) if pred_keys else 0.0
    recall = len(matched) / len(gold_keys) if gold_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    type_acc = 0.0
    block_acc = 0.0
    if matched:
        type_acc = sum(1 for k in matched if gold_by_boundary[k].get("unit_type") == pred_by_boundary[k].get("unit_type")) / len(matched)
        block_acc = sum(jaccard(block_set(gold_by_boundary[k]), block_set(pred_by_boundary[k])) for k in matched) / len(matched)
    return {
        "boundary_precision": round(precision * 100, 2),
        "boundary_recall": round(recall * 100, 2),
        "boundary_f1": round(f1 * 100, 2),
        "unit_type_accuracy": round(type_acc * 100, 2),
        "block_assignment_accuracy": round(block_acc * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    results = evaluate_structure(gold, pred)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
