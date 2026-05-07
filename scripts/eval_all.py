import argparse
import json
from pathlib import Path

from eval_structure import load_jsonl, evaluate_structure
from eval_integrity import evaluate_integrity
from eval_downstream import evaluate_downstream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system_name", default="")
    args = parser.parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    merged = {
        "system_name": args.system_name,
        "gold_path": args.gold,
        "pred_path": args.pred,
        "structure": evaluate_structure(gold, pred),
        "integrity": evaluate_integrity(gold, pred),
        "downstream": evaluate_downstream(gold, pred),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
