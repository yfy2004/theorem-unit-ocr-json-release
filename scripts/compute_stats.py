import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

THEOREM_LIKE_TYPES = {"theorem", "lemma", "proposition", "corollary"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_split(manifest_rows, units):
    pages = len(manifest_rows)
    ocr_blocks = sum(int(r["ocr_block_count"]) for r in manifest_rows if r.get("ocr_block_count"))
    theorem_units = len(units)
    definitions = sum(1 for u in units if u.get("unit_type") == "definition")
    theorem_like = sum(1 for u in units if u.get("unit_type") in THEOREM_LIKE_TYPES)
    examples = sum(1 for u in units if u.get("unit_type") == "example")
    blocks_per_unit = [len(u.get("source_blocks", [])) for u in units]
    formulas_per_unit = [len(u.get("formula_spans", [])) for u in units]
    multi_block_ratio = sum(1 for u in units if len(u.get("source_blocks", [])) > 1) / theorem_units if theorem_units > 0 else 0.0
    return {
        "pages": pages,
        "ocr_blocks": ocr_blocks,
        "theorem_units": theorem_units,
        "definitions": definitions,
        "theorem_like": theorem_like,
        "examples": examples,
        "avg_blocks_per_unit": round(mean(blocks_per_unit), 1) if blocks_per_unit else 0.0,
        "avg_formulas_per_unit": round(mean(formulas_per_unit), 1) if formulas_per_unit else 0.0,
        "multi_block_unit_ratio": round(multi_block_ratio, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--gold_train", required=True)
    parser.add_argument("--gold_dev", required=True)
    parser.add_argument("--gold_test", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest_csv))
    split_rows = defaultdict(list)
    for row in manifest:
        split_rows[row["split"]].append(row)

    gold_train = load_jsonl(Path(args.gold_train))
    gold_dev = load_jsonl(Path(args.gold_dev))
    gold_test = load_jsonl(Path(args.gold_test))
    gold_total = gold_train + gold_dev + gold_test

    stats = {
        "train": summarize_split(split_rows["train"], gold_train),
        "dev": summarize_split(split_rows["dev"], gold_dev),
        "test": summarize_split(split_rows["test"], gold_test),
        "total": summarize_split(manifest, gold_total),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split","pages","ocr_blocks","theorem_units","definitions","theorem_like","examples","avg_blocks_per_unit","avg_formulas_per_unit","multi_block_unit_ratio"])
        for split in ["train", "dev", "test", "total"]:
            s = stats[split]
            writer.writerow([split, s["pages"], s["ocr_blocks"], s["theorem_units"], s["definitions"], s["theorem_like"], s["examples"], s["avg_blocks_per_unit"], s["avg_formulas_per_unit"], s["multi_block_unit_ratio"]])

    print(f"Wrote stats to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
