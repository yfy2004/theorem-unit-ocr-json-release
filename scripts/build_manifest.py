import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_blocks(ocr_obj: dict[str, Any]) -> int:
    for key in ["blocks", "items", "layout_blocks", "elements"]:
        if key in ocr_obj and isinstance(ocr_obj[key], list):
            return len(ocr_obj[key])
    return 0


def extract_blocks(ocr_obj: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["blocks", "items", "layout_blocks", "elements"]:
        if key in ocr_obj and isinstance(ocr_obj[key], list):
            return ocr_obj[key]
    return []


def guess_formula_dense(blocks: list[dict[str, Any]]) -> int:
    math_like = 0
    total = 0
    pattern = re.compile(r"[=∑∫∀∃≤≥λμσπ]|\\[a-zA-Z]+|\bmath\b", re.I)
    for b in blocks:
        text = str(b.get("text", ""))
        total += 1
        if pattern.search(text):
            math_like += 1
    if total == 0:
        return 0
    return int(math_like / total >= 0.2)


def guess_proof_adjacent(blocks: list[dict[str, Any]]) -> int:
    text_all = " ".join(str(b.get("text", "")) for b in blocks).lower()
    return int("proof" in text_all)


def infer_page_id(json_path: Path) -> str:
    return json_path.stem


def infer_pdf_page_no(page_id: str) -> str:
    m = re.search(r"(\d+)$", page_id)
    return m.group(1) if m else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", required=True)
    parser.add_argument("--ocr_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--default_split", default="")
    args = parser.parse_args()

    ocr_dir = Path(args.ocr_dir)
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)

    json_files = sorted(ocr_dir.glob("*.json"))
    rows = []

    for json_file in json_files:
        page_id = infer_page_id(json_file)
        image_candidates = list(image_dir.glob(f"{page_id}.*"))
        image_path = image_candidates[0] if image_candidates else None

        ocr_obj = load_json(json_file)
        blocks = extract_blocks(ocr_obj)
        row = {
            "doc_id": args.doc_id,
            "chapter_id": "",
            "page_id": page_id,
            "pdf_page_no": infer_pdf_page_no(page_id),
            "image_path": str(image_path) if image_path else "",
            "ocr_json_path": str(json_file),
            "split": args.default_split,
            "is_challenge": 0,
            "has_formula_dense": guess_formula_dense(blocks),
            "has_proof_adjacent": guess_proof_adjacent(blocks),
            "ocr_block_count": count_blocks(ocr_obj),
            "notes": "",
        }
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id", "chapter_id", "page_id", "pdf_page_no", "image_path",
                "ocr_json_path", "split", "is_challenge", "has_formula_dense",
                "has_proof_adjacent", "ocr_block_count", "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
