import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

VALID_TYPES = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "other"}


def parse_int_list(text: str) -> list[int]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"[,\s;]+", text)
    out = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def split_formula_texts(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("||")]
    return [p for p in parts if p]


def normalize_whitespace(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def build_formula_spans(page_id: str, formula_texts: list[str], formula_blocks: list[int]) -> list[dict[str, Any]]:
    if not formula_texts and not formula_blocks:
        return []
    if len(formula_texts) != len(formula_blocks):
        raise ValueError(f"Formula text count ({len(formula_texts)}) != formula block count ({len(formula_blocks)}) for page_id={page_id}")
    spans = []
    for i, (txt, blk) in enumerate(zip(formula_texts, formula_blocks), start=1):
        spans.append({"formula_id": f"f_{page_id}_{i:02d}", "text": normalize_whitespace(txt), "source_block": int(blk)})
    return spans


def row_to_unit(row: dict[str, str], doc_id: str, strict: bool = True):
    status = (row.get("status") or "").strip().lower()
    if status and status not in {"done", "reviewed"}:
        return None
    page_id = (row.get("page_id") or "").strip()
    unit_id = (row.get("unit_id") or "").strip()
    unit_type = (row.get("unit_type") or "").strip().lower()
    title = normalize_whitespace(row.get("title") or "")
    statement_text = normalize_whitespace(row.get("statement_text") or "")
    notes = normalize_whitespace(row.get("notes") or "")
    if not page_id:
        raise ValueError("Missing page_id")
    if not unit_id:
        raise ValueError(f"Missing unit_id for page_id={page_id}")
    if unit_type not in VALID_TYPES:
        raise ValueError(f"Invalid unit_type='{unit_type}' for unit_id={unit_id}")
    if not statement_text:
        raise ValueError(f"Empty statement_text for unit_id={unit_id}")
    source_blocks = parse_int_list(row.get("source_blocks") or "")
    reading_order = parse_int_list(row.get("reading_order") or "")
    if not source_blocks:
        raise ValueError(f"Empty source_blocks for unit_id={unit_id}")
    if not reading_order:
        reading_order = list(source_blocks)
    formula_texts = split_formula_texts(row.get("formula_span_texts") or "")
    formula_blocks = parse_int_list(row.get("formula_source_blocks") or "")
    formula_spans = build_formula_spans(page_id, formula_texts, formula_blocks)
    unit = {"doc_id": doc_id, "page_id": page_id, "unit_id": unit_id, "unit_type": unit_type, "title": title, "statement_text": statement_text, "formula_spans": formula_spans, "source_blocks": source_blocks, "reading_order": reading_order, "notes": notes}
    if not strict:
        unit["_annotation_confidence"] = (row.get("annotation_confidence") or "").strip()
        unit["_is_detached_heading"] = (row.get("is_detached_heading") or "").strip()
        unit["_is_proof_adjacent"] = (row.get("is_proof_adjacent") or "").strip()
    return unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--doc_id", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows_out, skipped = [], 0
    with Path(args.input_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            try:
                unit = row_to_unit(row, doc_id=args.doc_id, strict=args.strict)
                if unit is None:
                    skipped += 1
                    continue
                rows_out.append(unit)
            except Exception as e:
                if args.strict:
                    raise
                skipped += 1
                print(f"[WARN] Skip row {line_no}: {e}")
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for unit in rows_out:
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows_out)} units to {output_jsonl}")
    print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
