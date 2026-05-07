#!/usr/bin/env python3
"""
rule_based.py
=============
Rule-based theorem unit extraction from OCR block JSON.

This is the simplest baseline: pure pattern matching, no LLM, no learning.
It identifies theorem-like units by scanning for keywords in text blocks,
collects subsequent blocks as the theorem body, and classifies types
directly from the heading keyword.

Usage:
    python scripts/rule_based.py --ocr_dir real_data/Klenke14/OCR/Chapter1 \
                                  --output predictions/rule/Chapter1_pred.jsonl

    # Process all chapters:
    python scripts/rule_based.py --ocr_dir real_data/Klenke14/OCR \
                                  --output predictions/rule/pred_units.jsonl \
                                  --all
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Keyword patterns
# ---------------------------------------------------------------------------

# Map keyword → unit_type
KEYWORD_TO_TYPE = {
    "definition": "definition",
    "theorem": "theorem",
    "lemma": "lemma",
    "proposition": "proposition",
    "corollary": "corollary",
    "example": "example",
    "remark": "remark",
    "exercise": "exercise",
}

# Regex to detect theorem-like headings at the start of a block
THEOREM_HEADING_RE = re.compile(
    r"^\s*(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise)"
    r"\s*([\d.]+)?",
    re.IGNORECASE,
)

# Regex to detect proof starts
PROOF_START_RE = re.compile(
    r"^\s*Proof\b",
    re.IGNORECASE,
)

# Regex to detect section/chapter headings (e.g., "1.2 Set Functions")
SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+\.[\d.]*)\s+[A-Z]",
)

# Regex to detect proof end markers
PROOF_END_RE = re.compile(
    r"[□■∎]|\bQ\.?E\.?D\.?\b",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_theorem_start(block: dict) -> tuple[bool, str, str]:
    """
    Check if a block starts a new theorem-like unit.
    Returns (is_start, unit_type, title).
    """
    text = block.get("text", "").strip()
    block_type = block.get("type", "")

    # Exclude proof blocks — they are NOT theorem units
    if PROOF_START_RE.match(text):
        return False, "", ""

    # Check explicit theorem_heading type from OCR
    if block_type in ("theorem_heading", "theorem_body"):
        m = THEOREM_HEADING_RE.match(text)
        if m:
            keyword = m.group(1).lower()
            unit_type = KEYWORD_TO_TYPE.get(keyword, "other")
            # Extract title (first line or up to first period)
            first_line = text.split("\n")[0].strip()
            return True, unit_type, first_line
        # Even without regex match, if OCR tagged it as theorem_heading
        return True, "other", text.split("\n")[0].strip()

    # Check text blocks that start with theorem keywords
    m = THEOREM_HEADING_RE.match(text)
    if m:
        keyword = m.group(1).lower()
        unit_type = KEYWORD_TO_TYPE.get(keyword, "other")
        first_line = text.split("\n")[0].strip()
        return True, unit_type, first_line

    return False, "", ""


def is_proof_start(block: dict) -> bool:
    """Check if a block starts a proof."""
    text = block.get("text", "").strip()
    return bool(PROOF_START_RE.match(text))


def is_section_heading(block: dict) -> bool:
    """Check if a block is a section/chapter heading."""
    block_type = block.get("type", "")
    if block_type == "heading":
        return True
    text = block.get("text", "").strip()
    font_size = block.get("font_size", 10)
    # Large font section headings
    if font_size > 13 and SECTION_HEADING_RE.match(text):
        return True
    return False


def is_boundary(block: dict) -> bool:
    """Check if a block represents a boundary that should stop collection."""
    if is_section_heading(block):
        return True
    if is_proof_start(block):
        return True
    is_start, _, _ = is_theorem_start(block)
    return is_start


def extract_formulas_from_blocks(blocks: list[dict], block_ids: list[int],
                                  page_id: str) -> list[dict]:
    """Extract formula spans from the collected blocks."""
    formulas = []
    block_map = {b["block_id"]: b for b in blocks}
    formula_counter = 1

    for bid in block_ids:
        b = block_map.get(bid)
        if not b:
            continue
        if b.get("type") == "formula":
            formulas.append({
                "formula_id": f"f_{page_id}_{formula_counter:02d}",
                "text": b.get("text", "").strip(),
                "source_block": bid,
            })
            formula_counter += 1

    return formulas


def collect_statement_text(blocks: list[dict], block_ids: list[int]) -> str:
    """Assemble the statement text from collected blocks."""
    block_map = {b["block_id"]: b for b in blocks}
    parts = []
    for bid in block_ids:
        b = block_map.get(bid)
        if not b:
            continue
        text = b.get("text", "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_units_from_page(page_data: dict, chapter_number: int = 1,
                            current_section_title: str = "",
                            current_section_number: int = 0) -> tuple[list[dict], str, int]:
    """
    Extract theorem-like units from a single page's OCR data.

    Returns (units, updated_section_title, updated_section_number).
    """
    doc_id = page_data.get("doc_id", "")
    page_id = page_data.get("page_id", "")
    blocks = page_data.get("blocks", [])

    if not blocks:
        return [], current_section_title, current_section_number

    # Detect section headings on this page
    for b in blocks:
        if is_section_heading(b):
            text = b.get("text", "").strip()
            m = re.match(r"(\d+)\.(\d+)\s+(.+)", text)
            if m:
                current_section_number = int(m.group(2))
                current_section_title = m.group(3).strip()

    units = []
    i = 0

    while i < len(blocks):
        block = blocks[i]
        is_start, unit_type, title = is_theorem_start(block)

        if not is_start:
            i += 1
            continue

        # Found a theorem-like heading at position i
        # Collect blocks: start with the heading block
        collected_ids = [block["block_id"]]

        # Look ahead and collect subsequent blocks until boundary
        j = i + 1
        while j < len(blocks):
            next_block = blocks[j]

            # Stop at boundaries
            if is_boundary(next_block):
                break

            # Skip image blocks
            if next_block.get("type") == "image":
                j += 1
                continue

            collected_ids.append(next_block["block_id"])
            j += 1

        # Also collect proof blocks if they follow immediately
        proof_ids = []
        proof_j = j
        if proof_j < len(blocks) and is_proof_start(blocks[proof_j]):
            proof_ids.append(blocks[proof_j]["block_id"])
            proof_j += 1
            while proof_j < len(blocks):
                pb = blocks[proof_j]
                # Stop at next theorem heading or section heading
                ps, _, _ = is_theorem_start(pb)
                if ps or is_section_heading(pb):
                    break
                proof_ids.append(pb["block_id"])
                # Stop at proof end marker
                if PROOF_END_RE.search(pb.get("text", "")):
                    proof_j += 1
                    break
                proof_j += 1

        # Build the unit — use gold-compatible format
        statement_text = collect_statement_text(blocks, collected_ids)
        proof_text = collect_statement_text(blocks, proof_ids) if proof_ids else ""
        formula_spans = extract_formulas_from_blocks(blocks, collected_ids, page_id)

        # Extract canonical label: "Definition 1.1" etc.
        label_match = re.match(
            r"(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise)"
            r"\s+([\d.]+)",
            title, re.IGNORECASE
        )
        label = f"{label_match.group(1).capitalize()} {label_match.group(2).rstrip('.')}" if label_match else title

        # Map unit_type → env (gold format)
        TYPE_TO_ENV = {
            "definition": "def", "theorem": "thm", "lemma": "lem",
            "proposition": "prop", "corollary": "cor", "example": "ex",
            "remark": "rem", "exercise": "exr", "other": "rem",
        }

        # Parse number_components from label, e.g. "Definition 1.23" → [1, 23]
        num_part = label.split(None, 1)[-1] if " " in label else ""
        number_components = [int(x) for x in num_part.split(".") if x.isdigit()] if num_part else []

        unit = {
            "label": label,
            "env": TYPE_TO_ENV.get(unit_type, "rem"),
            "number_components": number_components,
            "extracted_labels": [],
            "context": {
                "chapter": "",
                "section": current_section_title,
                "subsection": "",
                "chapter_number": chapter_number,
                "section_number": current_section_number,
                "subsection_number": 0,
            },
            "content": statement_text,
            "dependencies": [],
            "proof": proof_text,
            "index": len(units) + 1,
            # Keep internal fields for traceability
            "_source_blocks": collected_ids,
            "_proof_blocks": proof_ids,
            "_page_id": page_id,
            "_doc_id": doc_id,
            "_formula_spans": formula_spans,
        }
        units.append(unit)

        # Move to after proof (or after theorem body)
        i = proof_j if proof_ids else j

    return units, current_section_title, current_section_number


def process_chapter_dir(chapter_dir: str) -> list[dict]:
    """Process all page JSON files in a chapter directory."""
    chapter_path = Path(chapter_dir)
    page_files = sorted(chapter_path.glob("p*.json"))

    # Infer chapter number from dir name
    chapter_match = re.search(r"Chapter(\d+)", str(chapter_path))
    chapter_number = int(chapter_match.group(1)) if chapter_match else 1

    all_units = []
    section_title = ""
    section_number = 0

    for page_file in page_files:
        with page_file.open("r", encoding="utf-8") as f:
            page_data = json.load(f)
        units, section_title, section_number = extract_units_from_page(
            page_data, chapter_number, section_title, section_number
        )
        all_units.extend(units)

    return all_units


def write_json(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as JSON array (matching gold format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based theorem unit extraction from OCR block JSON"
    )
    parser.add_argument(
        "--ocr_dir", required=True,
        help="Directory containing per-page OCR JSON files (or parent dir with --all)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all chapter subdirectories in ocr_dir"
    )
    args = parser.parse_args()

    all_units = []

    if args.all:
        # Process all chapter subdirectories
        ocr_root = Path(args.ocr_dir)
        chapter_dirs = sorted(
            d for d in ocr_root.iterdir()
            if d.is_dir() and d.name.startswith("Chapter")
        )
        if not chapter_dirs:
            # Maybe the dir itself contains page files
            chapter_dirs = [ocr_root]

        for chapter_dir in chapter_dirs:
            units = process_chapter_dir(str(chapter_dir))
            print(f"  {chapter_dir.name}: {len(units)} units extracted")
            all_units.extend(units)
    else:
        all_units = process_chapter_dir(args.ocr_dir)

    # Write output
    output_path = Path(args.output)
    write_json(output_path, all_units)

    print(f"\nRule-based extraction complete:")
    print(f"  Total units: {len(all_units)}")
    print(f"  Output: {output_path}")

    # Print type distribution
    type_counts = {}
    for u in all_units:
        t = u["env"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Type distribution: {json.dumps(type_counts, indent=2)}")


if __name__ == "__main__":
    main()
