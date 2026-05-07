#!/usr/bin/env python3
"""
build_gold_jsonl.py
===================
Convert Klenke14_revised/json (human-annotated theorem units) into the
evaluation JSONL format by matching each theorem to OCR page blocks via
text similarity.

The revised JSON has: label, env, content, proof, etc.
The eval JSONL needs: page_id, source_blocks, unit_type, statement_text, etc.

This script bridges the gap by:
1. Loading the revised theorem units
2. Loading the OCR block data for each page
3. For each theorem, finding which page and which blocks best match its content
4. Outputting a JSONL with proper page_id and source_blocks

Usage:
    python scripts/build_gold_jsonl.py \
        --revised_json Klenke14_revised/json/Chapter1.json \
        --ocr_dir real_data/Klenke14/OCR/Chapter1 \
        --output gold/Chapter1_gold.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# ENV type mapping (revised format → eval format)
# ---------------------------------------------------------------------------

ENV_TO_TYPE = {
    "def": "definition",
    "defn": "definition",
    "thm": "theorem",
    "lem": "lemma",
    "prop": "proposition",
    "cor": "corollary",
    "ex": "example",
    "exr": "other",
    "rem": "other",
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison: strip LaTeX commands, collapse whitespace."""
    # Remove LaTeX commands like \begin{...}, \end{...}, \label{...}
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\item\b", "", text)
    # Remove $ delimiters
    text = text.replace("$", "")
    # Replace LaTeX math commands with Unicode-like representations
    text = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)  # remove remaining commands
    text = re.sub(r"[{}]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title_from_label(label: str) -> str:
    """Extract title from label like 'Definition 1.1' or 'Theorem 1.3'."""
    return label.strip()


def similarity(a: str, b: str) -> float:
    """Compute text similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    # Use first 300 chars for efficiency
    return SequenceMatcher(None, a[:300].lower(), b[:300].lower()).ratio()


def find_best_page_and_blocks(theorem_text: str, title: str,
                               ocr_pages: list[dict]) -> tuple[str, list[int]]:
    """
    Find the page and blocks that best match the theorem content.
    Returns (page_id, source_block_ids).
    """
    normalized_thm = normalize_text(theorem_text)
    title_lower = title.lower()

    best_page_id = ""
    best_blocks = []
    best_score = 0.0

    for page_data in ocr_pages:
        page_id = page_data["page_id"]
        blocks = page_data["blocks"]

        # Strategy 1: Find the block that contains the theorem title
        title_block_idx = -1
        for i, block in enumerate(blocks):
            block_text = block.get("text", "").lower()
            # Check if block starts with the theorem label
            if title_lower and title_lower.split()[0] in block_text:
                # More precise: check if the full label matches
                label_parts = title_lower.split()
                if len(label_parts) >= 2:
                    keyword = label_parts[0]  # e.g., "definition"
                    number = label_parts[1]   # e.g., "1.1"
                    if keyword in block_text and number in block_text:
                        title_block_idx = i
                        break

        if title_block_idx < 0:
            continue

        # Collect blocks from title block forward until we hit another theorem
        collected_ids = [blocks[title_block_idx]["block_id"]]
        collected_text = blocks[title_block_idx].get("text", "")

        for j in range(title_block_idx + 1, len(blocks)):
            next_block = blocks[j]
            next_text = next_block.get("text", "").strip()

            # Stop at proof, next theorem heading, or section heading
            if re.match(r"^\s*(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise)\s+\d",
                        next_text, re.IGNORECASE):
                break
            if re.match(r"^\s*Proof\b", next_text, re.IGNORECASE):
                break
            if next_block.get("type") == "heading":
                break

            collected_ids.append(next_block["block_id"])
            collected_text += " " + next_text

        # Score this match
        score = similarity(normalized_thm, normalize_text(collected_text))

        if score > best_score:
            best_score = score
            best_page_id = page_id
            best_blocks = collected_ids

    return best_page_id, best_blocks


def load_ocr_pages(ocr_dir: str) -> list[dict]:
    """Load all per-page OCR JSON files from a directory."""
    ocr_path = Path(ocr_dir)
    pages = []
    for page_file in sorted(ocr_path.glob("p*.json")):
        with page_file.open("r", encoding="utf-8") as f:
            pages.append(json.load(f))
    return pages


def main():
    parser = argparse.ArgumentParser(
        description="Convert revised JSON to evaluation JSONL with block mappings"
    )
    parser.add_argument("--revised_json", required=True,
                        help="Path to Klenke14_revised/json/ChapterN.json")
    parser.add_argument("--ocr_dir", required=True,
                        help="Path to OCR block JSON directory for same chapter")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    args = parser.parse_args()

    # Load data
    with open(args.revised_json, "r", encoding="utf-8") as f:
        revised_units = json.load(f)

    ocr_pages = load_ocr_pages(args.ocr_dir)
    doc_id = Path(args.revised_json).stem  # e.g., "Chapter1"

    print(f"Revised units: {len(revised_units)}")
    print(f"OCR pages: {len(ocr_pages)}")

    # Convert each theorem unit
    gold_units = []
    matched = 0
    unmatched = 0

    for idx, unit in enumerate(revised_units):
        label = unit.get("label", "")
        env = unit.get("env", "")
        content = unit.get("content", "")
        proof = unit.get("proof", "")

        # Map env → unit_type
        unit_type = ENV_TO_TYPE.get(env, "other")

        # Find matching page and blocks
        page_id, source_blocks = find_best_page_and_blocks(
            content, label, ocr_pages
        )

        if not page_id or not source_blocks:
            unmatched += 1
            continue

        matched += 1

        # Build statement text from matched blocks
        block_map = {}
        for p in ocr_pages:
            if p["page_id"] == page_id:
                block_map = {b["block_id"]: b for b in p["blocks"]}
                break

        statement_text = " ".join(
            block_map[bid].get("text", "").strip()
            for bid in source_blocks
            if bid in block_map
        )

        # Extract formula spans
        formula_spans = []
        f_count = 1
        for bid in source_blocks:
            b = block_map.get(bid)
            if b and b.get("type") == "formula":
                formula_spans.append({
                    "formula_id": f"f_{page_id}_{f_count:02d}",
                    "text": b.get("text", "").strip(),
                    "source_block": bid,
                })
                f_count += 1

        gold_unit = {
            "doc_id": doc_id,
            "page_id": page_id,
            "unit_id": f"gold_{doc_id}_{idx:04d}",
            "unit_type": unit_type,
            "title": extract_title_from_label(label),
            "statement_text": statement_text,
            "formula_spans": formula_spans,
            "source_blocks": source_blocks,
            "reading_order": list(source_blocks),
        }
        gold_units.append(gold_unit)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for u in gold_units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    print(f"\nGold JSONL generated:")
    print(f"  Matched: {matched}/{len(revised_units)}")
    print(f"  Unmatched: {unmatched}/{len(revised_units)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
