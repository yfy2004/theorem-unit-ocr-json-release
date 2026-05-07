#!/usr/bin/env python3
"""
normalize_blocks.py
===================
Layout-aware JSON normalization for OCR blocks.

Stage 1 of the Structure-aware constructor (Ours method).
Uses bbox coordinates to:
  1. Filter noise (headers, footers, page numbers, copyright)
  2. Fix reading order (sort by y then x)
  3. Merge fragment blocks (shattered formulas, split text)

All operations are pure rule-based — no LLM needed.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-task 1: Noise filtering
# ---------------------------------------------------------------------------

# Patterns for copyright / DOI / publisher lines
COPYRIGHT_RE = re.compile(
    r"(©|DOI\s|ISBN\s|Springer|Universitext|10\.\d{4}/)", re.IGNORECASE
)

PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Pattern for running headers like "2  1  Basic Measure Theory"
RUNNING_HEADER_RE = re.compile(r"^\s*\d+\s+\d+\s+[A-Z]")


def filter_noise(blocks: list[dict], page_height: float) -> list[dict]:
    """Remove headers, footers, page numbers, and copyright blocks."""
    cleaned = []
    for b in blocks:
        text = b.get("text", "").strip()
        bbox = b.get("bbox", [0, 0, 0, 0])
        font_size = b.get("font_size", 10.0)
        y_top = bbox[1]
        y_bottom = bbox[3]

        # Skip empty blocks
        if not text:
            continue

        # Footer: near bottom of page + small font
        if y_bottom > page_height * 0.91 and font_size < 9.5:
            # Check if it looks like copyright / publisher / page number
            if COPYRIGHT_RE.search(text) or PAGE_NUMBER_RE.match(text):
                continue

        # Header: near top of page + small font + matches running header pattern
        if y_top < page_height * 0.07 and font_size < 9.5:
            if RUNNING_HEADER_RE.match(text) or PAGE_NUMBER_RE.match(text):
                continue

        # Standalone page number anywhere (very short, just digits)
        if PAGE_NUMBER_RE.match(text) and font_size < 9.5:
            continue

        cleaned.append(b)

    return cleaned


# ---------------------------------------------------------------------------
# Sub-task 2: Reading order repair
# ---------------------------------------------------------------------------

def fix_reading_order(blocks: list[dict], row_threshold: float = 8.0) -> list[dict]:
    """
    Re-sort blocks by geometric position: top-to-bottom, left-to-right.

    Blocks whose y_top values are within `row_threshold` pixels are considered
    to be on the same row and sorted left-to-right within that row.
    """
    if not blocks:
        return blocks

    # Sort by y_top first
    sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])

    # Group into rows
    rows = []
    current_row = [sorted_blocks[0]]
    for b in sorted_blocks[1:]:
        prev_y = current_row[0]["bbox"][1]
        curr_y = b["bbox"][1]
        if abs(curr_y - prev_y) <= row_threshold:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
    rows.append(current_row)

    # Sort within each row by x_left
    result = []
    for row in rows:
        row.sort(key=lambda b: b["bbox"][0])
        result.extend(row)

    return result


# ---------------------------------------------------------------------------
# Sub-task 3: Fragment merging
# ---------------------------------------------------------------------------

def _x_overlap(bbox1, bbox2) -> float:
    """Compute horizontal overlap ratio between two bboxes."""
    x1_left, _, x1_right, _ = bbox1
    x2_left, _, x2_right, _ = bbox2
    overlap_left = max(x1_left, x2_left)
    overlap_right = min(x1_right, x2_right)
    if overlap_right <= overlap_left:
        return 0.0
    overlap = overlap_right - overlap_left
    min_width = min(x1_right - x1_left, x2_right - x2_left)
    if min_width <= 0:
        return 0.0
    return overlap / min_width


def _y_gap(bbox1, bbox2) -> float:
    """Vertical gap between bottom of bbox1 and top of bbox2."""
    return bbox2[1] - bbox1[3]


def _union_bbox(bbox1, bbox2) -> list:
    """Compute the union bounding box."""
    return [
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3]),
    ]


def _is_heading(block: dict) -> bool:
    """Check if block is a heading type."""
    return block.get("type") in ("heading",)


def _is_theorem_start(block: dict) -> bool:
    """Check if block starts a new theorem unit."""
    btype = block.get("type", "")
    if btype in ("theorem_heading", "theorem_body"):
        return True
    text = block.get("text", "").strip()
    return bool(re.match(
        r"^(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise|Proof)\b",
        text, re.IGNORECASE
    ))


def should_merge(b1: dict, b2: dict) -> bool:
    """Decide whether two adjacent blocks should be merged."""
    # Never merge headings
    if _is_heading(b1) or _is_heading(b2):
        return False

    # Never merge two theorem starts together
    if _is_theorem_start(b1) and _is_theorem_start(b2):
        return False

    # Don't merge INTO a new theorem start
    if _is_theorem_start(b2) and not _is_theorem_start(b1):
        return False

    bbox1 = b1.get("bbox", [0, 0, 100, 20])
    bbox2 = b2.get("bbox", [0, 0, 100, 20])
    text1 = b1.get("text", "").strip()
    text2 = b2.get("text", "").strip()

    # At least one must be short (fragment indicator)
    is_fragment = len(text1) < 15 or len(text2) < 15

    # Both are long complete texts → don't merge
    if len(text1) > 50 and len(text2) > 50:
        return False

    # Vertical gap must be small
    gap = _y_gap(bbox1, bbox2)
    if gap > 15:
        return False
    # Allow slight overlap (OCR sometimes overlaps bbox)
    if gap < -10:
        return False

    # X overlap or adjacency
    x_ovlp = _x_overlap(bbox1, bbox2)
    x_gap = bbox2[0] - bbox1[2]  # horizontal gap

    # Case 1: Vertical stacking with x overlap (formula fragments)
    if x_ovlp > 0.3 and is_fragment:
        return True

    # Case 2: Horizontal adjacency on same line (split text)
    y_diff = abs(bbox1[1] - bbox2[1])
    if y_diff < 8 and x_gap < 20 and x_gap > -5:
        return True

    # Case 3: Very short fragments that are spatially close
    if is_fragment and len(text1) < 5 and len(text2) < 5:
        if gap < 10 and abs(bbox1[0] - bbox2[0]) < 30:
            return True

    return False


def merge_two_blocks(b1: dict, b2: dict) -> dict:
    """Merge two blocks into one."""
    text1 = b1.get("text", "").strip()
    text2 = b2.get("text", "").strip()

    # Determine merged type: formula takes priority
    type1 = b1.get("type", "text")
    type2 = b2.get("type", "text")
    if "formula" in type1 or "formula" in type2:
        merged_type = "formula"
    elif "theorem" in type1:
        merged_type = type1
    elif "theorem" in type2:
        merged_type = type2
    else:
        merged_type = type1

    return {
        "block_id": b1["block_id"],
        "type": merged_type,
        "text": text1 + " " + text2,
        "bbox": _union_bbox(b1["bbox"], b2["bbox"]),
        "font_size": max(b1.get("font_size", 10), b2.get("font_size", 10)),
        "is_bold": b1.get("is_bold", False) or b2.get("is_bold", False),
    }


def merge_fragments(blocks: list[dict]) -> list[dict]:
    """Merge adjacent fragment blocks."""
    if not blocks:
        return blocks

    merged = []
    current = blocks[0]

    for i in range(1, len(blocks)):
        next_b = blocks[i]
        if should_merge(current, next_b):
            current = merge_two_blocks(current, next_b)
        else:
            merged.append(current)
            current = next_b

    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_page_blocks(page_data: dict) -> dict:
    """
    Full normalization pipeline for one page's OCR blocks.

    1. Filter noise (headers, footers, page numbers)
    2. Fix reading order (sort by y then x)
    3. Merge fragment blocks
    4. Re-number block_ids

    Returns the modified page_data (in-place modification).
    """
    blocks = page_data.get("blocks", [])
    page_height = page_data.get("page_height", 666.0)

    # Step 1: Filter noise
    blocks = filter_noise(blocks, page_height)

    # Step 2: Fix reading order
    blocks = fix_reading_order(blocks)

    # Step 3: Merge fragments
    # Run merge pass twice — first pass may create new merge opportunities
    blocks = merge_fragments(blocks)
    blocks = merge_fragments(blocks)

    # Step 4: Re-number block_ids
    for i, b in enumerate(blocks):
        b["block_id"] = i

    page_data["blocks"] = blocks
    page_data["num_blocks"] = len(blocks)
    return page_data


# ---------------------------------------------------------------------------
# CLI: standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python normalize_blocks.py <page.json> [--verbose]")
        sys.exit(1)

    page_path = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv

    with page_path.open("r", encoding="utf-8") as f:
        page_data = json.load(f)

    orig_count = len(page_data.get("blocks", []))

    page_data = normalize_page_blocks(page_data)

    new_count = len(page_data["blocks"])
    print(f"{page_path.name}: {orig_count} blocks → {new_count} blocks")

    if verbose:
        for b in page_data["blocks"]:
            text_preview = b["text"][:60].replace("\n", " ")
            print(f"  [{b['block_id']:2d}] {b['type']:18s} {text_preview}")
