#!/usr/bin/env python3
"""
pdf_to_ocr_json.py
==================
Extract text blocks from PDF pages using PyMuPDF, producing per-page OCR-like
JSON files suitable for the theorem unit benchmark pipeline.

Each page is saved as a separate JSON file with block-level text, bounding
boxes, and basic type classification (text / heading / formula).

Usage:
    python scripts/pdf_to_ocr_json.py <pdf_path> <output_dir>
    python scripts/pdf_to_ocr_json.py real_data/Klenke14/pdf/Chapter1.pdf real_data/Klenke14/OCR/Chapter1

    # Process all chapters:
    python scripts/pdf_to_ocr_json.py --all real_data/Klenke14/pdf real_data/Klenke14/OCR
"""

import argparse
import json
import os
import re
import sys

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Block type heuristics
# ---------------------------------------------------------------------------

THEOREM_KEYWORDS = re.compile(
    r"^\s*(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise|Proof)\b",
    re.IGNORECASE,
)

SECTION_KEYWORDS = re.compile(
    r"^\s*(\d+\.[\d.]*)\s+[A-Z]",  # e.g. "1.2 Set Functions"
)

def classify_block(text: str, font_size: float, is_bold: bool) -> str:
    """Heuristic classification of a text block."""
    stripped = text.strip()
    if not stripped:
        return "empty"

    # Section / chapter headings tend to have larger font
    if font_size > 13 or (SECTION_KEYWORDS.match(stripped) and font_size > 11):
        return "heading"

    # Theorem-like headings
    if THEOREM_KEYWORDS.match(stripped):
        if len(stripped) < 200:
            return "theorem_heading"
        return "theorem_body"

    # Lines that are mostly math symbols / short formulas
    math_chars = sum(1 for c in stripped if c in r"\∀∃∈∉⊂⊃∪∩∅∞∑∏∫≤≥≠≈→←↔⇒⇐⇔∧∨¬∥⊥αβγδεζηθικλμνξπρστυφχψω")
    if len(stripped) > 0 and math_chars / len(stripped) > 0.3:
        return "formula"

    # Displayed equations often appear as short centered blocks
    if len(stripped) < 120 and ("=" in stripped or "≤" in stripped or "≥" in stripped):
        alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
        if alpha_ratio < 0.4:
            return "formula"

    return "text"


# ---------------------------------------------------------------------------
# Font analysis helpers
# ---------------------------------------------------------------------------

def get_span_info(block_dict):
    """Extract average font size and bold status from a block's spans."""
    sizes = []
    bold_count = 0
    total_spans = 0
    if "lines" not in block_dict:
        return 10.0, False
    for line in block_dict["lines"]:
        for span in line["spans"]:
            sizes.append(span["size"])
            total_spans += 1
            if "bold" in span.get("font", "").lower() or "Bold" in span.get("font", ""):
                bold_count += 1
    avg_size = sum(sizes) / len(sizes) if sizes else 10.0
    is_bold = (bold_count > total_spans / 2) if total_spans > 0 else False
    return avg_size, is_bold


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_page_blocks(page, page_index: int, doc_id: str) -> dict:
    """Extract blocks from a single page."""
    page_id = f"p{page_index + 1:03d}"
    blocks_out = []
    block_id = 0

    # Get detailed block info (dict mode)
    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for b in page_dict["blocks"]:
        if b["type"] == 0:  # text block
            # Reconstruct text from spans
            lines_text = []
            for line in b.get("lines", []):
                line_text = "".join(span["text"] for span in line["spans"])
                lines_text.append(line_text)
            text = "\n".join(lines_text).strip()

            if not text:
                continue

            avg_size, is_bold = get_span_info(b)
            block_type = classify_block(text, avg_size, is_bold)

            if block_type == "empty":
                continue

            blocks_out.append({
                "block_id": block_id,
                "type": block_type,
                "text": text,
                "bbox": [round(c, 1) for c in b["bbox"]],
                "font_size": round(avg_size, 1),
                "is_bold": is_bold,
            })
            block_id += 1

        elif b["type"] == 1:  # image block
            blocks_out.append({
                "block_id": block_id,
                "type": "image",
                "text": "[image]",
                "bbox": [round(c, 1) for c in b["bbox"]],
                "font_size": 0,
                "is_bold": False,
            })
            block_id += 1

    return {
        "doc_id": doc_id,
        "page_id": page_id,
        "page_index": page_index,
        "page_width": round(page.rect.width, 1),
        "page_height": round(page.rect.height, 1),
        "num_blocks": len(blocks_out),
        "blocks": blocks_out,
    }


def process_pdf(pdf_path: str, output_dir: str):
    """Process a single PDF and output per-page JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"Processing: {pdf_path} ({total_pages} pages)")

    manifest = {
        "doc_id": doc_id,
        "source_pdf": os.path.basename(pdf_path),
        "total_pages": total_pages,
        "pages": [],
    }

    for i in range(total_pages):
        page = doc[i]
        page_data = extract_page_blocks(page, i, doc_id)

        page_file = os.path.join(output_dir, f"{page_data['page_id']}.json")
        with open(page_file, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)

        manifest["pages"].append({
            "page_id": page_data["page_id"],
            "page_index": i,
            "num_blocks": page_data["num_blocks"],
            "file": f"{page_data['page_id']}.json",
        })

        print(f"  Page {i + 1}/{total_pages}: {page_data['num_blocks']} blocks → {page_file}")

    # Write manifest
    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done! {total_pages} pages → {output_dir}/")
    print(f"Manifest: {manifest_file}")
    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Extract OCR-like JSON from PDF using PyMuPDF")
    parser.add_argument("pdf_path", help="Path to PDF file, or directory with --all")
    parser.add_argument("output_dir", help="Output directory for JSON files")
    parser.add_argument("--all", action="store_true", help="Process all PDFs in the directory")
    args = parser.parse_args()

    if args.all:
        # Process all PDFs in directory
        pdf_dir = args.pdf_path
        if not os.path.isdir(pdf_dir):
            print(f"Error: {pdf_dir} is not a directory")
            sys.exit(1)
        pdfs = sorted(f for f in os.listdir(pdf_dir) if f.endswith(".pdf"))
        print(f"Found {len(pdfs)} PDF files in {pdf_dir}")
        for pdf_name in pdfs:
            chapter_name = os.path.splitext(pdf_name)[0]
            pdf_path = os.path.join(pdf_dir, pdf_name)
            out_dir = os.path.join(args.output_dir, chapter_name)
            process_pdf(pdf_path, out_dir)
        print(f"\nAll done! Processed {len(pdfs)} PDFs.")
    else:
        process_pdf(args.pdf_path, args.output_dir)


if __name__ == "__main__":
    main()
