#!/usr/bin/env python3
"""
llm_based.py
============
LLM-only theorem unit extraction from OCR block JSON.
Sends each page's OCR blocks to GPT and asks it to extract theorem-like units
in the same JSON format as the gold standard.

Usage:
    python scripts/llm_based.py \
        --ocr_dir real_data/Klenke14/OCR/Chapter1 \
        --output predictions/llm/Chapter1_pred.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Part 1: Configuration
# ---------------------------------------------------------------------------

def load_config():
    """Load API config from .env file or environment variables."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }


# ---------------------------------------------------------------------------
# Part 2: Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""You are a mathematical document structure analyzer. Your task is to extract all theorem-like units from OCR-scanned textbook pages.

## Input
You will receive a JSON array of text blocks from one OCR page. Each block has:
- `block_id`: integer ID
- `type`: "text", "formula", "heading", "theorem_heading", "theorem_body", etc.
- `text`: the OCR-recognized text content

## Task
Identify and extract every theorem-like unit from the blocks. A "theorem-like unit" is any of the following 8 types:

| Keyword in text | `env` value | Description |
|-----------------|-------------|-------------|
| Definition X.Y  | `def`       | Mathematical definition |
| Theorem X.Y     | `thm`       | Theorem statement |
| Lemma X.Y       | `lem`       | Lemma statement |
| Corollary X.Y   | `cor`       | Corollary statement |
| Proposition X.Y | `prop`      | Proposition statement |
| Remark X.Y      | `rem`       | Remark or note |
| Example X.Y     | `ex`        | Worked example |
| Exercise X.Y.Z  | `exr`       | Exercise problem |

## Output Format
Return a JSON array. Each element must have exactly these fields:

{
  "label": "Definition 1.1",
  "env": "def",
  "number_components": [1, 1],
  "extracted_labels": [],
  "context": {
    "chapter": "",
    "section": "",
    "subsection": "",
    "chapter_number": 0,
    "section_number": 0,
    "subsection_number": 0
  },
  "content": "The full statement text assembled from relevant OCR blocks.",
  "dependencies": [],
  "proof": "",
  "index": 1
}

## Field Rules
- `label`: "{Type} {Number}", e.g. "Definition 1.1", "Exercise 1.1.3"
- `env`: one of def, thm, lem, cor, prop, rem, ex, exr
- `number_components`: number part split by dots as integers, e.g. "1.23" → [1, 23]
- `content`: combine all blocks belonging to this unit. Include the heading line. STOP at next theorem heading, proof start, or section heading. Do NOT include proof text.
- `proof`: if a "Proof" block immediately follows, include it here. Otherwise empty string.
- `index`: sequential numbering starting from 1 on this page.
- `context`: infer chapter_number from theorem numbers, section from visible headings.

## Critical Rules
1. Extract ALL 8 types of units. Do not skip Remarks, Examples, or Exercises.
2. Keep content and proof SEPARATE.
3. Preserve the original OCR text faithfully. Do not correct OCR errors.
4. Concatenate multi-block units with newlines.
5. Return ONLY a JSON array. No markdown fences, no explanation."""


USER_PROMPT_TEMPLATE = """Here is the OCR output for page {page_id} of document {doc_id}.
Current chapter: {chapter_number}, last known section: "{section_title}" (section {section_number}).

Extract all theorem-like units from the following blocks:

{blocks_json}"""


# ---------------------------------------------------------------------------
# Part 3: API calling
# ---------------------------------------------------------------------------

def call_openai(messages, config, max_retries=3):
    """Call OpenAI-compatible API with retry logic.
    Returns (content_str, usage_dict). usage_dict has prompt_tokens, completion_tokens, total_tokens.
    """
    import urllib.request
    import urllib.error

    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    body = {
        "model": config["model"],
        "messages": messages,
    }

    # Reasoning models (gpt-5.x) don't support temperature; use max_completion_tokens
    model = config["model"]
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        body["max_completion_tokens"] = 4096
    else:
        body["temperature"] = 0.2
        body["max_tokens"] = 4096

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            return content, usage
        except Exception as e:
            print(f"  API attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, {}


def simplify_blocks(blocks):
    """Keep only essential fields to reduce token usage."""
    simplified = []
    for b in blocks:
        simplified.append({
            "block_id": b.get("block_id"),
            "type": b.get("type", "text"),
            "text": b.get("text", ""),
        })
    return simplified


def has_theorem_keywords(blocks):
    """Quick check if page likely contains theorem-like units."""
    keywords = [
        "definition", "theorem", "lemma", "corollary", "proposition",
        "remark", "example", "exercise", "proof",
    ]
    for b in blocks:
        text_lower = b.get("text", "").lower()
        for kw in keywords:
            if kw in text_lower:
                return True
    return False


def parse_llm_response(response_text):
    """Parse LLM response as JSON array, handling common formatting issues."""
    if not response_text:
        return []

    # Remove markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```)
        lines = text.split("\n")
        lines = lines[1:]
        # Remove last ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def extract_units_from_page(page_data, config, chapter_number, section_title, section_number):
    """Send one page to LLM and get theorem units back."""
    doc_id = page_data.get("doc_id", "")
    page_id = page_data.get("page_id", "")
    blocks = page_data.get("blocks", [])

    if not blocks or not has_theorem_keywords(blocks):
        return [], section_title, section_number, {}

    # Update section context from this page's headings
    for b in blocks:
        if b.get("type") == "heading":
            text = b.get("text", "").strip()
            m = re.match(r"(\d+)\.(\d+)\s+(.+)", text)
            if m:
                section_number = int(m.group(2))
                section_title = m.group(3).strip()

    # Simplify blocks for API call
    simple_blocks = simplify_blocks(blocks)
    blocks_json = json.dumps(simple_blocks, ensure_ascii=False, indent=1)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        page_id=page_id,
        doc_id=doc_id,
        chapter_number=chapter_number,
        section_title=section_title,
        section_number=section_number,
        blocks_json=blocks_json,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    print(f"  Calling LLM for {page_id}...", end=" ", flush=True)
    response, usage = call_openai(messages, config)
    units = parse_llm_response(response)
    print(f"got {len(units)} units")

    return units, section_title, section_number, usage


# ---------------------------------------------------------------------------
# Part 4: Cross-page merging
# ---------------------------------------------------------------------------

def merge_and_reindex(all_units):
    """Merge units from all pages, deduplicate by label, reindex."""
    seen = {}
    for unit in all_units:
        label = unit.get("label", "")
        if label in seen:
            # Keep the longer content version
            if len(unit.get("content", "")) > len(seen[label].get("content", "")):
                seen[label] = unit
        else:
            seen[label] = unit

    # Sort by number_components for natural ordering
    def sort_key(u):
        comps = u.get("number_components", [0])
        return comps

    merged = sorted(seen.values(), key=sort_key)

    # Reindex
    for i, unit in enumerate(merged, 1):
        unit["index"] = i

    return merged


# ---------------------------------------------------------------------------
# Part 5: Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-only theorem unit extraction from OCR block JSON"
    )
    parser.add_argument("--ocr_dir", required=True, help="Directory with page JSON files")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    config = load_config()
    if not config["api_key"]:
        print("ERROR: OPENAI_API_KEY not set. Check .env file.")
        return

    print(f"Model: {config['model']}")
    print(f"OCR dir: {args.ocr_dir}")

    # Read all page files
    ocr_path = Path(args.ocr_dir)
    page_files = sorted(ocr_path.glob("p*.json"))
    print(f"Found {len(page_files)} pages")

    # Infer chapter number from directory name
    chapter_match = re.search(r"Chapter(\d+)", str(ocr_path))
    chapter_number = int(chapter_match.group(1)) if chapter_match else 1

    # Process each page
    all_units = []
    section_title = ""
    section_number = 0

    # Stats tracking
    import time as _time
    stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    wall_start = _time.time()

    for page_file in page_files:
        with page_file.open("r", encoding="utf-8") as f:
            page_data = json.load(f)

        units, section_title, section_number, usage = extract_units_from_page(
            page_data, config, chapter_number, section_title, section_number
        )
        all_units.extend(units)

        # Accumulate stats
        if usage:
            stats["llm_calls"] += 1
            stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stats["completion_tokens"] += usage.get("completion_tokens", 0)
            stats["total_tokens"] += usage.get("total_tokens", 0)

        if args.delay > 0 and units:
            time.sleep(args.delay)

    wall_end = _time.time()
    stats["wall_time_seconds"] = round(wall_end - wall_start, 1)

    # Merge and write output
    merged = merge_and_reindex(all_units)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Save stats
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nLLM-only extraction complete:")
    print(f"  Total units: {len(merged)}")
    print(f"  Output: {output_path}")
    print(f"  Stats: {stats_path}")
    print(f"  LLM calls: {stats['llm_calls']}")
    print(f"  Prompt tokens: {stats['prompt_tokens']}")
    print(f"  Completion tokens: {stats['completion_tokens']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Wall time: {stats['wall_time_seconds']}s")

    # Type distribution
    env_counts = {}
    for u in merged:
        e = u.get("env", "?")
        env_counts[e] = env_counts.get(e, 0) + 1
    print(f"  Type distribution: {json.dumps(env_counts, indent=2)}")


if __name__ == "__main__":
    main()
