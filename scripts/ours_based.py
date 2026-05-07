#!/usr/bin/env python3
"""
ours_based.py
=============
Structure-aware theorem unit extraction (Ours method).

Pipeline:
  Stage 1: Layout-aware normalization (noise filter, reading order, fragment merge)
  Stage 3: Rule-based boundary detection (same as rule_based.py)
  Stage 4: LLM-based symbol repair (fix U+FFFD replacement chars)
  Stage 5: Assembly (produce final output)

Key difference from baselines:
  - Rule:   raw OCR → keyword extraction → output
  - Hybrid: raw OCR → Rule extraction → LLM validation → symbol repair → output
  - Ours:   raw OCR → NORMALIZATION → extraction → symbol repair → output

Usage:
    python scripts/ours_based.py \\
        --ocr_dir "real_data/Klenke14/OCR/Chapter2" \\
        --output "predictions/ours/Chapter2_pred.json"
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared modules
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

# Stage 1: Normalization
from normalize_blocks import normalize_page_blocks

# Stage 3: Extraction (reuse rule_based functions)
from rule_based import (
    extract_units_from_page,
    SECTION_HEADING_RE,
)


# ---------------------------------------------------------------------------
# Stage 4: Symbol repair (adapted from hybrid_based.py)
# ---------------------------------------------------------------------------

SYMBOL_REPAIR_PROMPT = r"""You are a mathematical OCR error corrector. You will receive text fragments from a math textbook where the OCR scanner failed to recognize certain characters, shown as "�" (U+FFFD replacement character).

For each fragment, determine what the correct character or characters should be based on mathematical context.

## Input
A JSON array of fragments, each with:
- "id": fragment identifier
- "context": the text around the damaged character, with "�" marking the unknown character

## Output
A JSON array with the same ids and the corrected character(s):
[
  {"id": 0, "fix": "⋂"},
  {"id": 1, "fix": "("},
  {"id": 2, "fix": "∈"}
]

## Rules
1. Return ONLY a JSON array. No explanation.
2. Each "fix" should be exactly the character(s) that replace ONE "�".
3. Common OCR failures in math: ⋂, ⋃, ∈, ∉, ⊂, ⊃, ≤, ≥, ≠, →, ←, ↦, ∀, ∃, ∅, ∞, ∂, ∇, ∏, ∑, √, ∫, ≈, ≡, ∝, ⊕, ⊗, ⊆, ⊇, (, ), [, ]"""

SYMBOL_REPAIR_USER = """Fix the replacement characters in these fragments:

{fragments_json}"""


def extract_repair_contexts(content, window=30):
    """Extract context windows around each replacement character."""
    REPL = '\ufffd'
    fragments = []
    for i, ch in enumerate(content):
        if ch == REPL:
            start = max(0, i - window)
            end = min(len(content), i + window + 1)
            ctx = content[start:end].replace('\n', ' ')
            fragments.append({"id": len(fragments), "pos": i, "context": ctx})
    return fragments


def call_openai(messages, config, max_retries=3):
    """Call OpenAI-compatible API with retry logic.
    Returns (content_str, usage_dict).
    """
    import urllib.request

    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    body = {
        "model": config["model"],
        "messages": messages,
    }

    model = config["model"]
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        body["max_completion_tokens"] = 1536
    else:
        body["temperature"] = 0.2
        body["max_tokens"] = 1536

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


def parse_llm_response(response_text):
    """Parse LLM response as JSON array."""
    if not response_text:
        return None
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return None
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def repair_symbols(units, config):
    """Stage 4: Repair replacement characters in units using LLM."""
    units_to_repair = []
    for unit in units:
        content = unit.get("content", "")
        if '\ufffd' in content or '\ufffd' in unit.get("proof", ""):
            units_to_repair.append(unit)

    if not units_to_repair:
        print("  Symbol repair: no units need repair")
        return units, {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print(f"  Symbol repair: {len(units_to_repair)} units have replacement chars...", end=" ", flush=True)
    repaired_count = 0
    stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for unit in units_to_repair:
        # Repair content
        content = unit.get("content", "")
        if '\ufffd' in content:
            fragments = extract_repair_contexts(content)
            if fragments:
                frag_json = json.dumps(fragments, ensure_ascii=False, indent=1)
                user_msg = SYMBOL_REPAIR_USER.format(fragments_json=frag_json)
                messages = [
                    {"role": "system", "content": SYMBOL_REPAIR_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                response, usage = call_openai(messages, config)
                if usage:
                    stats["llm_calls"] += 1
                    stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    stats["completion_tokens"] += usage.get("completion_tokens", 0)
                    stats["total_tokens"] += usage.get("total_tokens", 0)
                fixes = parse_llm_response(response)
                if fixes:
                    fix_map = {}
                    for fix_entry in fixes:
                        fid = fix_entry.get("id")
                        fix_char = fix_entry.get("fix", "\ufffd")
                        if fid is not None and fid < len(fragments):
                            fix_map[fragments[fid]["pos"]] = fix_char
                    chars = list(content)
                    for pos in sorted(fix_map.keys(), reverse=True):
                        if pos < len(chars) and chars[pos] == '\ufffd':
                            chars[pos] = fix_map[pos]
                            repaired_count += 1
                    unit["content"] = "".join(chars)

        # Repair proof
        proof = unit.get("proof", "")
        if '\ufffd' in proof:
            pfrags = extract_repair_contexts(proof)
            if pfrags:
                pfrag_json = json.dumps(pfrags, ensure_ascii=False, indent=1)
                pmsg = SYMBOL_REPAIR_USER.format(fragments_json=pfrag_json)
                pmessages = [
                    {"role": "system", "content": SYMBOL_REPAIR_PROMPT},
                    {"role": "user", "content": pmsg},
                ]
                presp, pusage = call_openai(pmessages, config)
                if pusage:
                    stats["llm_calls"] += 1
                    stats["prompt_tokens"] += pusage.get("prompt_tokens", 0)
                    stats["completion_tokens"] += pusage.get("completion_tokens", 0)
                    stats["total_tokens"] += pusage.get("total_tokens", 0)
                pfixes = parse_llm_response(presp)
                if pfixes:
                    pfix_map = {}
                    for pf in pfixes:
                        pid = pf.get("id")
                        pfix = pf.get("fix", "\ufffd")
                        if pid is not None and pid < len(pfrags):
                            pfix_map[pfrags[pid]["pos"]] = pfix
                    pchars = list(proof)
                    for pos in sorted(pfix_map.keys(), reverse=True):
                        if pos < len(pchars) and pchars[pos] == '\ufffd':
                            pchars[pos] = pfix_map[pos]
                    unit["proof"] = "".join(pchars)

    print(f"fixed {repaired_count} chars")
    return units, stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_config():
    """Load LLM config from .env file or environment variables."""
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


def process_chapter(ocr_dir: str, output_path: str, delay: float = 0.3):
    """Full Ours pipeline for one chapter."""
    ocr_path = Path(ocr_dir)
    page_files = sorted(ocr_path.glob("p*.json"))

    # Infer chapter number
    chapter_match = re.search(r"Chapter(\d+)", str(ocr_path))
    chapter_number = int(chapter_match.group(1)) if chapter_match else 1

    config = load_config()
    print(f"Model: {config['model']}")
    print(f"OCR dir: {ocr_dir}")
    print(f"Method: Ours (Normalize → Extract → Symbol Repair)")
    print(f"Found {len(page_files)} pages")

    # ===== Stage 1: Normalization =====
    print("\n--- Stage 1: Normalization ---")
    pages = []
    total_orig = 0
    total_norm = 0
    for pf in page_files:
        with pf.open("r", encoding="utf-8") as f:
            page_data = json.load(f)
        orig_count = len(page_data.get("blocks", []))
        total_orig += orig_count
        normalize_page_blocks(page_data)
        norm_count = len(page_data.get("blocks", []))
        total_norm += norm_count
        pages.append(page_data)
    print(f"  Blocks: {total_orig} → {total_norm} (removed {total_orig - total_norm})")

    # ===== Stage 3: Boundary detection & extraction =====
    print("\n--- Stage 3: Extraction ---")
    all_units = []
    section_title = ""
    section_number = 0

    for page_data in pages:
        units, section_title, section_number = extract_units_from_page(
            page_data, chapter_number, section_title, section_number
        )
        all_units.extend(units)

    print(f"  Extracted: {len(all_units)} units")

    # ===== Stage 4: Symbol repair =====
    print("\n--- Stage 4: Symbol repair ---")
    import time as _time
    wall_start = _time.time()
    all_units, stats = repair_symbols(all_units, config)
    wall_end = _time.time()
    stats["wall_time_seconds"] = round(wall_end - wall_start, 1)

    # ===== Stage 5: Assembly =====
    print("\n--- Stage 5: Assembly ---")
    # Deduplicate by label
    seen = set()
    deduped = []
    for u in all_units:
        key = u["label"]
        if key not in seen:
            seen.add(key)
            deduped.append(u)
    print(f"  Deduplicated: {len(all_units)} → {len(deduped)} units")

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    # Save stats
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    type_counts = {}
    for u in deduped:
        t = u["env"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\nOurs extraction complete:")
    print(f"  Total units: {len(deduped)}")
    print(f"  Output: {out_path}")
    print(f"  Stats: {stats_path}")
    print(f"  LLM calls: {stats['llm_calls']}")
    print(f"  Prompt tokens: {stats['prompt_tokens']}")
    print(f"  Completion tokens: {stats['completion_tokens']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Wall time: {stats['wall_time_seconds']}s")
    print(f"  Type distribution: {json.dumps(type_counts, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Structure-aware theorem extraction (Ours method)"
    )
    parser.add_argument(
        "--ocr_dir", required=True,
        help="Directory containing per-page OCR JSON files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between LLM API calls (seconds)"
    )
    args = parser.parse_args()

    process_chapter(args.ocr_dir, args.output, args.delay)


if __name__ == "__main__":
    main()
