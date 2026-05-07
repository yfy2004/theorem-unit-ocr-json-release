#!/usr/bin/env python3
"""
hybrid_based.py
===============
Hybrid theorem unit extraction: Rule-based extraction + LLM content repair.

Pipeline:
  Phase 1: Rule-based extraction (same as rule_based.py)
  Phase 2: LLM content repair — LLM outputs ONLY the corrections (find→replace),
           not the full content.  Much more token-efficient than rewriting.

Key difference from Ours:
  - Ours:   Normalize OCR structure BEFORE extraction (pure rule) → then fix U+FFFD
  - Hybrid: Extract from raw OCR → then ask LLM to fix ALL OCR errors in content

Usage:
    python scripts/hybrid_based.py \
        --ocr_dir real_data/Klenke14/OCR/Chapter1 \
        --output predictions/hybrid/Chapter1_pred.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_based import extract_units_from_page as rule_extract


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
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
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }


# ---------------------------------------------------------------------------
# LLM Repair Prompt — output corrections only, NOT full content
# ---------------------------------------------------------------------------

REPAIR_SYSTEM_PROMPT = r"""You are a mathematical OCR error corrector. You will receive the content of theorem units extracted from OCR output of a math textbook. The text may contain:

- Unicode replacement characters (U+FFFD, "�") where OCR failed
- Corrupted or wrong mathematical symbols
- Broken LaTeX commands (e.g., missing braces, wrong escapes)
- OCR artifacts (garbage characters, wrong letters in formulas)

Your task: output ONLY the corrections needed. Do NOT reproduce the full content.

## Output Format

For each unit, output a JSON object with:
- "label": the unit's label (unchanged)
- "fixes": array of {"find": "broken text", "replace": "corrected text"}

If a unit needs no fixes, output {"label": "...", "fixes": []}.

Return a JSON array of all units. No markdown fences, no explanation.

## Example

Input unit: "Definition 1.1 A class of sets A is called ∩-closed if A � B ∈ A"

Output:
[{"label": "Definition 1.1", "fixes": [{"find": "A � B", "replace": "A ∩ B"}]}]

## Rules
1. Each "find" string must be an EXACT substring of the original content.
2. Keep "find" strings short but unique (include enough context to avoid ambiguity).
3. Fix ALL types of errors: replacement chars, wrong symbols, broken LaTeX, OCR typos in math.
4. Do NOT change correct mathematical content. Only fix actual errors.
5. Common OCR failures: ⋂→∩, ⋃→∪, ∈, ∉, ⊂, ⊃, ≤, ≥, ≠, →, ←, ∀, ∃, ∅, ∞, ∂, ∇, ∑, ∏, √, ∫
6. Return ONLY the JSON array."""

REPAIR_USER_TEMPLATE = """Fix OCR errors in these {count} theorem units. Output ONLY the corrections as find/replace pairs.

{units_json}"""


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

def call_openai(messages, config, max_tokens=2048, max_retries=3):
    """Returns (content_str, usage_dict)."""
    import urllib.request
    url = f"{config['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    body = {"model": config["model"], "messages": messages}
    model = config["model"]
    if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
        body["max_completion_tokens"] = max_tokens
    else:
        body["temperature"] = 0.2
        body["max_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"], result.get("usage", {})
        except Exception as e:
            print(f"  API attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, {}


def parse_llm_response(response_text):
    if not response_text:
        return None
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
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


# ---------------------------------------------------------------------------
# Repair Pipeline
# ---------------------------------------------------------------------------

def prepare_units_for_repair(units):
    """Compact representation: only label + content preview."""
    items = []
    for u in units:
        item = {"label": u.get("label", ""), "content": u.get("content", "")}
        proof = u.get("proof", "")
        if proof:
            item["proof"] = proof
        items.append(item)
    return items


def apply_fixes(unit, fixes):
    """Apply find/replace fixes to a unit's content and proof."""
    if not fixes:
        return 0
    count = 0
    content = unit.get("content", "")
    proof = unit.get("proof", "")
    for fix in fixes:
        find = fix.get("find", "")
        replace = fix.get("replace", "")
        if not find:
            continue
        if find in content:
            content = content.replace(find, replace, 1)
            count += 1
        if find in proof:
            proof = proof.replace(find, replace, 1)
            count += 1
    unit["content"] = content
    if proof:
        unit["proof"] = proof
    return count


def repair_batch(units, config):
    """Send a batch of units to LLM for content repair (diff-only output)."""
    items = prepare_units_for_repair(units)
    units_json = json.dumps(items, ensure_ascii=False, indent=1)

    user_msg = REPAIR_USER_TEMPLATE.format(
        count=len(items), units_json=units_json,
    )

    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response, usage = call_openai(messages, config, max_tokens=2048)
    repairs = parse_llm_response(response)

    if not repairs:
        return 0, usage

    # Build label → fixes map
    fix_map = {}
    for r in repairs:
        label = r.get("label", "")
        fixes = r.get("fixes", [])
        if label and fixes:
            fix_map[label] = fixes

    total_fixes = 0
    for unit in units:
        label = unit.get("label", "")
        fixes = fix_map.get(label, [])
        total_fixes += apply_fixes(unit, fixes)

    return total_fixes, usage


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def merge_and_reindex(all_units):
    seen = {}
    for unit in all_units:
        label = unit.get("label", "")
        if not label:
            continue
        if label in seen:
            if len(unit.get("content", "")) > len(seen[label].get("content", "")):
                seen[label] = unit
        else:
            seen[label] = unit
    merged = sorted(seen.values(), key=lambda u: u.get("number_components", [0]))
    for i, unit in enumerate(merged, 1):
        unit["index"] = i
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid: Rule extraction + LLM content repair (diff-only)"
    )
    parser.add_argument("--ocr_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Units per LLM repair call (can be larger since output is small)")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config()
    if not config["api_key"]:
        print("ERROR: OPENAI_API_KEY not set.")
        return

    print(f"Model: {config['model']}")
    print(f"OCR dir: {args.ocr_dir}")
    print(f"Method: Hybrid (Rule extraction + LLM content repair)")

    ocr_path = Path(args.ocr_dir)
    page_files = sorted(ocr_path.glob("p*.json"))
    print(f"Found {len(page_files)} pages")

    chapter_match = re.search(r"Chapter(\d+)", str(ocr_path))
    chapter_number = int(chapter_match.group(1)) if chapter_match else 1

    # ===== Phase 1: Rule-based extraction =====
    print("\n--- Phase 1: Rule-based extraction ---")
    all_units = []
    section_title = ""
    section_number = 0

    for page_file in page_files:
        with page_file.open("r", encoding="utf-8") as f:
            page_data = json.load(f)
        rule_units, section_title, section_number = rule_extract(
            page_data, chapter_number, section_title, section_number
        )
        all_units.extend(rule_units)

    merged = merge_and_reindex(all_units)
    print(f"  Extracted: {len(merged)} units")

    # ===== Phase 2: LLM content repair =====
    print("\n--- Phase 2: LLM content repair ---")
    total_fixes = 0
    batch_size = args.batch_size
    num_batches = (len(merged) + batch_size - 1) // batch_size

    # Stats tracking
    import time as _time
    stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    wall_start = _time.time()

    for i in range(0, len(merged), batch_size):
        batch = merged[i:i + batch_size]
        batch_num = i // batch_size + 1
        labels = [u.get("label", "?") for u in batch]
        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} units: {labels[0]}..{labels[-1]})...",
              end=" ", flush=True)

        fixes, usage = repair_batch(batch, config)
        total_fixes += fixes
        print(f"→ {fixes} fixes")

        # Accumulate stats
        if usage:
            stats["llm_calls"] += 1
            stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            stats["completion_tokens"] += usage.get("completion_tokens", 0)
            stats["total_tokens"] += usage.get("total_tokens", 0)

        if args.delay > 0:
            time.sleep(args.delay)

    wall_end = _time.time()
    stats["wall_time_seconds"] = round(wall_end - wall_start, 1)
    print(f"  Total fixes applied: {total_fixes}")

    # ===== Write output =====
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Save stats
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    env_counts = {}
    for u in merged:
        e = u.get("env", "?")
        env_counts[e] = env_counts.get(e, 0) + 1

    print(f"\nHybrid extraction complete:")
    print(f"  Rule candidates: {len(merged)}")
    print(f"  LLM repair batches: {num_batches}")
    print(f"  Total fixes: {total_fixes}")
    print(f"  Final units: {len(merged)}")
    print(f"  Output: {output_path}")
    print(f"  Stats: {stats_path}")
    print(f"  LLM calls: {stats['llm_calls']}")
    print(f"  Prompt tokens: {stats['prompt_tokens']}")
    print(f"  Completion tokens: {stats['completion_tokens']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Wall time: {stats['wall_time_seconds']}s")
    print(f"  Type distribution: {json.dumps(env_counts, indent=2)}")


if __name__ == "__main__":
    main()
