#!/usr/bin/env python3
"""
eval_against_gold.py
====================
Evaluate predicted theorem units against the human-annotated gold standard
(Klenke14_revised/json). Matching by theorem label, NOT by source_blocks.

Outputs all metrics from paper Table 2 and Table 4.

Usage:
    python scripts/eval_against_gold.py \
        --gold Klenke14_revised/json/Chapter1.json \
        --pred predictions/rule/Chapter1_pred.jsonl \
        --output results/rule/Chapter1_eval.json \
        --system_name "Rule-based"
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

# ===================================================================
# Part 1: Label normalization & matching
# ===================================================================

ENV_TO_TYPE = {
    "def": "definition", "defn": "definition",
    "thm": "theorem", "lem": "lemma", "prop": "proposition",
    "cor": "corollary", "ex": "example",
    "exr": "other", "rem": "other",
}

LABEL_RE = re.compile(
    r"(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise)"
    r"\s+([\d.]+)",
    re.IGNORECASE,
)


def canonical_label(text):
    """Extract 'Definition 1.1' style label from any text."""
    m = LABEL_RE.search(text)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2).rstrip('.')}"
    return None


def match_by_label(gold_units, pred_units):
    """Match gold and pred by canonical theorem label.
    Returns (matched_pairs, gold_only_labels, pred_only_labels)."""
    gold_map = {}
    for g in gold_units:
        label = canonical_label(g.get("label", ""))
        if label:
            gold_map[label] = g

    pred_map = {}
    for p in pred_units:
        label = canonical_label(p.get("label", "") or p.get("title", ""))
        if label:
            if label not in pred_map or len(p.get("content", "")) > len(pred_map[label].get("content", "")):
                pred_map[label] = p

    matched = []
    for label in gold_map:
        if label in pred_map:
            matched.append((gold_map[label], pred_map[label]))

    gold_only = set(gold_map.keys()) - set(pred_map.keys())
    pred_only = set(pred_map.keys()) - set(gold_map.keys())
    return matched, gold_only, pred_only, len(gold_map), len(pred_map)


# ===================================================================
# Part 2: Structural metrics (B-P, B-R, B-F1, Type)
# ===================================================================

def calc_structure(matched, gold_only, pred_only, gold_count, pred_count):
    matched_count = len(matched)
    bp = matched_count / pred_count if pred_count else 0.0
    br = matched_count / gold_count if gold_count else 0.0
    bf1 = (2 * bp * br / (bp + br)) if (bp + br) else 0.0

    type_correct = 0
    for g, p in matched:
        gold_type = ENV_TO_TYPE.get(g.get("env", ""), "other")
        pred_type = ENV_TO_TYPE.get(p.get("env", ""), "other")
        if gold_type == pred_type:
            type_correct += 1
    type_acc = type_correct / len(matched) if matched else 0.0

    return {
        "B-P": round(bp * 100, 1),
        "B-R": round(br * 100, 1),
        "B-F1": round(bf1 * 100, 1),
        "Type": round(type_acc * 100, 1),
    }


# ===================================================================
# Part 3: Text normalization helpers
# ===================================================================

def strip_latex(text):
    """Remove LaTeX markup for fair comparison with OCR text."""
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(textbf|emph|textit|mathrm|mathcal|mathbb)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\item\b", "", text)
    text = text.replace("$", "")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_ocr(text):
    """Normalize OCR text for comparison."""
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ===================================================================
# Part 4: F-Attach (Formula Attachment Accuracy)
# ===================================================================

def calc_formula_attach(matched):
    """For each matched pair, check if pred's formulas belong to the gold content."""
    if not matched:
        return 0.0
    scores = []
    for g, p in matched:
        pred_formulas = p.get("_formula_spans", []) or p.get("formula_spans", [])
        if not pred_formulas:
            # No formulas claimed → correct if gold also has no display math
            gold_content = g.get("content", "")
            # Check if gold has display math ($$, \[, \begin{equation}, etc.)
            has_display = bool(re.search(r"\$\$|\\begin\{(equation|align|gather)", gold_content))
            scores.append(1.0 if not has_display else 0.0)
        else:
            gold_norm = strip_latex(g.get("content", ""))
            match_count = 0
            for fs in pred_formulas:
                f_text = normalize_ocr(fs.get("text", ""))
                # Check if formula tokens appear in gold content
                f_tokens = set(re.findall(r"[a-zA-Z0-9]+", f_text))
                g_tokens = set(re.findall(r"[a-zA-Z0-9]+", gold_norm))
                if not f_tokens:
                    match_count += 1
                elif len(f_tokens & g_tokens) / len(f_tokens) > 0.3:
                    match_count += 1
            scores.append(match_count / len(pred_formulas))
    return sum(scores) / len(scores)


# ===================================================================
# Part 5: Compl. (Statement Completeness) + Edit ↓ (Edit Distance)
# ===================================================================

def calc_completeness(matched):
    """Token recall: what fraction of gold tokens appear in pred."""
    if not matched:
        return 0.0
    recalls = []
    for g, p in matched:
        g_tokens = re.findall(r"[a-zA-Z0-9]+", strip_latex(g.get("content", "")))
        p_tokens = re.findall(r"[a-zA-Z0-9]+", normalize_ocr(p.get("content", "")))
        if not g_tokens:
            recalls.append(1.0)
            continue
        g_set = set(g_tokens)
        p_set = set(p_tokens)
        recalls.append(len(g_set & p_set) / len(g_set))
    return sum(recalls) / len(recalls)


def levenshtein(a, b):
    """Compute Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def calc_edit_distance(matched):
    """Normalized edit distance (lower is better)."""
    if not matched:
        return 1.0
    dists = []
    for g, p in matched:
        gt = strip_latex(g.get("content", ""))[:500]
        pt = normalize_ocr(p.get("content", ""))[:500]
        max_len = max(len(gt), len(pt), 1)
        dists.append(levenshtein(gt, pt) / max_len)
    return sum(dists) / len(dists)


# ===================================================================
# Part 6: Downstream Readiness (Parse, Symbol, Retrieval, Ready)
# ===================================================================

VALID_TYPES = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "other"}


def score_parse(unit):
    """Check if unit has all required fields with valid values."""
    ok = (
        isinstance(unit.get("label"), str) and len(unit.get("label", "")) > 0
        and isinstance(unit.get("env"), str) and len(unit.get("env", "")) > 0
        and isinstance(unit.get("content"), str) and len(unit.get("content", "").strip()) > 0
    )
    return 1.0 if ok else 0.0


def score_symbol(unit):
    """Check symbol stability: no OCR noise, balanced delimiters, sufficient text."""
    text = unit.get("content", "")
    for fs in (unit.get("_formula_spans", []) or unit.get("formula_spans", [])):
        text += " " + fs.get("text", "")
    score = 1.0
    # Check for replacement character
    if "\ufffd" in text:
        score -= 0.35
    # Check balanced delimiters
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                score -= 0.35
                break
            stack.pop()
    if stack:
        score -= 0.35
    # Check text length
    if len(text.strip()) < 10:
        score -= 0.15
    return max(0.0, min(1.0, score))


def score_retrieval(unit):
    """Check if unit has sufficient metadata for retrieval."""
    score = 1.0
    if not unit.get("label"):
        score -= 0.2
    if not unit.get("env"):
        score -= 0.2
    env = unit.get("env", "")
    valid_envs = {"def", "thm", "lem", "prop", "cor", "ex", "rem"}
    if env not in valid_envs:
        score -= 0.2
    if not isinstance(unit.get("content"), str) or not unit.get("content", "").strip():
        score -= 0.2
    if not unit.get("_doc_id") and not unit.get("doc_id"):
        score -= 0.1
    if not unit.get("_page_id") and not unit.get("page_id"):
        score -= 0.1
    return max(0.0, min(1.0, score))


def calc_downstream(matched_pred_units):
    """Calculate downstream readiness for matched prediction units."""
    if not matched_pred_units:
        return {"Parse": 0.0, "Symbol": 0.0, "Retrieval": 0.0, "Ready": 0.0}
    parse_scores = []
    symbol_scores = []
    retrieval_scores = []
    for p in matched_pred_units:
        ps = score_parse(p)
        ss = score_symbol(p)
        rs = score_retrieval(p)
        parse_scores.append(ps)
        symbol_scores.append(ss)
        retrieval_scores.append(rs)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    ready = avg([(p + s + r) / 3.0 for p, s, r in zip(parse_scores, symbol_scores, retrieval_scores)])
    return {
        "Parse": round(avg(parse_scores) * 100, 1),
        "Symbol": round(avg(symbol_scores) * 100, 1),
        "Retrieval": round(avg(retrieval_scores) * 100, 1),
        "Ready": round(ready * 100, 1),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system_name", default="")
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_units = json.load(f)

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_units = json.load(f)

    matched, gold_only, pred_only, gold_count, pred_count = match_by_label(gold_units, pred_units)

    structure = calc_structure(matched, gold_only, pred_only, gold_count, pred_count)
    f_attach = calc_formula_attach(matched)
    compl = calc_completeness(matched)
    edit = calc_edit_distance(matched)

    structure["F-Attach"] = round(f_attach * 100, 1)
    structure["Compl"] = round(compl * 100, 1)
    structure["Edit"] = round(edit, 3)

    # Downstream readiness (only on matched pred units)
    matched_preds = [p for _, p in matched]
    downstream = calc_downstream(matched_preds)
    structure["Ready"] = downstream["Ready"]

    results = {
        "system_name": args.system_name,
        "table2": structure,
        "table4": downstream,
        "detail": {
            "gold_count": gold_count,
            "pred_count": pred_count,
            "matched_count": len(matched),
            "missed": sorted(list(gold_only))[:20],
            "false_positives": sorted(list(pred_only))[:20],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
