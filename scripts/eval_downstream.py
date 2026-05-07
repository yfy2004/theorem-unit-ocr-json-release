import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

VALID_TYPES = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "other"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def block_set(unit: dict[str, Any]) -> set[int]:
    return set(unit.get("source_blocks", []))


def boundary_key(unit: dict[str, Any]) -> tuple[str, int, int]:
    blocks = unit.get("source_blocks", [])
    if not blocks:
        return (unit["page_id"], -1, -1)
    return (unit["page_id"], min(blocks), max(blocks))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def greedy_match_units(gold, pred):
    gold_by_boundary = {}
    pred_by_boundary = {}
    for i, u in enumerate(gold):
        gold_by_boundary[boundary_key(u)] = i
    for p in pred:
        pred_by_boundary[boundary_key(p)] = p
    matches = {i: None for i in range(len(gold))}
    used_pred_ids = set()
    for k, gi in gold_by_boundary.items():
        if k in pred_by_boundary:
            p = pred_by_boundary[k]
            matches[gi] = p
            used_pred_ids.add(id(p))
    pred_by_page = defaultdict(list)
    for p in pred:
        if id(p) not in used_pred_ids:
            pred_by_page[p["page_id"]].append(p)
    for gi, g in enumerate(gold):
        if matches[gi] is not None:
            continue
        best_p = None
        best_score = 0.0
        g_blocks = block_set(g)
        for p in pred_by_page.get(g["page_id"], []):
            if id(p) in used_pred_ids:
                continue
            score = jaccard(g_blocks, block_set(p))
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is not None and best_score > 0.0:
            matches[gi] = best_p
            used_pred_ids.add(id(best_p))
    return matches


def has_balanced_delimiters(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                return False
            stack.pop()
    return len(stack) == 0


def suspicious_ocr_noise(text: str) -> bool:
    if "�" in text:
        return True
    if re.search(r"\b[Il1]{5,}\b", text):
        return True
    if re.search(r"[^\x00-\x7F]{4,}", text) and "σ" not in text and "π" not in text:
        return True
    return False


def all_text(unit: dict[str, Any]) -> str:
    statement = unit.get("statement_text", "") or ""
    formulas = " ".join((f.get("text", "") or "") for f in unit.get("formula_spans", []))
    return f"{statement} {formulas}".strip()


def score_structured_parse_success(unit: dict[str, Any]) -> float:
    if not isinstance(unit, dict):
        return 0.0
    required_ok = (
        isinstance(unit.get("doc_id"), str) and len(unit.get("doc_id", "")) > 0 and
        isinstance(unit.get("page_id"), str) and len(unit.get("page_id", "")) > 0 and
        isinstance(unit.get("unit_type"), str) and unit.get("unit_type") in VALID_TYPES and
        isinstance(unit.get("statement_text"), str) and len(unit.get("statement_text", "").strip()) > 0 and
        isinstance(unit.get("source_blocks"), list) and len(unit.get("source_blocks")) > 0 and
        isinstance(unit.get("reading_order"), list) and len(unit.get("reading_order")) > 0
    )
    if not required_ok:
        return 0.0
    if not all(isinstance(x, int) for x in unit["source_blocks"]):
        return 0.0
    if not all(isinstance(x, int) for x in unit["reading_order"]):
        return 0.0
    return 1.0


def score_symbol_stability(unit: dict[str, Any]) -> float:
    text = all_text(unit)
    if not text.strip():
        return 0.0
    score = 1.0
    if not has_balanced_delimiters(text):
        score -= 0.35
    if suspicious_ocr_noise(text):
        score -= 0.35
    if len(unit.get("statement_text", "").strip()) < 10:
        score -= 0.15
    ro = unit.get("reading_order", [])
    if len(ro) != len(set(ro)):
        score -= 0.15
    return max(0.0, min(1.0, score))


def score_retrieval_compatibility(unit: dict[str, Any]) -> float:
    score = 1.0
    if not unit.get("doc_id"):
        score -= 0.2
    if not unit.get("page_id"):
        score -= 0.2
    if unit.get("unit_type") not in VALID_TYPES:
        score -= 0.2
    if not isinstance(unit.get("source_blocks"), list) or len(unit.get("source_blocks", [])) == 0:
        score -= 0.2
    if not isinstance(unit.get("reading_order"), list) or len(unit.get("reading_order", [])) == 0:
        score -= 0.1
    if not isinstance(unit.get("statement_text"), str) or len(unit.get("statement_text", "").strip()) == 0:
        score -= 0.1
    return max(0.0, min(1.0, score))


def evaluate_downstream(gold, pred):
    matches = greedy_match_units(gold, pred)
    parse_scores = []
    symbol_scores = []
    retrieval_scores = []
    readiness_scores = []
    for gi, g in enumerate(gold):
        p = matches[gi]
        if p is None:
            parse_scores.append(0.0)
            symbol_scores.append(0.0)
            retrieval_scores.append(0.0)
            readiness_scores.append(0.0)
            continue
        ps = score_structured_parse_success(p)
        ss = score_symbol_stability(p)
        rs = score_retrieval_compatibility(p)
        fr = (ps + ss + rs) / 3.0
        parse_scores.append(ps)
        symbol_scores.append(ss)
        retrieval_scores.append(rs)
        readiness_scores.append(fr)
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "matched_unit_count": sum(1 for x in matches.values() if x is not None),
        "structured_parse_success": round(avg(parse_scores) * 100, 2),
        "symbol_stability": round(avg(symbol_scores) * 100, 2),
        "retrieval_compatibility": round(avg(retrieval_scores) * 100, 2),
        "formalization_readiness": round(avg(readiness_scores) * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    results = evaluate_downstream(gold, pred)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
