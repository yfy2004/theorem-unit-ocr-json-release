import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def boundary_key(unit: dict[str, Any]) -> tuple[str, int, int]:
    blocks = unit.get("source_blocks", [])
    if not blocks:
        return (unit["page_id"], -1, -1)
    return (unit["page_id"], min(blocks), max(blocks))


def block_set(unit: dict[str, Any]) -> set[int]:
    return set(unit.get("source_blocks", []))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z0-9_]+|[^\s]", text)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def formula_source_blocks(unit: dict[str, Any]) -> set[int]:
    out = set()
    for f in unit.get("formula_spans", []):
        sb = f.get("source_block")
        if isinstance(sb, int):
            out.add(sb)
    return out


def greedy_match_units(gold, pred):
    gold_by_boundary = {boundary_key(u): u for u in gold}
    pred_by_boundary = {boundary_key(u): u for u in pred}
    matched_pairs = []
    used_gold_ids = set()
    used_pred_ids = set()
    for k in set(gold_by_boundary) & set(pred_by_boundary):
        g = gold_by_boundary[k]
        p = pred_by_boundary[k]
        matched_pairs.append((g, p))
        used_gold_ids.add(id(g))
        used_pred_ids.add(id(p))
    pred_by_page = defaultdict(list)
    for p in pred:
        if id(p) not in used_pred_ids:
            pred_by_page[p["page_id"]].append(p)
    for g in gold:
        if id(g) in used_gold_ids:
            continue
        candidates = pred_by_page.get(g["page_id"], [])
        best_p = None
        best_score = 0.0
        g_blocks = block_set(g)
        for p in candidates:
            if id(p) in used_pred_ids:
                continue
            score = jaccard(g_blocks, block_set(p))
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is not None and best_score > 0.0:
            matched_pairs.append((g, best_p))
            used_gold_ids.add(id(g))
            used_pred_ids.add(id(best_p))
    return matched_pairs


def formula_attachment_accuracy(matched_pairs):
    if not matched_pairs:
        return 0.0
    correct = 0
    for g, p in matched_pairs:
        if formula_source_blocks(g) == formula_source_blocks(p):
            correct += 1
    return correct / len(matched_pairs)


def statement_completeness(matched_pairs):
    if not matched_pairs:
        return 0.0
    recalls = []
    for g, p in matched_pairs:
        g_tokens = tokenize(g.get("statement_text", ""))
        p_tokens = tokenize(p.get("statement_text", ""))
        if not g_tokens:
            recalls.append(1.0)
            continue
        g_counts = defaultdict(int)
        p_counts = defaultdict(int)
        for t in g_tokens:
            g_counts[t] += 1
        for t in p_tokens:
            p_counts[t] += 1
        overlap = sum(min(g_counts[t], p_counts[t]) for t in g_counts)
        recalls.append(overlap / len(g_tokens))
    return sum(recalls) / len(recalls)


def normalized_edit_distance(matched_pairs):
    if not matched_pairs:
        return 1.0
    dists = []
    for g, p in matched_pairs:
        gt = normalize_text(g.get("statement_text", ""))
        pt = normalize_text(p.get("statement_text", ""))
        max_len = max(len(gt), len(pt), 1)
        d = levenshtein(gt, pt) / max_len
        dists.append(d)
    return sum(dists) / len(dists)


def evaluate_integrity(gold, pred):
    matched_pairs = greedy_match_units(gold, pred)
    faa = formula_attachment_accuracy(matched_pairs)
    sc = statement_completeness(matched_pairs)
    ned = normalized_edit_distance(matched_pairs)
    return {
        "matched_unit_count": len(matched_pairs),
        "formula_attachment_accuracy": round(faa * 100, 2),
        "statement_completeness": round(sc * 100, 2),
        "normalized_edit_distance": round(ned, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    results = evaluate_integrity(gold, pred)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
