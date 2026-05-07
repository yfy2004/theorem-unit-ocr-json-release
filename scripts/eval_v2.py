#!/usr/bin/env python3
"""
eval_v2.py — 评测脚本 v2
========================
分步骤执行，--step 控制运行哪一步：
  match      只看匹配结果
  structure  B-P / B-R / B-F1 / Type
  content    Compl / Edit
  formula    F-Attach
  downstream Parse / Symbol / Retrieval / Ready
  all        全部指标
"""

import argparse
import json
import re
from pathlib import Path

# ===================================================================
# 子任务 1：匹配引擎
# ===================================================================

LABEL_RE = re.compile(
    r"(Definition|Theorem|Lemma|Proposition|Corollary|Example|Remark|Exercise)"
    r"\s+([\d.]+)",
    re.IGNORECASE,
)


def canonical_label(text: str):
    """提取标准化 label，如 'Definition 1.1'。"""
    if not text:
        return None
    m = LABEL_RE.search(text)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2).rstrip('.')}"
    return None


def tokenize(text: str) -> set:
    """把文本拆成小写 word token 集合。"""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def token_similarity(text_a: str, text_b: str) -> float:
    """两段文本的 Jaccard 相似度。"""
    ta = tokenize(text_a)
    tb = tokenize(text_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_units(gold_units: list, pred_units: list):
    """
    两级匹配：
      第1级 — label 精确匹配
      第2级 — content token 相似度兜底（阈值 0.5）

    返回 (matched_pairs, gold_only, pred_only)
    """
    # ---- 第 1 级：Label 精确匹配 ----
    gold_by_label = {}
    for g in gold_units:
        lbl = canonical_label(g.get("label", ""))
        if lbl:
            gold_by_label[lbl] = g

    pred_by_label = {}
    for p in pred_units:
        lbl = canonical_label(p.get("label", ""))
        if lbl:
            # 同 label 多个 pred 时，保留 content 最长的
            if lbl not in pred_by_label or len(p.get("content", "")) > len(pred_by_label[lbl].get("content", "")):
                pred_by_label[lbl] = p

    matched = []
    matched_gold_labels = set()
    matched_pred_labels = set()

    for lbl, g in gold_by_label.items():
        if lbl in pred_by_label:
            matched.append((g, pred_by_label[lbl]))
            matched_gold_labels.add(lbl)
            matched_pred_labels.add(lbl)

    # ---- 第 2 级：Content 相似度兜底 ----
    remaining_gold = [g for g in gold_units if canonical_label(g.get("label", "")) not in matched_gold_labels]
    remaining_pred = [p for p in pred_units if canonical_label(p.get("label", "")) not in matched_pred_labels]

    used_pred = set()
    for g in remaining_gold:
        best_score = 0.0
        best_idx = -1
        for idx, p in enumerate(remaining_pred):
            if idx in used_pred:
                continue
            sim = token_similarity(g.get("content", ""), p.get("content", ""))
            if sim > best_score:
                best_score = sim
                best_idx = idx
        if best_score > 0.5 and best_idx >= 0:
            matched.append((g, remaining_pred[best_idx]))
            used_pred.add(best_idx)

    # 计算未匹配的
    matched_gold_set = {id(g) for g, _ in matched}
    matched_pred_set = {id(p) for _, p in matched}
    gold_only = [g for g in gold_units if id(g) not in matched_gold_set]
    pred_only = [p for p in pred_units if id(p) not in matched_pred_set]

    return matched, gold_only, pred_only


# ===================================================================
# 子任务 2：结构指标 (B-P, B-R, B-F1, Type)
# ===================================================================

def calc_boundary(matched, gold_count, pred_count):
    """B-P / B-R / B-F1."""
    n = len(matched)
    bp = n / pred_count if pred_count else 0.0
    br = n / gold_count if gold_count else 0.0
    bf1 = (2 * bp * br / (bp + br)) if (bp + br) else 0.0
    return {
        "B-P": round(bp * 100, 1),
        "B-R": round(br * 100, 1),
        "B-F1": round(bf1 * 100, 1),
    }


def calc_type(matched):
    """Type accuracy: env 精确匹配，不做任何归并。"""
    if not matched:
        return 0.0
    correct = 0
    for g, p in matched:
        if g.get("env", "") == p.get("env", ""):
            correct += 1
    return round(correct / len(matched) * 100, 1)


# ===================================================================
# 子任务 3：内容指标 (Compl, Edit)
# ===================================================================

def strip_latex(text: str) -> str:
    """Strip LaTeX commands → plain text for fair comparison."""
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(textbf|emph|textit|mathrm|mathcal|mathbb)\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\item\b", "", text)
    text = text.replace("$", "")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_ocr(text: str) -> str:
    """Normalize OCR text for comparison."""
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def calc_completeness(matched) -> float:
    """Token recall: gold tokens preserved in pred."""
    if not matched:
        return 0.0
    recalls = []
    for g, p in matched:
        g_tokens = set(re.findall(r"[a-zA-Z0-9]+", strip_latex(g.get("content", ""))))
        p_tokens = set(re.findall(r"[a-zA-Z0-9]+", normalize_ocr(p.get("content", ""))))
        if not g_tokens:
            recalls.append(1.0)
        else:
            recalls.append(len(g_tokens & p_tokens) / len(g_tokens))
    return round(sum(recalls) / len(recalls) * 100, 1)


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance."""
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


def calc_edit(matched) -> float:
    """Normalized edit distance (lower is better)."""
    if not matched:
        return 1.0
    dists = []
    for g, p in matched:
        gt = strip_latex(g.get("content", ""))[:500]
        pt = normalize_ocr(p.get("content", ""))[:500]
        max_len = max(len(gt), len(pt), 1)
        dists.append(levenshtein(gt, pt) / max_len)
    return round(sum(dists) / len(dists), 3)


# ===================================================================
# 子任务 4：公式指标 (F-Attach)
# ===================================================================

FORMULA_RE = re.compile(
    r"\$\$.*?\$\$"           # display math $$...$$
    r"|\$[^$]+\$"            # inline math $...$
    r"|\\\\\\(.*?\\\\\\)"    # \(...\)
    r"|\\\\\\[.*?\\\\\\]"    # \[...\]
    r"|\\\\begin\{(equation|align|gather|eqnarray)\*?\}.*?\\\\end\{\1\*?\}",
    re.DOTALL,
)


def extract_formula_tokens(latex_text: str) -> list:
    """Extract formula token sets from gold LaTeX content."""
    formulas = FORMULA_RE.findall(latex_text)
    if not formulas:
        # Fallback: look for any math-like content
        formulas = re.findall(r"\$([^$]+)\$", latex_text)
    results = []
    for f in formulas:
        tokens = set(re.findall(r"[a-zA-Z0-9]+", str(f).lower()))
        if tokens:
            results.append(tokens)
    return results


def calc_formula_attach(matched) -> float:
    """Check if pred contains the formula tokens from gold."""
    if not matched:
        return 0.0
    scores = []
    for g, p in matched:
        gold_formulas = extract_formula_tokens(g.get("content", ""))
        if not gold_formulas:
            scores.append(1.0)  # No formulas to check
            continue
        pred_tokens = set(re.findall(r"[a-zA-Z0-9]+", normalize_ocr(p.get("content", ""))))
        hit = 0
        for f_tokens in gold_formulas:
            if not f_tokens:
                hit += 1
            elif len(f_tokens & pred_tokens) / len(f_tokens) > 0.5:
                hit += 1
        scores.append(hit / len(gold_formulas))
    return round(sum(scores) / len(scores) * 100, 1)


# ===================================================================
# 子任务 5：下游指标 (Parse, Symbol, Retrieval, Ready)
# ===================================================================

VALID_ENVS = {"def", "thm", "lem", "cor", "prop", "rem", "ex", "exr"}


def score_parse(pred: dict) -> float:
    """Check structural parsability: fields present + content quality."""
    score = 1.0

    # 1. Field completeness (original)
    if not (isinstance(pred.get("label"), str) and len(pred.get("label", "")) > 0):
        score -= 0.15
    if not (isinstance(pred.get("env"), str) and pred.get("env", "") in VALID_ENVS):
        score -= 0.15
    if not (isinstance(pred.get("content"), str) and len(pred.get("content", "").strip()) > 0):
        score -= 0.3
    if not (isinstance(pred.get("number_components"), list) and len(pred.get("number_components", [])) > 0):
        score -= 0.1

    # 2. Content parsability: heavy corruption makes parsing fail
    content = pred.get("content", "")
    fffd_count = content.count("\ufffd")
    if fffd_count > 3:
        score -= 0.2
    elif fffd_count > 0:
        score -= 0.1

    # 3. Label completeness: truncated labels like "Theorem 5." are unparseable
    label = pred.get("label", "")
    if re.match(r"^(Theorem|Definition|Lemma|Corollary|Proposition)\s+\d+\.\s*$", label):
        score -= 0.15

    return max(0.0, min(1.0, score))


def score_symbol(pred: dict) -> float:
    """Check symbol stability: proportional noise scoring + delimiter balance + page noise."""
    text = pred.get("content", "") + " " + pred.get("proof", "")
    score = 1.0

    # 1. U+FFFD proportional scoring (not just has/doesn't-have)
    fffd_count = text.count("\ufffd")
    text_len = max(len(text), 1)
    fffd_ratio = fffd_count / text_len
    if fffd_ratio > 0.05:
        score -= 0.4
    elif fffd_ratio > 0.01:
        score -= 0.25
    elif fffd_count > 0:
        score -= 0.1

    # 2. Balanced delimiters
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    broken = False
    for ch in text:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack[-1] != ch:
                broken = True
                break
            stack.pop()
    if broken or stack:
        score -= 0.25

    # 3. Page noise leaked into content
    if re.search(r"(Springer|DOI\s*10\.\d|Universitext|©)", text):
        score -= 0.2

    # 4. Very short content
    if len(text.strip()) < 10:
        score -= 0.15

    return max(0.0, min(1.0, score))


def score_retrieval(pred: dict) -> float:
    """Check retrieval usability: metadata + content sufficiency + noise level."""
    score = 1.0

    # 1. Metadata presence
    if not pred.get("label"):
        score -= 0.15
    if pred.get("env", "") not in VALID_ENVS:
        score -= 0.15

    # 2. Content sufficiency: too short to be retrievable
    content = pred.get("content", "").strip()
    if len(content) < 20:
        score -= 0.25
    elif len(content) < 50:
        score -= 0.1

    # 3. Noise ratio: high noise makes retrieval unreliable
    if content:
        noise_ratio = content.count("\ufffd") / max(len(content), 1)
        if noise_ratio > 0.03:
            score -= 0.2
        elif noise_ratio > 0.01:
            score -= 0.1

    # 4. Number components present
    if not isinstance(pred.get("number_components"), list) or not pred.get("number_components"):
        score -= 0.1

    # 5. Section context present
    ctx = pred.get("context")
    if not isinstance(ctx, dict) or (not ctx.get("section") and ctx.get("section_number", 0) == 0):
        score -= 0.15

    return max(0.0, min(1.0, score))


def calc_downstream(matched_preds: list) -> dict:
    """Calculate Parse, Symbol, Retrieval, Ready for matched pred units."""
    if not matched_preds:
        return {"Parse": 0.0, "Symbol": 0.0, "Retrieval": 0.0, "Ready": 0.0}
    parse_s, symbol_s, retrieval_s = [], [], []
    for p in matched_preds:
        parse_s.append(score_parse(p))
        symbol_s.append(score_symbol(p))
        retrieval_s.append(score_retrieval(p))
    avg = lambda xs: sum(xs) / len(xs)
    ready = avg([(a + b + c) / 3 for a, b, c in zip(parse_s, symbol_s, retrieval_s)])
    return {
        "Parse": round(avg(parse_s) * 100, 1),
        "Symbol": round(avg(symbol_s) * 100, 1),
        "Retrieval": round(avg(retrieval_s) * 100, 1),
        "Ready": round(ready * 100, 1),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluation v2")
    parser.add_argument("--gold", required=True, help="Gold JSON file")
    parser.add_argument("--pred", required=True, help="Pred JSON file")
    parser.add_argument("--output", default="", help="Output JSON file")
    parser.add_argument("--system_name", default="")
    parser.add_argument("--step", default="match",
                        choices=["match", "structure", "content", "formula", "downstream", "all"])
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_units = json.load(f)
    with open(args.pred, "r", encoding="utf-8") as f:
        pred_units = json.load(f)

    matched, gold_only, pred_only = match_units(gold_units, pred_units)

    if args.step in ("match", "all"):
        print(f"=== Matching ===")
        print(f"  Gold count:     {len(gold_units)}")
        print(f"  Pred count:     {len(pred_units)}")
        print(f"  Matched:        {len(matched)}")
        print(f"  Gold unmatched: {len(gold_only)}")
        print(f"  Pred extra:     {len(pred_only)}")
        if gold_only:
            print(f"  Missed gold: {[g.get('label','?') for g in gold_only]}")
        if pred_only:
            print(f"  Extra pred:  {[p.get('label','?') for p in pred_only]}")

    if args.step in ("structure", "all"):
        boundary = calc_boundary(matched, len(gold_units), len(pred_units))
        type_acc = calc_type(matched)
        print(f"=== Structure ===")
        print(f"  B-P:  {boundary['B-P']}")
        print(f"  B-R:  {boundary['B-R']}")
        print(f"  B-F1: {boundary['B-F1']}")
        print(f"  Type: {type_acc}")

    if args.step in ("content", "all"):
        compl = calc_completeness(matched)
        edit = calc_edit(matched)
        print(f"=== Content ===")
        print(f"  Compl: {compl}")
        print(f"  Edit:  {edit}")

    if args.step in ("formula", "all"):
        f_attach = calc_formula_attach(matched)
        print(f"=== Formula ===")
        print(f"  F-Attach: {f_attach}")

    if args.step in ("downstream", "all"):
        matched_preds = [p for _, p in matched]
        downstream = calc_downstream(matched_preds)
        print(f"=== Downstream ===")
        print(f"  Parse:     {downstream['Parse']}")
        print(f"  Symbol:    {downstream['Symbol']}")
        print(f"  Retrieval: {downstream['Retrieval']}")
        print(f"  Ready:     {downstream['Ready']}")

    if args.step == "all" and args.output:
        boundary = calc_boundary(matched, len(gold_units), len(pred_units))
        results = {
            "system_name": args.system_name,
            "table2": {
                **boundary,
                "Type": calc_type(matched),
                "F-Attach": calc_formula_attach(matched),
                "Compl": calc_completeness(matched),
                "Edit": calc_edit(matched),
                "Ready": calc_downstream([p for _, p in matched])["Ready"],
            },
            "table4": calc_downstream([p for _, p in matched]),
            "detail": {
                "gold_count": len(gold_units),
                "pred_count": len(pred_units),
                "matched_count": len(matched),
                "missed": [g.get("label", "?") for g in gold_only],
                "false_positives": [p.get("label", "?") for p in pred_only],
            },
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
