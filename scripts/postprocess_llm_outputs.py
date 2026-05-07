import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

VALID_TYPES = {"definition", "theorem", "lemma", "proposition", "corollary", "example", "other"}
TYPE_MAP = {"def": "definition", "definition": "definition", "theorem": "theorem", "thm": "theorem", "lemma": "lemma", "lem": "lemma", "proposition": "proposition", "prop": "proposition", "corollary": "corollary", "cor": "corollary", "example": "example", "ex": "example", "other": "other"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_blob(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start_candidates = [i for i in [text.find("["), text.find("{")] if i != -1]
    if not start_candidates:
        raise ValueError("No JSON start found in raw_output")
    start = min(start_candidates)
    sub = text[start:]
    opening = sub[0]
    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(sub):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                candidate = sub[: i + 1]
                return json.loads(candidate)
    raise ValueError("Could not extract valid JSON blob from raw_output")


def normalize_whitespace(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def get_blocks_list(ocr_obj: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["blocks", "items", "layout_blocks", "elements"]:
        if key in ocr_obj and isinstance(ocr_obj[key], list):
            return ocr_obj[key]
    return []


def get_valid_block_ids(ocr_json_path: Path) -> list[int]:
    ocr_obj = load_json(ocr_json_path)
    blocks = get_blocks_list(ocr_obj)
    valid_ids = []
    for idx, blk in enumerate(blocks):
        if isinstance(blk, dict):
            if "block_id" in blk and isinstance(blk["block_id"], int):
                valid_ids.append(blk["block_id"])
            elif "id" in blk and isinstance(blk["id"], int):
                valid_ids.append(blk["id"])
            else:
                valid_ids.append(idx)
        else:
            valid_ids.append(idx)
    return valid_ids


def normalize_block_list(blocks: Any, valid_ids: set[int]) -> list[int]:
    if blocks is None:
        return []
    out = []
    if isinstance(blocks, list):
        items = blocks
    elif isinstance(blocks, str):
        items = re.split(r"[,\s;]+", blocks.strip())
    else:
        items = [blocks]
    for x in items:
        if x == "" or x is None:
            continue
        try:
            val = int(x)
        except Exception:
            continue
        if val in valid_ids and val not in out:
            out.append(val)
    return out


def canonicalize_unit_type(x: Any) -> str:
    s = normalize_whitespace(str(x).lower())
    return TYPE_MAP.get(s, "other")


def normalize_formula_spans(page_id: str, formula_spans: Any, valid_ids: set[int]) -> list[dict[str, Any]]:
    if not formula_spans:
        return []
    out = []
    if isinstance(formula_spans, str):
        formula_spans = [{"text": formula_spans}]
    elif isinstance(formula_spans, dict):
        formula_spans = [formula_spans]
    if not isinstance(formula_spans, list):
        return []
    for i, f in enumerate(formula_spans, start=1):
        if isinstance(f, str):
            text = normalize_whitespace(f)
            source_block = None
        elif isinstance(f, dict):
            text = normalize_whitespace(f.get("text", ""))
            source_block = f.get("source_block", None)
        else:
            continue
        if not text:
            continue
        try:
            sb = int(source_block) if source_block is not None else None
        except Exception:
            sb = None
        if sb is not None and sb not in valid_ids:
            sb = None
        if sb is None:
            continue
        out.append({"formula_id": f"f_{page_id}_{i:02d}", "text": text, "source_block": sb})
    return out


def deduplicate_stage1(rows):
    seen = set(); out = []
    for row in rows:
        key = (row["page_id"], tuple(row["source_blocks"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def deduplicate_stage2(rows):
    grouped = defaultdict(list)
    for row in rows:
        norm_stmt = normalize_whitespace(row.get("statement_text", "")).lower()
        key = (row["page_id"], tuple(row["source_blocks"]), row["unit_type"], norm_stmt)
        grouped[key].append(row)
    out = []
    for _, candidates in grouped.items():
        best = max(candidates, key=lambda r: float(r.get("confidence_score", 0.0)))
        out.append(best)
    return out


def stage1_postprocess(input_jsonl: Path, ocr_json_dir: Path, output_jsonl: Path) -> None:
    records = load_jsonl(input_jsonl)
    output_rows = []
    for rec in records:
        page_id = rec["page_id"]
        raw_output = rec["raw_output"]
        ocr_json_path = ocr_json_dir / f"{page_id}.json"
        valid_ids = set(get_valid_block_ids(ocr_json_path))
        parsed = extract_json_blob(raw_output)
        if isinstance(parsed, dict):
            parsed = parsed["units"] if "units" in parsed and isinstance(parsed["units"], list) else [parsed]
        if not isinstance(parsed, list):
            continue
        local_idx = 1
        for item in parsed:
            if not isinstance(item, dict):
                continue
            pred_page_id = item.get("page_id", page_id)
            if pred_page_id != page_id:
                pred_page_id = page_id
            source_blocks = normalize_block_list(item.get("source_blocks"), valid_ids)
            if not source_blocks:
                continue
            output_rows.append({"page_id": pred_page_id, "candidate_id": f"cand_{page_id}_{local_idx:02d}", "source_blocks": source_blocks})
            local_idx += 1
    output_rows = deduplicate_stage1(output_rows)
    write_jsonl(output_jsonl, output_rows)
    print(f"Wrote {len(output_rows)} stage1 candidates to {output_jsonl}")


def stage2_postprocess(input_jsonl: Path, ocr_json_dir: Path, output_jsonl: Path) -> None:
    records = load_jsonl(input_jsonl)
    output_rows = []
    page_counters = defaultdict(int)
    for rec in records:
        doc_id = rec.get("doc_id", "")
        page_id = rec["page_id"]
        raw_output = rec["raw_output"]
        fallback_blocks = rec.get("candidate_source_blocks") or rec.get("source_blocks") or []
        ocr_json_path = ocr_json_dir / f"{page_id}.json"
        valid_ids = set(get_valid_block_ids(ocr_json_path))
        parsed = extract_json_blob(raw_output)
        if isinstance(parsed, list):
            objects = [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            objects = [parsed]
        else:
            objects = []
        for obj in objects:
            page_counters[page_id] += 1
            unit_type = canonicalize_unit_type(obj.get("unit_type"))
            title = normalize_whitespace(obj.get("title", ""))
            statement_text = normalize_whitespace(obj.get("statement_text", ""))
            source_blocks = normalize_block_list(obj.get("source_blocks"), valid_ids)
            if not source_blocks:
                source_blocks = normalize_block_list(fallback_blocks, valid_ids)
            if not source_blocks:
                continue
            reading_order = normalize_block_list(obj.get("reading_order"), valid_ids)
            if not reading_order or not set(reading_order).issubset(set(source_blocks)):
                reading_order = list(source_blocks)
            formula_spans = normalize_formula_spans(page_id=page_id, formula_spans=obj.get("formula_spans", []), valid_ids=valid_ids)
            if not statement_text:
                continue
            conf = obj.get("confidence_score", 0.5)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.5
            conf = min(max(conf, 0.0), 1.0)
            pred_unit_id = obj.get("unit_id", "") or f"pred_{page_id}_{page_counters[page_id]:02d}"
            row = {"doc_id": doc_id or obj.get("doc_id", ""), "page_id": page_id, "unit_id": pred_unit_id, "unit_type": unit_type, "title": title, "statement_text": statement_text, "formula_spans": formula_spans, "source_blocks": source_blocks, "reading_order": reading_order, "confidence_score": round(conf, 4)}
            output_rows.append(row)
    output_rows = deduplicate_stage2(output_rows)
    final_rows = []
    final_page_counters = defaultdict(int)
    for row in sorted(output_rows, key=lambda x: (x["page_id"], x["source_blocks"])):
        final_page_counters[row["page_id"]] += 1
        row["unit_id"] = f"pred_{row['page_id']}_{final_page_counters[row['page_id']]:02d}"
        final_rows.append(row)
    write_jsonl(output_jsonl, final_rows)
    print(f"Wrote {len(final_rows)} stage2 predictions to {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    p1 = subparsers.add_parser("stage1")
    p1.add_argument("--input_jsonl", required=True)
    p1.add_argument("--ocr_json_dir", required=True)
    p1.add_argument("--output_jsonl", required=True)
    p2 = subparsers.add_parser("stage2")
    p2.add_argument("--input_jsonl", required=True)
    p2.add_argument("--ocr_json_dir", required=True)
    p2.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    if args.cmd == "stage1":
        stage1_postprocess(Path(args.input_jsonl), Path(args.ocr_json_dir), Path(args.output_jsonl))
    else:
        stage2_postprocess(Path(args.input_jsonl), Path(args.ocr_json_dir), Path(args.output_jsonl))


if __name__ == "__main__":
    main()
