import argparse
import json
import re
from pathlib import Path
from typing import Any

REQUIRED_MERGED_KEYS = {"structure": ["boundary_precision","boundary_recall","boundary_f1","unit_type_accuracy","block_assignment_accuracy"], "integrity": ["formula_attachment_accuracy","statement_completeness","normalized_edit_distance"], "downstream": ["structured_parse_success","symbol_stability","retrieval_compatibility","formalization_readiness"]}
DANGER_PATTERNS = [r"illustrative draft results", r"replace with final", r"placeholder", r"todo", r"tbd"]

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def check_benchmark_stats(stats):
    errors = []
    for split in ["train","dev","test","total"]:
        if split not in stats:
            errors.append(f"Missing split in benchmark_stats.json: {split}")
            return errors
    numeric_keys = ["pages","ocr_blocks","theorem_units","definitions","theorem_like","examples"]
    for key in numeric_keys:
        lhs = stats["train"][key] + stats["dev"][key] + stats["test"][key]
        rhs = stats["total"][key]
        if lhs != rhs:
            errors.append(f"Benchmark stats mismatch for '{key}': train+dev+test={lhs}, total={rhs}")
    return errors

def check_merged_metrics(metrics, name):
    errors = []
    for section, keys in REQUIRED_MERGED_KEYS.items():
        if section not in metrics:
            errors.append(f"[{name}] Missing section: {section}")
            continue
        for key in keys:
            if key not in metrics[section]:
                errors.append(f"[{name}] Missing key: {section}.{key}")
    return errors

def check_required_files(paths):
    return [f"Missing required file: {p}" for p in paths if not p.exists()]

def scan_text_for_danger(path: Path):
    errors = []
    if not path.exists():
        return [f"Cannot scan missing file: {path}"]
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pat in DANGER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            errors.append(f"Danger pattern found in {path.name}: /{pat}/")
    return errors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_stats", required=True)
    parser.add_argument("--rule_metrics", required=True)
    parser.add_argument("--llm_metrics", required=True)
    parser.add_argument("--hybrid_metrics", required=True)
    parser.add_argument("--ours_metrics", required=True)
    parser.add_argument("--main_tex", required=True)
    parser.add_argument("--required_table_files", nargs="*", default=[])
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()
    errors, warnings = [], []
    paths = [Path(args.benchmark_stats), Path(args.rule_metrics), Path(args.llm_metrics), Path(args.hybrid_metrics), Path(args.ours_metrics), Path(args.main_tex)] + [Path(p) for p in args.required_table_files]
    errors.extend(check_required_files(paths))
    if not errors:
        errors.extend(check_benchmark_stats(load_json(Path(args.benchmark_stats))))
        errors.extend(check_merged_metrics(load_json(Path(args.rule_metrics)), "Rule-based"))
        errors.extend(check_merged_metrics(load_json(Path(args.llm_metrics)), "LLM-only"))
        errors.extend(check_merged_metrics(load_json(Path(args.hybrid_metrics)), "Hybrid baseline"))
        errors.extend(check_merged_metrics(load_json(Path(args.ours_metrics)), "Ours"))
        warnings.extend(scan_text_for_danger(Path(args.main_tex)))
    report = {"status": "PASS" if not errors else "FAIL", "error_count": len(errors), "warning_count": len(warnings), "errors": errors, "warnings": warnings}
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
