import argparse
import csv
import json
from pathlib import Path
from typing import Any

from latex_table_utils import write_latex_table


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_table1(stats):
    fieldnames = ["split","pages","ocr_blocks","theorem_units","definitions","theorem_like","examples","avg_blocks_per_unit","avg_formulas_per_unit","multi_block_unit_ratio"]
    csv_rows, latex_rows = [], []
    for split in ["train","dev","test","total"]:
        s = stats[split]
        row = {"split": split.capitalize(), "pages": s["pages"], "ocr_blocks": s["ocr_blocks"], "theorem_units": s["theorem_units"], "definitions": s["definitions"], "theorem_like": s["theorem_like"], "examples": s["examples"], "avg_blocks_per_unit": s["avg_blocks_per_unit"], "avg_formulas_per_unit": s["avg_formulas_per_unit"], "multi_block_unit_ratio": s["multi_block_unit_ratio"]}
        csv_rows.append(row)
        latex_rows.append([row["split"], row["pages"], row["ocr_blocks"], row["theorem_units"], row["definitions"], row["theorem_like"], row["examples"], row["avg_blocks_per_unit"], row["avg_formulas_per_unit"], row["multi_block_unit_ratio"]])
    return fieldnames, csv_rows, latex_rows


def build_table2(system_metrics):
    fieldnames = ["method","boundary_p","boundary_r","boundary_f1","unit_type_acc","formula_attach_acc","statement_completeness","norm_edit_dist","formalization_readiness"]
    csv_rows, latex_rows = [], []
    for m in system_metrics:
        row = {"method": m["system_name"], "boundary_p": m["structure"]["boundary_precision"], "boundary_r": m["structure"]["boundary_recall"], "boundary_f1": m["structure"]["boundary_f1"], "unit_type_acc": m["structure"]["unit_type_accuracy"], "formula_attach_acc": m["integrity"]["formula_attachment_accuracy"], "statement_completeness": m["integrity"]["statement_completeness"], "norm_edit_dist": m["integrity"]["normalized_edit_distance"], "formalization_readiness": m["downstream"]["formalization_readiness"]}
        csv_rows.append(row)
        latex_rows.append([row["method"], row["boundary_p"], row["boundary_r"], row["boundary_f1"], row["unit_type_acc"], row["formula_attach_acc"], row["statement_completeness"], row["norm_edit_dist"], row["formalization_readiness"]])
    return fieldnames, csv_rows, latex_rows


def build_table4(system_metrics):
    mapping = []
    for m in system_metrics:
        name = m["system_name"]
        if name == "Rule-based":
            input_name = "Raw OCR / weak baseline"
        elif name == "LLM-only":
            input_name = "LLM-only theorem units"
        elif name == "Hybrid baseline":
            input_name = "Hybrid baseline"
        elif name == "Ours":
            input_name = "Proposed theorem units"
        else:
            input_name = name
        mapping.append((input_name, m))
    fieldnames = ["input_representation","structured_parse_success","symbol_stability","retrieval_compatibility","formalization_readiness"]
    csv_rows, latex_rows = [], []
    for input_name, m in mapping:
        row = {"input_representation": input_name, "structured_parse_success": m["downstream"]["structured_parse_success"], "symbol_stability": m["downstream"]["symbol_stability"], "retrieval_compatibility": m["downstream"]["retrieval_compatibility"], "formalization_readiness": m["downstream"]["formalization_readiness"]}
        csv_rows.append(row)
        latex_rows.append([row["input_representation"], row["structured_parse_success"], row["symbol_stability"], row["retrieval_compatibility"], row["formalization_readiness"]])
    return fieldnames, csv_rows, latex_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_stats", required=True)
    parser.add_argument("--rule_metrics", required=True)
    parser.add_argument("--llm_metrics", required=True)
    parser.add_argument("--hybrid_metrics", required=True)
    parser.add_argument("--ours_metrics", required=True)
    parser.add_argument("--output_dir_csv", required=True)
    parser.add_argument("--output_dir_tex", required=True)
    args = parser.parse_args()
    stats = load_json(Path(args.benchmark_stats))
    system_metrics = [load_json(Path(args.rule_metrics)), load_json(Path(args.llm_metrics)), load_json(Path(args.hybrid_metrics)), load_json(Path(args.ours_metrics))]
    out_csv = Path(args.output_dir_csv); out_tex = Path(args.output_dir_tex)
    out_csv.mkdir(parents=True, exist_ok=True); out_tex.mkdir(parents=True, exist_ok=True)
    t1_fields, t1_csv_rows, t1_latex_rows = build_table1(stats)
    write_csv(out_csv / "table1_stats.csv", t1_fields, t1_csv_rows)
    write_latex_table(out_tex / "table1_stats.tex", ["Split", "# Pages", "# OCR Blocks", "# Theorem Units", "# Definitions", "# Thm-like", "# Examples", "Avg. Blocks/Unit", "Avg. Formulas/Unit", "% Multi-block"], t1_latex_rows, "Benchmark statistics for theorem unit construction from OCR-derived mathematical pages.", "tab:benchmark_stats", "lrrrrrrrrr")
    t2_fields, t2_csv_rows, t2_latex_rows = build_table2(system_metrics)
    write_csv(out_csv / "table2_main.csv", t2_fields, t2_csv_rows)
    write_latex_table(out_tex / "table2_main.tex", ["Method", "Boundary P", "Boundary R", "Boundary F1", "Unit Type Acc.", "Formula Attach. Acc.", "Statement Completeness", "Norm. Edit Dist. $\downarrow$", "Formalization Readiness"], t2_latex_rows, "Main benchmark results under theorem unit reconstruction from noisy OCR JSON.", "tab:main_results", "lrrrrrrrr")
    t4_fields, t4_csv_rows, t4_latex_rows = build_table4(system_metrics)
    write_csv(out_csv / "table4_downstream.csv", t4_fields, t4_csv_rows)
    write_latex_table(out_tex / "table4_downstream.tex", ["Input Representation", "Structured Parse Success", "Symbol Stability", "Retrieval Compatibility", "Formalization Readiness"], t4_latex_rows, "Downstream theorem usability under different input constructions.", "tab:downstream_results", "lrrrr")
    print(f"Wrote CSV files to {out_csv}")
    print(f"Wrote LaTeX files to {out_tex}")


if __name__ == "__main__":
    main()
