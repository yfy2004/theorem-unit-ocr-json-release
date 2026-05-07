import argparse, csv, json
from pathlib import Path
from latex_table_utils import bold_best, write_latex_table

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def extract_row(name, metrics):
    return [name, metrics["structure"]["boundary_f1"], metrics["structure"]["unit_type_accuracy"], metrics["integrity"]["formula_attachment_accuracy"], metrics["integrity"]["statement_completeness"], metrics["downstream"]["formalization_readiness"]]

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--full_metrics', required=True); p.add_argument('--wo_normalization', required=True); p.add_argument('--wo_alignment', required=True); p.add_argument('--wo_boundary_reranking', required=True); p.add_argument('--wo_ocr_correction', required=True); p.add_argument('--wo_confidence_aggregation', required=True); p.add_argument('--output_csv', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    systems=[('Full system', load_json(args.full_metrics)), ('w/o normalization', load_json(args.wo_normalization)), ('w/o alignment', load_json(args.wo_alignment)), ('w/o boundary reranking', load_json(args.wo_boundary_reranking)), ('w/o OCR correction', load_json(args.wo_ocr_correction)), ('w/o confidence aggregation', load_json(args.wo_confidence_aggregation))]
    rows=[extract_row(n,m) for n,m in systems]
    out=Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['variant','boundary_f1','unit_type_accuracy','formula_attachment_accuracy','statement_completeness','formalization_readiness']); w.writerows(rows)
    latex_rows=bold_best(rows, metric_cols=[1,2,3,4,5], higher_is_better={1:True,2:True,3:True,4:True,5:True})
    write_latex_table(Path(args.output_tex), ['Variant','Boundary F1','Unit Type Acc.','Formula Attach. Acc.','Statement Completeness','Formalization Readiness'], latex_rows, 'Factor analysis of theorem unit quality via ablation.', 'tab:ablation', 'lrrrrr')
    print(f'Wrote CSV to {out}'); print(f'Wrote LaTeX to {args.output_tex}')
if __name__=='__main__': main()
