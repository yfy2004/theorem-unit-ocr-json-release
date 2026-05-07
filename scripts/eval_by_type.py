import argparse, csv
from pathlib import Path
from eval_structure import load_jsonl, evaluate_structure
from eval_integrity import evaluate_integrity
from eval_downstream import evaluate_downstream
from latex_table_utils import bold_best, write_latex_table
GROUPS={"Definition":{"definition"},"Theorem/Lemma/Prop./Cor.":{"theorem","lemma","proposition","corollary"},"Example":{"example"},"Other theorem-like unit":{"other"}}

def filter_by_types(units, allowed): return [u for u in units if u.get('unit_type') in allowed]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gold', required=True); p.add_argument('--pred', required=True); p.add_argument('--output_csv', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    gold=load_jsonl(Path(args.gold)); pred=load_jsonl(Path(args.pred)); rows=[]
    for group_name, allowed in GROUPS.items():
        gs=filter_by_types(gold, allowed); ps=filter_by_types(pred, allowed)
        if not gs: rows.append([group_name,'-','-','-','-','-']); continue
        s=evaluate_structure(gs, ps); i=evaluate_integrity(gs, ps); d=evaluate_downstream(gs, ps)
        rows.append([group_name, s['boundary_f1'], s['unit_type_accuracy'], i['formula_attachment_accuracy'], i['statement_completeness'], d['formalization_readiness']])
    out=Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['unit_type_group','boundary_f1','unit_type_accuracy','formula_attachment_accuracy','statement_completeness','formalization_readiness']); w.writerows(rows)
    latex_rows=bold_best(rows, metric_cols=[1,2,3,4,5], higher_is_better={1:True,2:True,3:True,4:True,5:True})
    write_latex_table(Path(args.output_tex), ['Unit Type','Boundary F1','Unit Type Acc.','Formula Attach. Acc.','Statement Completeness','Formalization Readiness'], latex_rows, 'Benchmark results by theorem-unit type.', 'tab:by_type', 'lrrrrr')
    print(f'Wrote CSV to {out}'); print(f'Wrote LaTeX to {args.output_tex}')
if __name__=='__main__': main()
