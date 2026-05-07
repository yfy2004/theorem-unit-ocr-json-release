import argparse, csv
from pathlib import Path
from eval_structure import load_jsonl, evaluate_structure
from eval_integrity import evaluate_integrity
from eval_downstream import evaluate_downstream
from latex_table_utils import bold_best, write_latex_table

def load_page_ids(path): return {line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()}
def filter_by_pages(units, page_ids): return [u for u in units if u.get('page_id') in page_ids]
def evaluate_subset(gold,pred):
    s=evaluate_structure(gold,pred); i=evaluate_integrity(gold,pred); d=evaluate_downstream(gold,pred)
    return {'boundary_f1': s['boundary_f1'], 'formula_attachment_accuracy': i['formula_attachment_accuracy'], 'statement_completeness': i['statement_completeness'], 'formalization_readiness': d['formalization_readiness']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gold', required=True); p.add_argument('--challenge_pages', required=True); p.add_argument('--rule_pred', required=True); p.add_argument('--llm_pred', required=True); p.add_argument('--hybrid_pred', required=True); p.add_argument('--ours_pred', required=True); p.add_argument('--output_csv', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    gold=load_jsonl(Path(args.gold)); pages=load_page_ids(args.challenge_pages); gold_subset=filter_by_pages(gold,pages)
    systems=[('Rule-based', load_jsonl(Path(args.rule_pred))), ('LLM-only', load_jsonl(Path(args.llm_pred))), ('Hybrid baseline', load_jsonl(Path(args.hybrid_pred))), ('Ours', load_jsonl(Path(args.ours_pred)))]
    rows=[]
    for name,pred in systems:
        res=evaluate_subset(gold_subset, filter_by_pages(pred,pages)); rows.append([name, res['boundary_f1'], res['formula_attachment_accuracy'], res['statement_completeness'], res['formalization_readiness']])
    out=Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['method','boundary_f1','formula_attachment_accuracy','statement_completeness','formalization_readiness']); w.writerows(rows)
    latex_rows=bold_best(rows, metric_cols=[1,2,3,4], higher_is_better={1:True,2:True,3:True,4:True})
    write_latex_table(Path(args.output_tex), ['Method','Boundary F1','Formula Attach. Acc.','Statement Completeness','Formalization Readiness'], latex_rows, 'Results on structurally difficult pages.', 'tab:challenge_subset', 'lrrrr')
    print(f'Wrote CSV to {out}'); print(f'Wrote LaTeX to {args.output_tex}')
if __name__=='__main__': main()
