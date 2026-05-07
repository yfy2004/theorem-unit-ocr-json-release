import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from latex_table_utils import write_latex_table

def load_jsonl(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if line: rows.append(json.loads(line))
    return rows

def boundary_key(unit):
    blocks=unit.get('source_blocks', [])
    return (unit['page_id'], -1, -1) if not blocks else (unit['page_id'], min(blocks), max(blocks))

def formula_block_set(unit):
    out=set()
    for f in unit.get('formula_spans', []):
        sb=f.get('source_block')
        if isinstance(sb, int): out.add(sb)
    return out

def group_by_page(units):
    g=defaultdict(list)
    for u in units: g[u['page_id']].append(u)
    return g

def compute_boundary_agreement(a_units,b_units):
    a_keys={boundary_key(u) for u in a_units}; b_keys={boundary_key(u) for u in b_units}; union=a_keys|b_keys; inter=a_keys&b_keys
    return 1.0 if not union else len(inter)/len(union)

def compute_type_agreement(a_units,b_units):
    a_map={boundary_key(u):u for u in a_units}; b_map={boundary_key(u):u for u in b_units}; matched=set(a_map)&set(b_map)
    return 1.0 if not matched else sum(1 for k in matched if a_map[k].get('unit_type')==b_map[k].get('unit_type'))/len(matched)

def compute_formula_agreement(a_units,b_units):
    a_map={boundary_key(u):u for u in a_units}; b_map={boundary_key(u):u for u in b_units}; matched=set(a_map)&set(b_map)
    return 1.0 if not matched else sum(1 for k in matched if formula_block_set(a_map[k])==formula_block_set(b_map[k]))/len(matched)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--annotator_a', required=True); p.add_argument('--annotator_b', required=True); p.add_argument('--adjudicated', required=True); p.add_argument('--adjudication_policy', default='discussion + first-author adjudication'); p.add_argument('--output_json', required=True); p.add_argument('--output_csv', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    a=load_jsonl(args.annotator_a); b=load_jsonl(args.annotator_b); adj=load_jsonl(args.adjudicated)
    a_by=group_by_page(a); b_by=group_by_page(b); pages=sorted(set(a_by)|set(b_by))
    b_scores=[]; t_scores=[]; f_scores=[]
    for page_id in pages:
        au=a_by.get(page_id, []); bu=b_by.get(page_id, [])
        b_scores.append(compute_boundary_agreement(au,bu)); t_scores.append(compute_type_agreement(au,bu)); f_scores.append(compute_formula_agreement(au,bu))
    avg=lambda xs: sum(xs)/len(xs) if xs else 0.0
    report={'double_annotated_pages': len(pages), 'annotators': 2, 'boundary_exact_match_agreement': round(avg(b_scores),4), 'unit_type_agreement': round(avg(t_scores),4), 'formula_attachment_agreement': round(avg(f_scores),4), 'adjudicated_unit_count': len(adj), 'adjudication_policy': args.adjudication_policy}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True); Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with Path(args.output_csv).open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['metric','value']);
        [w.writerow([k,v]) for k,v in report.items()]
    rows=[[report['double_annotated_pages'], report['boundary_exact_match_agreement'], report['unit_type_agreement'], report['formula_attachment_agreement'], report['adjudication_policy']]]
    write_latex_table(Path(args.output_tex), ['# Doubly Annotated Pages','Boundary Agreement','Unit Type Agreement','Formula Attachment Agreement','Conflict Resolution Policy'], rows, 'Annotation agreement and quality control on a doubly annotated subset.', 'tab:agreement', 'rrrrl')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
