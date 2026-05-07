import argparse, csv, json, re
from pathlib import Path
from statistics import mean
from latex_table_utils import write_latex_table

def load_jsonl(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if line: rows.append(json.loads(line))
    return rows

def tokenize(text): return re.findall(r"[A-Za-z0-9_]+|[^\s]", text or "")
def avg_tokens_per_unit(units): return round(mean([len(tokenize(u.get('statement_text',''))) for u in units]),1) if units else 0.0
def avg_formulas_per_unit(units): return round(mean([len(u.get('formula_spans',[])) for u in units]),1) if units else 0.0
def has_detached_heading(unit):
    title=(unit.get('title') or '').strip(); blocks=unit.get('source_blocks', [])
    return bool(title and len(blocks)>=2 and (blocks[1]-blocks[0])>1)
def detached_heading_ratio(units): return round(sum(1 for u in units if has_detached_heading(u))/len(units),3) if units else 0.0
def proof_adjacent_ratio(units): return round(sum(1 for u in units if 'proof_adjacent' in (u.get('notes') or '').lower() or 'adjacent_to_proof' in (u.get('notes') or '').lower())/len(units),3) if units else 0.0
def avg_ocr_confidence(units):
    vals=[float(u.get('confidence_score')) for u in units if isinstance(u.get('confidence_score'), (int,float))]
    return round(mean(vals),2) if vals else 0.0
def summarize(units):
    return {'avg_tokens_per_unit': avg_tokens_per_unit(units), 'avg_formulas_per_unit': avg_formulas_per_unit(units), 'detached_heading_ratio': detached_heading_ratio(units), 'proof_adjacent_ratio': proof_adjacent_ratio(units), 'avg_ocr_confidence': avg_ocr_confidence(units)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--gold_train', required=True); p.add_argument('--gold_dev', required=True); p.add_argument('--gold_test', required=True); p.add_argument('--output_json', required=True); p.add_argument('--output_csv', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    train=load_jsonl(args.gold_train); dev=load_jsonl(args.gold_dev); test=load_jsonl(args.gold_test); total=train+dev+test
    stats={'train': summarize(train), 'dev': summarize(dev), 'test': summarize(test), 'total': summarize(total)}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True); Path(args.output_json).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    with Path(args.output_csv).open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['split','avg_tokens_per_unit','avg_formulas_per_unit','detached_heading_ratio','proof_adjacent_ratio','avg_ocr_confidence'])
        for split in ['train','dev','test','total']:
            s=stats[split]; w.writerow([split, s['avg_tokens_per_unit'], s['avg_formulas_per_unit'], s['detached_heading_ratio'], s['proof_adjacent_ratio'], s['avg_ocr_confidence']])
    rows=[]
    for split in ['train','dev','test','total']:
        s=stats[split]; rows.append([split.capitalize(), s['avg_tokens_per_unit'], s['avg_formulas_per_unit'], s['detached_heading_ratio'], s['proof_adjacent_ratio'], s['avg_ocr_confidence']])
    write_latex_table(Path(args.output_tex), ['Statistic','Avg. tokens per unit','Avg. formulas per theorem-like unit','% units with detached headings','% units adjacent to proof text','Avg. OCR confidence per unit'], rows, 'Additional descriptive statistics of theorem units.', 'tab:descriptive_stats', 'lrrrrr')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
