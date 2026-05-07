"""Quick tables output - reads all data and prints results."""
import json, sys, re
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from eval_v2 import (match_units, calc_boundary, calc_type,
                      calc_completeness, calc_edit, calc_formula_attach,
                      calc_downstream, score_parse, score_symbol)

METHODS = {'rule':'predictions/rule','llm':'predictions/llm',
           'hybrid':'predictions/hybrid','ours':'predictions/ours',
           'ours_no_llm':'predictions/ours_no_llm'}
GOLD_DIR = 'Klenke14_revised/json'
OCR_DIR = 'real_data/Klenke14/OCR'
CHAPTERS = list(range(1, 27))
FORMULA_RE = re.compile(r'\$\$.*?\$\$|\$[^$]+\$', re.DOTALL)

results = {}
for method, pred_dir in METHODS.items():
    all_matched, total_gold, total_pred, all_preds = [], 0, 0, []
    for ch in CHAPTERS:
        gf, pf = Path(GOLD_DIR)/f'Chapter{ch}.json', Path(pred_dir)/f'Chapter{ch}_pred.json'
        if not gf.exists() or not pf.exists(): continue
        gold = json.load(gf.open('r', encoding='utf-8'))
        pred = json.load(pf.open('r', encoding='utf-8'))
        matched, _, _ = match_units(gold, pred)
        all_matched.extend(matched)
        total_gold += len(gold); total_pred += len(pred)
        all_preds.extend(pred)
    results[method] = {'matched':all_matched, 'total_gold':total_gold,
                       'total_pred':total_pred, 'all_preds':all_preds}

# Table 1
total_pages, total_blocks, total_gold_units = 0, 0, 0
env_cnt = Counter(); gold_formula_count = 0; gold_with_formula = 0
for ch in CHAPTERS:
    gold = json.load((Path(GOLD_DIR)/f'Chapter{ch}.json').open('r',encoding='utf-8'))
    total_gold_units += len(gold)
    ch_ocr = Path(OCR_DIR)/f'Chapter{ch}'
    total_pages += len(list(ch_ocr.glob('p*.json')))
    for pf in ch_ocr.glob('p*.json'):
        total_blocks += len(json.load(pf.open('r',encoding='utf-8')).get('blocks',[]))
    for u in gold:
        env_cnt[u.get('env','?')] += 1
        c = u.get('content','')
        f = FORMULA_RE.findall(c)
        gold_formula_count += len(f)
        if f: gold_with_formula += 1

blocks_per_unit = []
for ch in CHAPTERS:
    pred = json.load((Path('predictions/rule')/f'Chapter{ch}_pred.json').open('r',encoding='utf-8'))
    for u in pred:
        blocks_per_unit.append(len(list(u.get('_source_blocks',[]))+list(u.get('_proof_blocks',[]))))

print("TABLE1_DATA = {")
print(f'  "pages": {total_pages},')
print(f'  "blocks": {total_blocks},')
print(f'  "units": {total_gold_units},')
print(f'  "avg_blocks_per_unit": {sum(blocks_per_unit)/len(blocks_per_unit):.1f},')
print(f'  "multi_block_ratio": {sum(1 for b in blocks_per_unit if b>1)/len(blocks_per_unit)*100:.1f},')
print(f'  "avg_formulas_per_unit": {gold_formula_count/total_gold_units:.1f},')
for env in ['thm','exr','def','ex','rem','lem','cor']:
    print(f'  "{env}": {env_cnt.get(env,0)},')
print("}")

# Table 2
print("\nTABLE2_DATA = {")
for method in ['rule','llm','hybrid','ours']:
    r = results[method]
    b = calc_boundary(r['matched'], r['total_gold'], r['total_pred'])
    t = calc_type(r['matched'])
    comp = calc_completeness(r['matched'])
    edit = calc_edit(r['matched'])
    fa = calc_formula_attach(r['matched'])
    mp = [p for _,p in r['matched']]
    ds = calc_downstream(mp)
    print(f'  "{method}": {{"B-P":{b["B-P"]},"B-R":{b["B-R"]},"B-F1":{b["B-F1"]},"Type":{t},"F-Att":{fa},"Compl":{comp},"Edit":{edit},"Parse":{ds["Parse"]},"Symbol":{ds["Symbol"]},"Retrieval":{ds["Retrieval"]},"Ready":{ds["Ready"]}}},')
print("}")

# Table 3
ENV_NAMES = {'thm':'Theorem','exr':'Exercise','def':'Definition','ex':'Example','rem':'Remark','lem':'Lemma','cor':'Corollary'}
print("\nTABLE3_DATA = {")
for method in ['rule','llm','hybrid','ours']:
    r = results[method]
    by_type = defaultdict(list)
    for g,p in r['matched']: by_type[g.get('env','?')].append((g,p))
    print(f'  "{method}": {{')
    for env in ['thm','exr','def','ex','rem','lem','cor']:
        pairs = by_type.get(env,[])
        if not pairs: continue
        preds = [p for _,p in pairs]
        ds = calc_downstream(preds)
        print(f'    "{env}": {{"n":{len(pairs)},"Parse":{ds["Parse"]},"Symbol":{ds["Symbol"]},"Retrieval":{ds["Retrieval"]},"Ready":{ds["Ready"]}}},')
    print('  },')
print("}")

# Table 9
print("\nTABLE9_DATA = {")
for method, label in [('rule','Rule'),('hybrid','Hybrid'),('ours_no_llm','Norm+Rule'),('ours','Ours')]:
    r = results[method]
    b = calc_boundary(r['matched'], r['total_gold'], r['total_pred'])
    mp = [p for _,p in r['matched']]
    ds = calc_downstream(mp)
    print(f'  "{label}": {{"B-F1":{b["B-F1"]},"Parse":{ds["Parse"]},"Symbol":{ds["Symbol"]},"Retrieval":{ds["Retrieval"]},"Ready":{ds["Ready"]}}},')
print("}")

# Table 10
print("\nTABLE10_DATA = {")
for method in ['rule','llm','hybrid','ours']:
    r = results[method]
    fn = r['total_gold'] - len(r['matched'])
    fp = r['total_pred'] - len(r['matched'])
    fffd = sum((u.get('content','')+u.get('proof','')).count('\ufffd') for u in r['all_preds'])
    pf = sum(1 for u in r['all_preds'] if score_parse(u)<1.0)
    sf = sum(1 for u in r['all_preds'] if score_symbol(u)<1.0)
    print(f'  "{method}": {{"FN":{fn},"FP":{fp},"FFFD":{fffd},"parse_fail":{pf},"symbol_fail":{sf}}},')
print("}")

# Table 11
print("\nTABLE11_DATA = {")
for method in ['rule','llm','hybrid','ours']:
    if method == 'rule':
        print(f'  "rule": {{"calls":0,"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"wall_time":0}},')
        continue
    agg = {"calls":0,"prompt":0,"completion":0,"total":0,"wall":0}
    for sf in sorted(Path(f'predictions/{method}').glob('*_stats.json')):
        s = json.load(sf.open('r',encoding='utf-8'))
        agg["calls"] += s.get("llm_calls",0)
        agg["prompt"] += s.get("prompt_tokens",0)
        agg["completion"] += s.get("completion_tokens",0)
        agg["total"] += s.get("total_tokens",0)
        agg["wall"] += s.get("wall_time_seconds",0)
    print(f'  "{method}": {{"calls":{agg["calls"]},"prompt_tokens":{agg["prompt"]},"completion_tokens":{agg["completion"]},"total_tokens":{agg["total"]},"wall_time":{agg["wall"]:.0f}}},')
print("}")
