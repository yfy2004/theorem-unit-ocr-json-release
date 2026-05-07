import argparse
import json
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct_from_ratio(x: float) -> float:
    return round(x * 100, 1)


def build_replacements(benchmark_stats, ours_metrics, agreement_report):
    total = benchmark_stats['total']
    return {
        'TWOBOUNDARYFONE': str(ours_metrics['structure']['boundary_f1']),
        'TWOFORMULAACC': str(ours_metrics['integrity']['formula_attachment_accuracy']),
        'TWOCOMPLETENESS': str(ours_metrics['integrity']['statement_completeness']),
        'TWOREADINESS': str(ours_metrics['downstream']['formalization_readiness']),
        'TOTALPAGES': str(total['pages']),
        'TOTALBLOCKS': str(total['ocr_blocks']),
        'TOTALUNITS': str(total['theorem_units']),
        'MULTIBLOCKRATIO': str(pct_from_ratio(total['multi_block_unit_ratio'])),
        'DOUBLEPAGES': str(agreement_report['double_annotated_pages']),
        'BOUNDARYAGREE': str(agreement_report['boundary_exact_match_agreement']),
        'TYPEAGREE': str(agreement_report['unit_type_agreement']),
        'FORMULAAGREE': str(agreement_report['formula_attachment_agreement']),
    }


def replace_newcommand(text: str, macro: str, value: str) -> str:
    pattern = rf"(\\newcommand\{{\\{re.escape(macro)}\}}\{{)([^}}]*)(\}})"
    new_text, count = re.subn(pattern, rf"\g<1>{value}\g<3>", text)
    if count == 0:
        new_text += f"\n\\newcommand{{\\{macro}}}{{{value}}}\n"
    return new_text


def main():
    p=argparse.ArgumentParser(); p.add_argument('--benchmark_stats', required=True); p.add_argument('--ours_metrics', required=True); p.add_argument('--agreement_report', required=True); p.add_argument('--main_tex', required=True); p.add_argument('--output_tex', required=True); args=p.parse_args()
    benchmark_stats=load_json(Path(args.benchmark_stats)); ours_metrics=load_json(Path(args.ours_metrics)); agreement_report=load_json(Path(args.agreement_report))
    replacements=build_replacements(benchmark_stats, ours_metrics, agreement_report)
    text=Path(args.main_tex).read_text(encoding='utf-8')
    for macro, value in replacements.items():
        text=replace_newcommand(text, macro, value)
    output=Path(args.output_tex); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text, encoding='utf-8')
    print('Updated macros:'); [print(f'{k} = {v}') for k,v in replacements.items()]; print(f'Wrote updated TeX to {output}')
if __name__=='__main__': main()
