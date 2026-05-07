import argparse, subprocess
from pathlib import Path

def run(cmd):
    print('>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--project_root', required=True); p.add_argument('--manifest_csv', required=True); p.add_argument('--gold_train', required=True); p.add_argument('--gold_dev', required=True); p.add_argument('--gold_test', required=True); p.add_argument('--rule_pred', required=True); p.add_argument('--llm_pred', required=True); p.add_argument('--hybrid_pred', required=True); p.add_argument('--ours_pred', required=True); p.add_argument('--challenge_pages', required=True); p.add_argument('--annotator_a', required=True); p.add_argument('--annotator_b', required=True); p.add_argument('--adjudicated', required=True); p.add_argument('--ours_wo_norm_pred'); p.add_argument('--ours_wo_align_pred'); p.add_argument('--ours_wo_rerank_pred'); p.add_argument('--ours_wo_ocrcorr_pred'); p.add_argument('--ours_wo_confagg_pred'); args=p.parse_args()
    root=Path(args.project_root); scripts=root/'scripts'; metrics_dir=root/'outputs'/'metrics'; tables_csv_dir=root/'outputs'/'tables'; tables_tex_dir=root/'paper'/'tables'; logs_dir=root/'outputs'/'logs';
    [d.mkdir(parents=True, exist_ok=True) for d in [metrics_dir, tables_csv_dir, tables_tex_dir, logs_dir]]
    py='python'
    run([py, str(scripts/'compute_stats.py'), '--manifest_csv', args.manifest_csv, '--gold_train', args.gold_train, '--gold_dev', args.gold_dev, '--gold_test', args.gold_test, '--output_json', str(metrics_dir/'benchmark_stats.json'), '--output_csv', str(tables_csv_dir/'table1_stats.csv')])
    for short_name, display_name, pred_path in [('rule','Rule-based',args.rule_pred),('llm','LLM-only',args.llm_pred),('hybrid','Hybrid baseline',args.hybrid_pred),('ours','Ours',args.ours_pred)]:
        run([py, str(scripts/'eval_all.py'), '--gold', args.gold_test, '--pred', pred_path, '--output', str(metrics_dir/f'{short_name}_test_all.json'), '--system_name', display_name])
    run([py, str(scripts/'export_tables.py'), '--benchmark_stats', str(metrics_dir/'benchmark_stats.json'), '--rule_metrics', str(metrics_dir/'rule_test_all.json'), '--llm_metrics', str(metrics_dir/'llm_test_all.json'), '--hybrid_metrics', str(metrics_dir/'hybrid_test_all.json'), '--ours_metrics', str(metrics_dir/'ours_test_all.json'), '--output_dir_csv', str(tables_csv_dir), '--output_dir_tex', str(tables_tex_dir)])
    run([py, str(scripts/'eval_by_type.py'), '--gold', args.gold_test, '--pred', args.ours_pred, '--output_csv', str(tables_csv_dir/'table8_by_type.csv'), '--output_tex', str(tables_tex_dir/'table8_by_type.tex')])
    run([py, str(scripts/'eval_challenge_subset.py'), '--gold', args.gold_test, '--challenge_pages', args.challenge_pages, '--rule_pred', args.rule_pred, '--llm_pred', args.llm_pred, '--hybrid_pred', args.hybrid_pred, '--ours_pred', args.ours_pred, '--output_csv', str(tables_csv_dir/'table9_challenge.csv'), '--output_tex', str(tables_tex_dir/'table9_challenge.tex')])
    run([py, str(scripts/'eval_agreement.py'), '--annotator_a', args.annotator_a, '--annotator_b', args.annotator_b, '--adjudicated', args.adjudicated, '--output_json', str(metrics_dir/'agreement_report.json'), '--output_csv', str(tables_csv_dir/'table10_agreement.csv'), '--output_tex', str(tables_tex_dir/'table10_agreement.tex')])
    run([py, str(scripts/'compute_descriptive_stats.py'), '--gold_train', args.gold_train, '--gold_dev', args.gold_dev, '--gold_test', args.gold_test, '--output_json', str(metrics_dir/'descriptive_stats.json'), '--output_csv', str(tables_csv_dir/'table11_descriptive.csv'), '--output_tex', str(tables_tex_dir/'table11_descriptive.tex')])
    ablation_args=[args.ours_wo_norm_pred,args.ours_wo_align_pred,args.ours_wo_rerank_pred,args.ours_wo_ocrcorr_pred,args.ours_wo_confagg_pred]
    if all(ablation_args):
        for short_name, pred_path in [('ours_wo_norm',args.ours_wo_norm_pred),('ours_wo_align',args.ours_wo_align_pred),('ours_wo_rerank',args.ours_wo_rerank_pred),('ours_wo_ocrcorr',args.ours_wo_ocrcorr_pred),('ours_wo_confagg',args.ours_wo_confagg_pred)]:
            run([py, str(scripts/'eval_all.py'), '--gold', args.gold_test, '--pred', pred_path, '--output', str(metrics_dir/f'{short_name}_test_all.json'), '--system_name', short_name])
        run([py, str(scripts/'eval_ablation.py'), '--full_metrics', str(metrics_dir/'ours_test_all.json'), '--wo_normalization', str(metrics_dir/'ours_wo_norm_test_all.json'), '--wo_alignment', str(metrics_dir/'ours_wo_align_test_all.json'), '--wo_boundary_reranking', str(metrics_dir/'ours_wo_rerank_test_all.json'), '--wo_ocr_correction', str(metrics_dir/'ours_wo_ocrcorr_test_all.json'), '--wo_confidence_aggregation', str(metrics_dir/'ours_wo_confagg_test_all.json'), '--output_csv', str(tables_csv_dir/'table3_ablation.csv'), '--output_tex', str(tables_tex_dir/'table3_ablation.tex')])
    else:
        print('>> Skip ablation: not all ablation predictions are provided.')
    required=[tables_tex_dir/'table1_stats.tex', tables_tex_dir/'table2_main.tex', tables_tex_dir/'table4_downstream.tex', tables_tex_dir/'table8_by_type.tex', tables_tex_dir/'table9_challenge.tex', tables_tex_dir/'table10_agreement.tex', tables_tex_dir/'table11_descriptive.tex']
    if (tables_tex_dir/'table3_ablation.tex').exists(): required.append(tables_tex_dir/'table3_ablation.tex')
    run([py, str(scripts/'check_consistency.py'), '--benchmark_stats', str(metrics_dir/'benchmark_stats.json'), '--rule_metrics', str(metrics_dir/'rule_test_all.json'), '--llm_metrics', str(metrics_dir/'llm_test_all.json'), '--hybrid_metrics', str(metrics_dir/'hybrid_test_all.json'), '--ours_metrics', str(metrics_dir/'ours_test_all.json'), '--main_tex', str(root/'paper'/'main.tex'), '--required_table_files', *[str(p) for p in required], '--output_report', str(logs_dir/'consistency_report.json')])
    print('Pipeline completed.')
if __name__=='__main__': main()
